"""
train_wav2vec.py — XGBoost + KNN sobre embeddings Wav2Vec2 (PCA dinámico).

Pipeline por modelo y actividad:
  1. Carga embeddings_wav2vec.csv (grabaciones × 1024 dim + metadata)
  2. Alinea con split paciente-nivel de train_80.csv / holdout_20.csv
  3. Split interno de PACIENTES 80/20:
       - cv_patients  (80%) → StratifiedGroupKFold 5-fold
       - val_patients (20%) → validacion interna (nivel paciente)
  4. Preproceso: StandardScaler → PCA(n_components=varianza_umbral)
       El número de componentes se selecciona automáticamente hasta
       alcanzar el umbral de varianza explicada acumulada (default: 95%).
  5. Umbral optimo: criterio de Youden sobre prob. agregadas por paciente
  6. Evaluacion sobre holdout externo (nivel paciente)

Salidas:
  models/wav2vec/{XGBoost|KNN}/{run}/{actividad}/  → model.pkl, scaler.pkl, pca.pkl, threshold.pkl, emb_cols.pkl
  reports/wav2vec/{XGBoost|KNN}/{run}/{actividad}/ → 6 figuras
  reports/wav2vec/{XGBoost|KNN}/{run}/comparison_4models.png
  logs/wav2vec/train_{run}_{timestamp}.log

Uso:
  uv run python -m src.models.train_wav2vec --run baseline
  uv run python -m src.models.train_wav2vec --run baseline --pca-variance 0.99
"""

import argparse
import sys
import warnings
import joblib
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split, StratifiedGroupKFold,
    RandomizedSearchCV, GridSearchCV,
    learning_curve, cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    recall_score, precision_score, f1_score,
    accuracy_score, balanced_accuracy_score,
)
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.config import SEED, PALETTE, apply_style
from src.utils.results_logger import save_run_json

warnings.filterwarnings('ignore')

# ─── Rutas ───────────────────────────────────────────────────────────────────
EMBEDDINGS_CSV   = Path("data/processed/combined/embeddings_wav2vec.csv")
TRAIN_CSV        = Path("data/processed/combined/train_80.csv")
HOLDOUT_CSV      = Path("data/processed/combined/holdout_20.csv")

REPORTS_BASE_XGB = Path("reports/wav2vec/XGBoost")
MODELS_BASE_XGB  = Path("models/wav2vec/XGBoost")
REPORTS_BASE_KNN = Path("reports/wav2vec/KNN")
MODELS_BASE_KNN  = Path("models/wav2vec/KNN")
LOG_DIR          = Path("logs/wav2vec")

ACTIVITIES: dict[str, list[str]] = {
    'vocal':      ['vocal'],
    'frase':      ['frase'],
    'espontanea': ['espontanea'],
    'all':        ['vocal', 'frase', 'espontanea'],
}

# Poblados dinamicamente en main()
EMB_COLS:     list[str] = []
PCA_VARIANCE: float     = 0.95   # fracción de varianza explicada acumulada


# =============================================================================
# TEE
# =============================================================================
class Tee:
    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._log_file = None
        self._original = sys.__stdout__

    def __enter__(self):
        self._log_file = open(self._log_path, 'w', encoding='utf-8')
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._original
        if self._log_file:
            self._log_file.close()

    def write(self, text: str):
        self._original.write(text)
        self._original.flush()
        self._log_file.write(text)
        self._log_file.flush()

    def flush(self):
        self._original.flush()
        self._log_file.flush()


# =============================================================================
# CARGA Y UTILIDADES COMPARTIDAS
# =============================================================================
def _cargar_split_paths(csv_path: Path) -> set:
    return set(pd.read_csv(csv_path)['audio_path'])


def cargar_grabaciones(emb_df: pd.DataFrame, split_paths: set,
                       activity_types: list[str]) -> pd.DataFrame:
    df = emb_df[emb_df['audio_path'].isin(split_paths)].copy()
    df = df[df['activity_type'].isin(activity_types)]
    df = df.dropna(subset=EMB_COLS[:5])
    return df.reset_index(drop=True)


def _agregar_proba_por_paciente(df: pd.DataFrame, y_proba: np.ndarray) -> pd.DataFrame:
    tmp = df[['ID_Paciente', 'Target']].copy()
    tmp['proba'] = y_proba
    return tmp.groupby('ID_Paciente').agg(
        proba=('proba', 'mean'),
        Target=('Target', 'first'),
    ).reset_index()


def _calcular_metricas(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr_, tpr_, _  = roc_curve(y_true, y_proba)
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall":            recall_score(y_true, y_pred),
        "specificity":       tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "f1":                f1_score(y_true, y_pred),
        "roc_auc":           auc(fpr_, tpr_),
    }


# =============================================================================
# XGBOOST TRAINER
# =============================================================================
class XGBoostWav2VecTrainer:

    def __init__(self, activity_name: str, cv_folds: int = 5, n_iter: int = 60):
        self.activity_name = activity_name
        self.cv_folds      = cv_folds
        self.n_iter        = n_iter

        self.models_dir  = MODELS_BASE_XGB  / activity_name
        self.reports_dir = REPORTS_BASE_XGB / activity_name
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.scaler            = None
        self.pca               = None
        self.best_model        = None
        self.best_params_      = {}
        self.optimal_threshold = 0.5
        self.metrics_internal_ = {}
        self.metrics_external_ = {}
        self._xgb_evals        = None
        self._X_val_pca        = None
        self._pca_full_variance: np.ndarray | None = None  # curva 1024-dim para el plot

        self._y_val = self._y_pred_val = self._y_proba_val = None
        self._y_ext = self._y_pred_ext = self._y_proba_ext = None

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(df[EMB_COLS]))

    def fit(self, df_train: pd.DataFrame, df_holdout: pd.DataFrame) -> "XGBoostWav2VecTrainer":
        n_pac_tr = df_train['ID_Paciente'].nunique()
        n_pac_ho = df_holdout['ID_Paciente'].nunique()

        print(f"\n{'='*60}")
        print(f"  ENTRENAMIENTO — XGBoost+PCA — [{self.activity_name.upper()}]")
        print(f"{'='*60}")
        print(f"  Train:   {len(df_train):>5} grab. / {n_pac_tr} pac.")
        print(f"  Holdout: {len(df_holdout):>5} grab. / {n_pac_ho} pac.")
        print(df_train.groupby(['Dataset', 'Target']).size()
              .rename('n').reset_index().to_string(index=False))

        unique_pats = df_train[['ID_Paciente', 'Target']].drop_duplicates()
        pat_cv_idx, pat_val_idx = train_test_split(
            unique_pats['ID_Paciente'],
            test_size=0.20, stratify=unique_pats['Target'], random_state=SEED,
        )
        df_cv  = df_train[df_train['ID_Paciente'].isin(pat_cv_idx)].copy()
        df_val = df_train[df_train['ID_Paciente'].isin(pat_val_idx)].copy()

        X_cv   = df_cv[EMB_COLS]
        y_cv   = df_cv['Target']
        groups = df_cv['ID_Paciente'].values

        ratio_clases = (y_cv == 0).sum() / max((y_cv == 1).sum(), 1)
        print(f"\n  CV-train: {len(df_cv)} grab. / {pat_cv_idx.nunique()} pac.")
        print(f"  Val interna: {len(df_val)} grab. / {pat_val_idx.nunique()} pac.")
        print(f"  PCA varianza umbral: {PCA_VARIANCE:.0%}  |  ratio clases: {ratio_clases:.3f}")

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=PCA_VARIANCE, random_state=SEED)),
            ("xgb",   xgb.XGBClassifier(random_state=SEED, eval_metric='logloss')),
        ])
        param_dist = {
            'xgb__n_estimators':     [100, 200, 300],
            'xgb__max_depth':        [2, 3, 4],
            'xgb__learning_rate':    [0.01, 0.05, 0.1],
            'xgb__gamma':            [0, 0.1, 0.5],
            'xgb__subsample':        [0.6, 0.8, 1.0],
            'xgb__colsample_bytree': [0.6, 0.8, 1.0],
            'xgb__min_child_weight': [1, 3, 5],
            'xgb__reg_alpha':        [0, 0.1, 1.0],
            'xgb__reg_lambda':       [1.0, 3.0, 10.0],
            'xgb__scale_pos_weight': [ratio_clases],
        }
        cv_strategy = StratifiedGroupKFold(n_splits=self.cv_folds, shuffle=True, random_state=SEED)
        search = RandomizedSearchCV(
            pipe, param_dist, n_iter=self.n_iter, cv=cv_strategy,
            scoring='balanced_accuracy', n_jobs=-1, verbose=1, random_state=SEED,
        )
        print(f"\n  RandomizedSearchCV ({self.n_iter} iter, {self.cv_folds}-fold)...")
        search.fit(X_cv, y_cv, groups=groups)

        self.best_params_ = {
            k.replace('xgb__', ''): v
            for k, v in search.best_params_.items()
            if k.startswith('xgb__')
        }
        print(f"\n  Mejores hiperparametros:")
        for k, v in self.best_params_.items():
            print(f"    {k}: {v}")
        print(f"  Balanced Accuracy CV: {search.best_score_:.4f}")

        # Refit con scaler+pca separados sobre todos los cv_patients
        self.scaler   = StandardScaler()
        X_cv_sc       = self.scaler.fit_transform(X_cv)

        # PCA completo (1024 dim) solo para la curva de varianza del plot
        pca_full = PCA(n_components=None, random_state=SEED).fit(X_cv_sc)
        self._pca_full_variance = pca_full.explained_variance_ratio_

        # PCA real con umbral de varianza — n_components_ se selecciona automáticamente
        self.pca      = PCA(n_components=PCA_VARIANCE, random_state=SEED)
        X_cv_pca      = self.pca.fit_transform(X_cv_sc)
        print(f"  PCA seleccionó {self.pca.n_components_} componentes "
              f"({PCA_VARIANCE:.0%} varianza, de {X_cv_sc.shape[1]} originales)")
        X_val_pca     = self._transform(df_val)
        y_val_rec     = df_val['Target'].values

        self.best_model = xgb.XGBClassifier(
            **self.best_params_, random_state=SEED, eval_metric='logloss'
        )
        self.best_model.fit(
            X_cv_pca, y_cv,
            eval_set=[(X_cv_pca, y_cv), (X_val_pca, y_val_rec)],
            early_stopping_rounds=20, verbose=False,
        )
        self._xgb_evals = self.best_model.evals_result()
        self._X_val_pca = X_val_pca

        # Umbral optimo (Youden, nivel paciente)
        pat_val_df = _agregar_proba_por_paciente(
            df_val, self.best_model.predict_proba(X_val_pca)[:, 1]
        )
        fpr, tpr, ths = roc_curve(pat_val_df['Target'], pat_val_df['proba'])
        finite = np.isfinite(ths)
        self.optimal_threshold = float(ths[finite][np.argmax((tpr - fpr)[finite])]) if finite.any() else 0.5
        print(f"\n  Umbral optimo (Youden, paciente-nivel): {self.optimal_threshold:.4f}")

        self._evaluate_pac(df_val,     X_val_pca,                tag="VAL INTERNA (paciente)",   internal=True)
        self._evaluate_pac(df_holdout, self._transform(df_holdout), tag="HOLDOUT EXTERNO (paciente)", internal=False)

        self._save()
        self._plot_all()
        return self

    def _evaluate_pac(self, df: pd.DataFrame, X_pca: np.ndarray, tag: str, internal: bool):
        y_proba_rec = self.best_model.predict_proba(X_pca)[:, 1]
        pat_df      = _agregar_proba_por_paciente(df, y_proba_rec)
        y_true      = pat_df['Target'].values
        y_proba     = pat_df['proba'].values
        y_pred      = (y_proba >= self.optimal_threshold).astype(int)
        metrics     = _calcular_metricas(y_true, y_pred, y_proba)

        if internal:
            self.metrics_internal_             = metrics
            self._y_val, self._y_pred_val, self._y_proba_val = y_true, y_pred, y_proba
        else:
            self.metrics_external_             = metrics
            self._y_ext, self._y_pred_ext, self._y_proba_ext = y_true, y_pred, y_proba

        print(f"\n  {'─'*52}")
        print(f"  METRICAS {tag}  ({len(pat_df)} pac., umbral={self.optimal_threshold:.3f})")
        print(f"  {'─'*52}")
        for k, v in metrics.items():
            print(f"  {k:22s}: {v:.4f}")
        print(f"  {'─'*52}")

    def _save(self):
        base = self.models_dir
        joblib.dump(self.best_model,        base / "model.pkl")
        joblib.dump(self.scaler,            base / "scaler.pkl")
        joblib.dump(self.pca,               base / "pca.pkl")
        joblib.dump(self.optimal_threshold, base / "threshold.pkl")
        joblib.dump(EMB_COLS,               base / "emb_cols.pkl")
        print(f"\n  Artefactos guardados en: {base}/")

    def _plot_all(self):
        apply_style()
        self._plot_confusion()
        self._plot_roc()
        self._plot_loss()
        self._plot_pca_variance()
        self._plot_importance()
        self._plot_summary()

    def _plot_confusion(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Matriz de Confusion — XGBoost+PCA [{self.activity_name}]",
                     fontsize=14, fontweight="bold", y=1.02)
        for ax, (y_true, y_pred, subtitle) in zip(axes, [
            (self._y_val, self._y_pred_val, "Val interna"),
            (self._y_ext, self._y_pred_ext, "Holdout externo"),
        ]):
            cm  = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                        xticklabels=["Control", "Parkinson"],
                        yticklabels=["Control", "Parkinson"],
                        cbar=False, annot_kws={"size": 12, "weight": "bold"})
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ax.set_title(f"{subtitle}\nSensib={sens:.3f} | Especif={spec:.3f}")
            ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
        plt.tight_layout()
        self._save_fig("1_confusion_matrix")

    def _plot_roc(self):
        fig, ax = plt.subplots(figsize=(7, 6))
        for y_true, y_proba, label, color in [
            (self._y_val, self._y_proba_val, "Val interna",     PALETTE["accent1"]),
            (self._y_ext, self._y_proba_ext, "Holdout externo", PALETTE["accent3"]),
        ]:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{label} (AUC={auc_val:.4f})")
            ax.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], s=100, color=color, zorder=5)
        ax.plot([0, 1], [0, 1], ":", color=PALETTE["subtext"], lw=2)
        ax.set_xlabel("FPR (1-Especificidad)"); ax.set_ylabel("TPR (Sensibilidad)")
        ax.set_title(f"Curva ROC — XGBoost+PCA [{self.activity_name}]", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        self._save_fig("2_roc_curve")

    def _plot_loss(self):
        if not self._xgb_evals:
            return
        tr      = self._xgb_evals["validation_0"]["logloss"]
        te      = self._xgb_evals["validation_1"]["logloss"]
        ep      = np.arange(1, len(tr) + 1)
        best_ep = int(np.argmin(te)) + 1
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ep, tr, color=PALETTE["accent1"], lw=2, label="Train logloss")
        ax.plot(ep, te, color=PALETTE["accent3"], lw=2, label="Val logloss")
        ax.axvline(best_ep, color=PALETTE["accent2"], lw=2, ls="--", label=f"Mejor iter ({best_ep})")
        ax.set_xlabel("Iteracion"); ax.set_ylabel("Logloss")
        ax.set_title(f"Curvas de perdida — XGBoost+PCA [{self.activity_name}]", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        self._save_fig("3_loss_curves")

    def _plot_pca_variance(self):
        # Curva completa (todos los 1024 valores extraídos)
        exp_var = self._pca_full_variance
        cum_var = np.cumsum(exp_var)
        pcs     = np.arange(1, len(exp_var) + 1)
        n_sel   = self.pca.n_components_   # componentes seleccionados automáticamente

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Scree plot completo: seleccionados en color, resto en gris
        colors = [PALETTE["accent1"] if i < n_sel else PALETTE["subtext"] for i in range(len(pcs))]
        axes[0].bar(pcs, exp_var * 100, color=colors, alpha=0.85)
        axes[0].axvline(n_sel + 0.5, color=PALETTE["accent3"], lw=2, ls="--",
                        label=f"Corte: {n_sel} componentes")
        axes[0].set_xlabel("Componente principal (de 1024 embeddings)")
        axes[0].set_ylabel("Varianza explicada (%)")
        axes[0].set_title(f"Scree plot completo — 1024 dimensiones Wav2Vec2")
        axes[0].legend(fontsize=9)

        # Curva acumulada completa con umbral y corte
        axes[1].plot(pcs, cum_var * 100, color=PALETTE["accent1"], lw=2)
        axes[1].axhline(PCA_VARIANCE * 100, color=PALETTE["accent3"], lw=1.5, ls="--",
                        label=f"Umbral {PCA_VARIANCE:.0%} varianza")
        axes[1].axvline(n_sel, color=PALETTE["accent4"], lw=2, ls=":",
                        label=f"{n_sel} componentes seleccionados")
        axes[1].scatter([n_sel], [cum_var[n_sel - 1] * 100],
                        color=PALETTE["accent4"], s=80, zorder=5)
        axes[1].set_xlabel("N componentes")
        axes[1].set_ylabel("Varianza acumulada (%)")
        axes[1].set_title(f"Varianza acumulada — umbral {PCA_VARIANCE:.0%} → {n_sel} PCs")
        axes[1].legend(fontsize=9)

        fig.suptitle(f"Analisis PCA dinamico — XGBoost [{self.activity_name}]",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save_fig("4_pca_variance")

    def _plot_importance(self):
        imp = self.best_model.feature_importances_
        idx = np.argsort(imp)[-20:]
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh([f"PC{i+1}" for i in idx], imp[idx], color=PALETTE["accent1"], alpha=0.9)
        for i in range(1, min(4, len(bars) + 1)):
            bars[-i].set_color(PALETTE["accent3"])
        ax.set_xlabel("Importancia (Gain)")
        ax.set_title(f"Top 20 Componentes — XGBoost+PCA [{self.activity_name}]", fontweight="bold")
        plt.tight_layout()
        self._save_fig("5_feature_importance")

    def _plot_summary(self):
        m_i, m_e = self.metrics_internal_, self.metrics_external_
        fig = plt.figure(figsize=(13, 5.5))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.8])
        cats   = ["Recall\n(Sensib.)", "Especif.", "Precision", "F1", "AUC-ROC"]
        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist() + [0]
        ax_r   = fig.add_subplot(gs[0], polar=True)
        for m, label, color in [(m_i, "Val interna", PALETTE["accent1"]),
                                 (m_e, "Holdout externo", PALETTE["accent3"])]:
            vr = [m["recall"], m["specificity"], m["precision"], m["f1"], m["roc_auc"]]
            vr = vr + [vr[0]]
            ax_r.plot(angles, vr, color=color, lw=2, label=label)
            ax_r.fill(angles, vr, color=color, alpha=0.1)
        ax_r.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=9)
        ax_r.set_ylim(0, 1)
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        ax_r.set_title(f"[{self.activity_name}]", fontweight="bold", pad=20)
        ax_t = fig.add_subplot(gs[1]); ax_t.axis("off")
        rows_data = [
            ["Metrica",            "Val interna",                     "Holdout externo"],
            ["Balanced Accuracy",  f"{m_i['balanced_accuracy']:.4f}", f"{m_e['balanced_accuracy']:.4f}"],
            ["Recall (PD)",        f"{m_i['recall']:.4f}",            f"{m_e['recall']:.4f}"],
            ["Especificidad (HC)", f"{m_i['specificity']:.4f}",       f"{m_e['specificity']:.4f}"],
            ["F1-Score",           f"{m_i['f1']:.4f}",                f"{m_e['f1']:.4f}"],
            ["AUC-ROC",            f"{m_i['roc_auc']:.4f}",           f"{m_e['roc_auc']:.4f}"],
            ["Accuracy",           f"{m_i['accuracy']:.4f}",          f"{m_e['accuracy']:.4f}"],
            ["Umbral Youden",      f"{self.optimal_threshold:.4f}",    "—"],
            ["PCA varianza umbral", f"{PCA_VARIANCE:.0%}",              "—"],
            ["PCA componentes",    f"{self.pca.n_components_}",         "—"],
        ]
        tbl = ax_t.table(cellText=rows_data, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(PALETTE["accent1"]); cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor(PALETTE["panel"])
        plt.subplots_adjust(bottom=0.1)
        self._save_fig("6_metrics_summary")

    def _save_fig(self, name: str, dpi: int = 300):
        path = self.reports_dir / f"{name}.png"
        plt.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close()
        print(f"  Guardado: {path}")


# =============================================================================
# KNN TRAINER
# =============================================================================
class KNNWav2VecTrainer:

    def __init__(self, activity_name: str, cv_folds: int = 5):
        self.activity_name = activity_name
        self.cv_folds      = cv_folds

        self.models_dir  = MODELS_BASE_KNN  / activity_name
        self.reports_dir = REPORTS_BASE_KNN / activity_name
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.scaler            = None
        self.pca               = None
        self.best_model        = None
        self.best_params_      = {}
        self.optimal_threshold = 0.5
        self.metrics_internal_ = {}
        self.metrics_external_ = {}
        self._X_cv_pca         = None
        self._y_cv             = None
        self._pca_full_variance: np.ndarray | None = None  # curva 1024-dim para el plot

        self._y_val = self._y_pred_val = self._y_proba_val = None
        self._y_ext = self._y_pred_ext = self._y_proba_ext = None

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(df[EMB_COLS]))

    def fit(self, df_train: pd.DataFrame, df_holdout: pd.DataFrame) -> "KNNWav2VecTrainer":
        n_pac_tr = df_train['ID_Paciente'].nunique()
        n_pac_ho = df_holdout['ID_Paciente'].nunique()

        print(f"\n{'='*60}")
        print(f"  ENTRENAMIENTO — KNN+PCA — [{self.activity_name.upper()}]")
        print(f"{'='*60}")
        print(f"  Train:   {len(df_train):>5} grab. / {n_pac_tr} pac.")
        print(f"  Holdout: {len(df_holdout):>5} grab. / {n_pac_ho} pac.")
        print(df_train.groupby(['Dataset', 'Target']).size()
              .rename('n').reset_index().to_string(index=False))

        unique_pats = df_train[['ID_Paciente', 'Target']].drop_duplicates()
        pat_cv_idx, pat_val_idx = train_test_split(
            unique_pats['ID_Paciente'],
            test_size=0.20, stratify=unique_pats['Target'], random_state=SEED,
        )
        df_cv  = df_train[df_train['ID_Paciente'].isin(pat_cv_idx)].copy()
        df_val = df_train[df_train['ID_Paciente'].isin(pat_val_idx)].copy()

        X_cv   = df_cv[EMB_COLS]
        y_cv   = df_cv['Target']
        groups = df_cv['ID_Paciente'].values

        print(f"\n  CV-train: {len(df_cv)} grab. / {pat_cv_idx.nunique()} pac.")
        print(f"  Val interna: {len(df_val)} grab. / {pat_val_idx.nunique()} pac.")
        print(f"  PCA varianza umbral: {PCA_VARIANCE:.0%}")

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=PCA_VARIANCE, random_state=SEED)),
            ("knn",   KNeighborsClassifier(n_jobs=-1)),
        ])
        param_grid = {
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'knn__weights':     ['uniform', 'distance'],
            'knn__metric':      ['euclidean', 'manhattan', 'minkowski'],
        }
        cv_strategy = StratifiedGroupKFold(n_splits=self.cv_folds, shuffle=True, random_state=SEED)
        search = GridSearchCV(
            pipe, param_grid, cv=cv_strategy,
            scoring='balanced_accuracy', n_jobs=-1, verbose=1,
        )
        print(f"\n  GridSearchCV ({self.cv_folds}-fold StratifiedGroupKFold)...")
        search.fit(X_cv, y_cv, groups=groups)

        self.best_params_ = {
            k.replace('knn__', ''): v
            for k, v in search.best_params_.items()
            if k.startswith('knn__')
        }
        print(f"\n  Mejores hiperparametros:")
        for k, v in self.best_params_.items():
            print(f"    {k}: {v}")
        print(f"  Balanced Accuracy CV: {search.best_score_:.4f}")

        # Refit con scaler+pca separados sobre todos los cv_patients
        self.scaler   = StandardScaler()
        X_cv_sc       = self.scaler.fit_transform(X_cv)

        # PCA completo (1024 dim) solo para la curva de varianza del plot
        pca_full = PCA(n_components=None, random_state=SEED).fit(X_cv_sc)
        self._pca_full_variance = pca_full.explained_variance_ratio_

        # PCA real con umbral de varianza — n_components_ se selecciona automáticamente
        self.pca      = PCA(n_components=PCA_VARIANCE, random_state=SEED)
        X_cv_pca      = self.pca.fit_transform(X_cv_sc)
        print(f"  PCA seleccionó {self.pca.n_components_} componentes "
              f"({PCA_VARIANCE:.0%} varianza, de {X_cv_sc.shape[1]} originales)")
        X_val_pca     = self._transform(df_val)

        self.best_model = KNeighborsClassifier(**self.best_params_, n_jobs=-1)
        self.best_model.fit(X_cv_pca, y_cv)

        self._X_cv_pca = X_cv_pca
        self._y_cv     = y_cv

        # Umbral optimo (Youden, nivel paciente)
        pat_val_df = _agregar_proba_por_paciente(
            df_val, self.best_model.predict_proba(X_val_pca)[:, 1]
        )
        fpr, tpr, ths = roc_curve(pat_val_df['Target'], pat_val_df['proba'])
        finite = np.isfinite(ths)
        self.optimal_threshold = float(ths[finite][np.argmax((tpr - fpr)[finite])]) if finite.any() else 0.5
        print(f"\n  Umbral optimo (Youden, paciente-nivel): {self.optimal_threshold:.4f}")

        self._evaluate_pac(df_val,     X_val_pca,                tag="VAL INTERNA (paciente)",   internal=True)
        self._evaluate_pac(df_holdout, self._transform(df_holdout), tag="HOLDOUT EXTERNO (paciente)", internal=False)

        self._save()
        apply_style()
        self._plot_all(X_cv_pca, y_cv)
        return self

    def _evaluate_pac(self, df: pd.DataFrame, X_pca: np.ndarray, tag: str, internal: bool):
        y_proba_rec = self.best_model.predict_proba(X_pca)[:, 1]
        pat_df      = _agregar_proba_por_paciente(df, y_proba_rec)
        y_true      = pat_df['Target'].values
        y_proba     = pat_df['proba'].values
        y_pred      = (y_proba >= self.optimal_threshold).astype(int)
        metrics     = _calcular_metricas(y_true, y_pred, y_proba)

        if internal:
            self.metrics_internal_                           = metrics
            self._y_val, self._y_pred_val, self._y_proba_val = y_true, y_pred, y_proba
        else:
            self.metrics_external_                           = metrics
            self._y_ext, self._y_pred_ext, self._y_proba_ext = y_true, y_pred, y_proba

        print(f"\n  {'─'*52}")
        print(f"  METRICAS {tag}  ({len(pat_df)} pac., umbral={self.optimal_threshold:.3f})")
        print(f"  {'─'*52}")
        for k, v in metrics.items():
            print(f"  {k:22s}: {v:.4f}")
        print(f"  {'─'*52}")

    def _save(self):
        base = self.models_dir
        joblib.dump(self.best_model,        base / "model.pkl")
        joblib.dump(self.scaler,            base / "scaler.pkl")
        joblib.dump(self.pca,               base / "pca.pkl")
        joblib.dump(self.optimal_threshold, base / "threshold.pkl")
        joblib.dump(EMB_COLS,               base / "emb_cols.pkl")
        print(f"\n  Artefactos guardados en: {base}/")

    def _plot_all(self, X_cv_pca: np.ndarray, y_cv: pd.Series):
        self._plot_confusion()
        self._plot_roc()
        self._plot_k_analysis(X_cv_pca, y_cv)
        self._plot_pca_variance()
        self._plot_permutation_importance(X_cv_pca, y_cv)
        self._plot_summary()

    def _plot_confusion(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Matriz de Confusion — KNN+PCA [{self.activity_name}]",
                     fontsize=14, fontweight="bold", y=1.02)
        for ax, (y_true, y_pred, subtitle) in zip(axes, [
            (self._y_val, self._y_pred_val, "Val interna"),
            (self._y_ext, self._y_pred_ext, "Holdout externo"),
        ]):
            cm  = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                        xticklabels=["Control", "Parkinson"],
                        yticklabels=["Control", "Parkinson"],
                        cbar=False, annot_kws={"size": 12, "weight": "bold"})
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ax.set_title(f"{subtitle}\nSensib={sens:.3f} | Especif={spec:.3f}")
            ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
        plt.tight_layout()
        self._save_fig("1_confusion_matrix")

    def _plot_roc(self):
        fig, ax = plt.subplots(figsize=(7, 6))
        for y_true, y_proba, label, color in [
            (self._y_val, self._y_proba_val, "Val interna",     PALETTE["accent1"]),
            (self._y_ext, self._y_proba_ext, "Holdout externo", PALETTE["accent3"]),
        ]:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{label} (AUC={auc_val:.4f})")
            ax.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], s=100, color=color, zorder=5)
        ax.plot([0, 1], [0, 1], ":", color=PALETTE["subtext"], lw=2)
        ax.set_xlabel("FPR (1-Especificidad)"); ax.set_ylabel("TPR (Sensibilidad)")
        ax.set_title(f"Curva ROC — KNN+PCA [{self.activity_name}]", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        self._save_fig("2_roc_curve")

    def _plot_k_analysis(self, X_cv_pca: np.ndarray, y_cv: pd.Series):
        k_range = list(range(1, 22, 2))
        ba_means, ba_stds = [], []
        for k in k_range:
            scores = cross_val_score(
                KNeighborsClassifier(n_neighbors=k,
                                     weights=self.best_params_['weights'],
                                     metric=self.best_params_['metric'], n_jobs=-1),
                X_cv_pca, y_cv, cv=self.cv_folds, scoring='balanced_accuracy', n_jobs=-1,
            )
            ba_means.append(scores.mean()); ba_stds.append(scores.std())

        ba_means = np.array(ba_means); ba_stds = np.array(ba_stds)
        best_k   = k_range[int(np.argmax(ba_means))]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(k_range, ba_means - ba_stds, ba_means + ba_stds,
                        alpha=0.15, color=PALETTE["accent1"])
        ax.plot(k_range, ba_means, "o-", color=PALETTE["accent1"], lw=2.5,
                label="Balanced Accuracy CV (media)")
        ax.axvline(best_k, color=PALETTE["accent3"], lw=2, ls="--", label=f"Mejor K = {best_k}")
        ax.set_xlabel("Numero de vecinos (K)"); ax.set_ylabel("Balanced Accuracy (CV)")
        ax.set_title(f"Analisis del parametro K — KNN+PCA [{self.activity_name}]", fontweight="bold")
        ax.set_xticks(k_range); ax.legend()
        plt.tight_layout()
        self._save_fig("3_k_analysis")

    def _plot_pca_variance(self):
        # Curva completa (todos los 1024 valores extraídos)
        exp_var = self._pca_full_variance
        cum_var = np.cumsum(exp_var)
        pcs     = np.arange(1, len(exp_var) + 1)
        n_sel   = self.pca.n_components_   # componentes seleccionados automáticamente

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Scree plot completo: seleccionados en color, resto en gris
        colors = [PALETTE["accent1"] if i < n_sel else PALETTE["subtext"] for i in range(len(pcs))]
        axes[0].bar(pcs, exp_var * 100, color=colors, alpha=0.85)
        axes[0].axvline(n_sel + 0.5, color=PALETTE["accent3"], lw=2, ls="--",
                        label=f"Corte: {n_sel} componentes")
        axes[0].set_xlabel("Componente principal (de 1024 embeddings)")
        axes[0].set_ylabel("Varianza explicada (%)")
        axes[0].set_title("Scree plot completo — 1024 dimensiones Wav2Vec2")
        axes[0].legend(fontsize=9)

        # Curva acumulada completa con umbral y corte
        axes[1].plot(pcs, cum_var * 100, color=PALETTE["accent1"], lw=2)
        axes[1].axhline(PCA_VARIANCE * 100, color=PALETTE["accent3"], lw=1.5, ls="--",
                        label=f"Umbral {PCA_VARIANCE:.0%} varianza")
        axes[1].axvline(n_sel, color=PALETTE["accent4"], lw=2, ls=":",
                        label=f"{n_sel} componentes seleccionados")
        axes[1].scatter([n_sel], [cum_var[n_sel - 1] * 100],
                        color=PALETTE["accent4"], s=80, zorder=5)
        axes[1].set_xlabel("N componentes")
        axes[1].set_ylabel("Varianza acumulada (%)")
        axes[1].set_title(f"Varianza acumulada — umbral {PCA_VARIANCE:.0%} → {n_sel} PCs")
        axes[1].legend(fontsize=9)

        fig.suptitle(f"Analisis PCA dinamico — KNN [{self.activity_name}]",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save_fig("4_pca_variance")

    def _plot_permutation_importance(self, X_cv_pca: np.ndarray, y_cv: pd.Series):
        result     = permutation_importance(
            self.best_model, X_cv_pca, y_cv,
            n_repeats=15, random_state=SEED, scoring='balanced_accuracy', n_jobs=-1,
        )
        sorted_idx = result.importances_mean.argsort()[-20:]
        fig, ax    = plt.subplots(figsize=(8, 6))
        bars = ax.barh(
            [f"PC{i+1}" for i in sorted_idx],
            result.importances_mean[sorted_idx],
            xerr=result.importances_std[sorted_idx],
            color=PALETTE["accent1"], alpha=0.9, capsize=4,
        )
        for i in range(1, min(4, len(bars) + 1)):
            bars[-i].set_color(PALETTE["accent3"])
        ax.set_xlabel("Disminucion media de Balanced Accuracy al permutar")
        ax.set_title(f"Permutation Importance (Top 20 PCs) — KNN+PCA [{self.activity_name}]",
                     fontweight="bold")
        plt.tight_layout()
        self._save_fig("5_permutation_importance")

    def _plot_summary(self):
        m_i, m_e = self.metrics_internal_, self.metrics_external_
        fig = plt.figure(figsize=(13, 5.5))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.8])
        cats   = ["Recall\n(Sensib.)", "Especif.", "Precision", "F1", "AUC-ROC"]
        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist() + [0]
        ax_r   = fig.add_subplot(gs[0], polar=True)
        for m, label, color in [(m_i, "Val interna", PALETTE["accent1"]),
                                 (m_e, "Holdout externo", PALETTE["accent3"])]:
            vr = [m["recall"], m["specificity"], m["precision"], m["f1"], m["roc_auc"]] + [m["recall"]]
            ax_r.plot(angles, vr, color=color, lw=2, label=label)
            ax_r.fill(angles, vr, color=color, alpha=0.1)
        ax_r.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=9)
        ax_r.set_ylim(0, 1)
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        ax_r.set_title(f"[{self.activity_name}]", fontweight="bold", pad=20)
        ax_t = fig.add_subplot(gs[1]); ax_t.axis("off")
        rows_data = [
            ["Metrica",            "Val interna",                     "Holdout externo"],
            ["Balanced Accuracy",  f"{m_i['balanced_accuracy']:.4f}", f"{m_e['balanced_accuracy']:.4f}"],
            ["Recall (PD)",        f"{m_i['recall']:.4f}",            f"{m_e['recall']:.4f}"],
            ["Especificidad (HC)", f"{m_i['specificity']:.4f}",       f"{m_e['specificity']:.4f}"],
            ["F1-Score",           f"{m_i['f1']:.4f}",                f"{m_e['f1']:.4f}"],
            ["AUC-ROC",            f"{m_i['roc_auc']:.4f}",           f"{m_e['roc_auc']:.4f}"],
            ["Accuracy",           f"{m_i['accuracy']:.4f}",          f"{m_e['accuracy']:.4f}"],
            ["Umbral Youden",      f"{self.optimal_threshold:.4f}",   "—"],
            ["K optimo",           f"{self.best_model.n_neighbors}",   "—"],
            ["PCA varianza umbral", f"{PCA_VARIANCE:.0%}",             "—"],
            ["PCA componentes",    f"{self.pca.n_components_}",        "—"],
        ]
        tbl = ax_t.table(cellText=rows_data, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(PALETTE["accent1"]); cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor(PALETTE["panel"])
        plt.subplots_adjust(bottom=0.1)
        self._save_fig("6_metrics_summary")

    def _save_fig(self, name: str, dpi: int = 300):
        path = self.reports_dir / f"{name}.png"
        plt.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close()
        print(f"  Guardado: {path}")


# =============================================================================
# COMPARACION Y RESUMEN
# =============================================================================
def plot_comparison(results: dict, title: str, out_path: Path):
    apply_style()
    METRIC_LABELS = {
        "balanced_accuracy": "Balanced Acc.",
        "roc_auc":           "AUC-ROC",
        "f1":                "F1-Score",
        "recall":            "Recall (Sensib.)",
        "specificity":       "Especificidad",
    }
    models = list(results.keys())
    x      = np.arange(len(models))
    width  = 0.14
    colors = [PALETTE["accent1"], PALETTE["accent3"], PALETTE["accent2"],
               PALETTE["accent4"], PALETTE["warn"]]
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (key, label) in enumerate(METRIC_LABELS.items()):
        vals   = [results[m]["external"][key] for m in models]
        offset = (i - len(METRIC_LABELS) / 2) * width + width / 2
        bars   = ax.bar(x + offset, vals, width, label=label, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xlabel("Tipo de actividad de habla"); ax.set_ylabel("Puntuacion")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels([m.capitalize() for m in models], fontsize=11)
    ax.set_ylim(0, 1.15); ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.5, color=PALETTE["subtext"], lw=1, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"\n  Comparacion guardada en: {out_path}")


def _print_summary_table(results: dict, model_name: str):
    print(f"\n\n{'='*68}")
    print(f"  RESUMEN — {model_name} — HOLDOUT EXTERNO")
    print(f"{'='*68}")
    print(f"  {'Actividad':12s} | {'Bal.Acc':8s} | {'AUC':8s} | {'F1':8s} | {'Recall':8s} | {'Specif.':8s}")
    print("  " + "─" * 64)
    for act, r in results.items():
        m = r["external"]
        print(f"  {act:12s} | {m['balanced_accuracy']:8.4f} | {m['roc_auc']:8.4f} | "
              f"{m['f1']:8.4f} | {m['recall']:8.4f} | {m['specificity']:8.4f}")
    print(f"{'='*68}")


# =============================================================================
# MAIN
# =============================================================================
def _age_match_neurovoz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retiene solo pacientes de NeuroVoz emparejados 1:1 por edad (±5 años).
    PC-GITA pasa sin cambios.

    Algoritmo greedy: itera sobre los PD ordenados por edad y asigna el HC
    más cercano disponible dentro de la ventana. Emparejar a nivel de paciente
    (no de grabación); todas las grabaciones del par quedan incluidas.
    """
    pcgita = df[df['Dataset'] != 'NeuroVoz'].copy()
    nv     = df[df['Dataset'] == 'NeuroVoz'].copy()

    if 'Age' not in nv.columns or nv['Age'].isna().all():
        print("  [!] Age no disponible en NeuroVoz — age-match omitido.")
        return df

    pats = nv[['ID_Paciente', 'Target', 'Age']].drop_duplicates('ID_Paciente')
    pd_pats = pats[pats['Target'] == 1].sort_values('Age').reset_index(drop=True)
    hc_pool = pats[pats['Target'] == 0].copy()

    matched_pd, matched_hc = [], []
    used_hc = set()

    for _, pd_row in pd_pats.iterrows():
        candidates = hc_pool[
            (~hc_pool['ID_Paciente'].isin(used_hc)) &
            (hc_pool['Age'] - pd_row['Age']).abs() <= 5
        ]
        if candidates.empty:
            continue
        best_hc = candidates.iloc[(candidates['Age'] - pd_row['Age']).abs().argsort()[:1]]
        matched_pd.append(pd_row['ID_Paciente'])
        matched_hc.append(best_hc.iloc[0]['ID_Paciente'])
        used_hc.add(best_hc.iloc[0]['ID_Paciente'])

    kept = set(matched_pd) | set(matched_hc)
    nv_matched = nv[nv['ID_Paciente'].isin(kept)]

    n_removed = pats.shape[0] - len(kept)
    print(f"  Age-match NeuroVoz: {len(matched_pd)} pares emparejados "
          f"({n_removed} pacientes descartados de NeuroVoz)")
    hc_ages = pats[pats['ID_Paciente'].isin(matched_hc)]['Age']
    pd_ages = pats[pats['ID_Paciente'].isin(matched_pd)]['Age']
    print(f"    HC edad: {hc_ages.mean():.1f} ± {hc_ages.std():.1f}  |  "
          f"PD edad: {pd_ages.mean():.1f} ± {pd_ages.std():.1f}")

    return pd.concat([nv_matched, pcgita], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',       default='baseline',
                        help='Nombre del run (subcarpeta de salida)')
    parser.add_argument('--pca-variance', type=float, default=0.95,
                        help='Fracción de varianza explicada acumulada para selección dinámica de PCs (default: 0.95)')
    parser.add_argument('--age-match', action='store_true',
                        help='Age-matching 1:1 (±5 años) en NeuroVoz para eliminar confound demográfico')
    args = parser.parse_args()

    global EMB_COLS, PCA_VARIANCE, REPORTS_BASE_XGB, MODELS_BASE_XGB, REPORTS_BASE_KNN, MODELS_BASE_KNN
    PCA_VARIANCE     = args.pca_variance
    REPORTS_BASE_XGB = REPORTS_BASE_XGB / args.run
    MODELS_BASE_XGB  = MODELS_BASE_XGB  / args.run
    REPORTS_BASE_KNN = REPORTS_BASE_KNN / args.run
    MODELS_BASE_KNN  = MODELS_BASE_KNN  / args.run

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_BASE_XGB.mkdir(parents=True, exist_ok=True)
    REPORTS_BASE_KNN.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = LOG_DIR / f"train_{args.run}_{timestamp}.log"

    with Tee(log_path):
        print(f"Log: {log_path}\n")

        for path in [EMBEDDINGS_CSV, TRAIN_CSV, HOLDOUT_CSV]:
            if not path.exists():
                print(f"[!] No encontrado: {path}")
                hint = ("uv run python -m src.features.embeddings"
                        if path == EMBEDDINGS_CSV else
                        "uv run python -m src.features.split_dataset")
                print(f"    Ejecuta primero: {hint}")
                return

        print(f"{'='*60}")
        print("  Cortex-AI — Embeddings Wav2Vec2 → XGBoost + KNN")
        print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  PCA varianza umbral: {PCA_VARIANCE:.0%}")
        print(f"{'='*60}")

        df_emb    = pd.read_csv(EMBEDDINGS_CSV)
        EMB_COLS[:] = [c for c in df_emb.columns if c.startswith('emb_')]

        print(f"\nEmbeddings: {len(df_emb)} grabaciones | "
              f"{df_emb['ID_Paciente'].nunique()} pacientes | "
              f"dim={len(EMB_COLS)}")

        if args.age_match:
            print("\n  [Age-match activo]")
            df_emb = _age_match_neurovoz(df_emb)
            print(f"  Tras age-match: {len(df_emb)} grabaciones | "
                  f"{df_emb['ID_Paciente'].nunique()} pacientes")

        train_paths   = _cargar_split_paths(TRAIN_CSV)
        holdout_paths = _cargar_split_paths(HOLDOUT_CSV)

        n_tr = df_emb['audio_path'].isin(train_paths).sum()
        n_ho = df_emb['audio_path'].isin(holdout_paths).sum()
        print(f"Alineacion con split: train={n_tr} | holdout={n_ho}")

        print("\nDistribucion (train):")
        print(df_emb[df_emb['audio_path'].isin(train_paths)]
              .groupby(['Dataset', 'activity_type', 'Target']).size()
              .rename('n').reset_index().to_string(index=False))

        # ── XGBOOST ──────────────────────────────────────────────────────
        print(f"\n\n{'#'*60}\n# XGBOOST + PCA\n{'#'*60}")
        results_xgb    = {}
        thresholds_xgb = {}
        for activity_name, activity_types in ACTIVITIES.items():
            print(f"\n\n{'#'*60}\n# XGB — {activity_name.upper()}\n{'#'*60}")
            df_tr = cargar_grabaciones(df_emb, train_paths,   activity_types)
            df_ho = cargar_grabaciones(df_emb, holdout_paths, activity_types)
            if df_tr['ID_Paciente'].nunique() < 20:
                print(f"  [!] Pacientes insuficientes — omitido.")
                continue
            trainer = XGBoostWav2VecTrainer(activity_name, cv_folds=5, n_iter=60)
            trainer.fit(df_tr, df_ho)
            results_xgb[activity_name] = {
                "internal": trainer.metrics_internal_,
                "external": trainer.metrics_external_,
            }
            thresholds_xgb[activity_name] = trainer.optimal_threshold

        _print_summary_table(results_xgb, "XGBoost+PCA")
        if len(results_xgb) > 1:
            plot_comparison(results_xgb,
                            f"XGBoost+PCA(var={PCA_VARIANCE:.0%}) — Holdout externo",
                            REPORTS_BASE_XGB / "comparison_4models.png")
        save_run_json("Wav2Vec-XGBoost", args.run, results_xgb, thresholds_xgb)

        # ── KNN ──────────────────────────────────────────────────────────
        print(f"\n\n{'#'*60}\n# KNN + PCA\n{'#'*60}")
        results_knn    = {}
        thresholds_knn = {}
        for activity_name, activity_types in ACTIVITIES.items():
            print(f"\n\n{'#'*60}\n# KNN — {activity_name.upper()}\n{'#'*60}")
            df_tr = cargar_grabaciones(df_emb, train_paths,   activity_types)
            df_ho = cargar_grabaciones(df_emb, holdout_paths, activity_types)
            if df_tr['ID_Paciente'].nunique() < 20:
                print(f"  [!] Pacientes insuficientes — omitido.")
                continue
            trainer = KNNWav2VecTrainer(activity_name, cv_folds=5)
            trainer.fit(df_tr, df_ho)
            results_knn[activity_name] = {
                "internal": trainer.metrics_internal_,
                "external": trainer.metrics_external_,
            }
            thresholds_knn[activity_name] = trainer.optimal_threshold

        _print_summary_table(results_knn, "KNN+PCA")
        if len(results_knn) > 1:
            plot_comparison(results_knn,
                            f"KNN+PCA(var={PCA_VARIANCE:.0%}) — Holdout externo",
                            REPORTS_BASE_KNN / "comparison_4models.png")
        save_run_json("Wav2Vec-KNN", args.run, results_knn, thresholds_knn)

        # ── TABLA FINAL COMPARATIVA ───────────────────────────────────────
        all_acts = sorted(set(results_xgb) | set(results_knn))
        print(f"\n\n{'='*80}")
        print("  COMPARACION FINAL — Wav2Vec2 Embeddings — HOLDOUT EXTERNO")
        print(f"{'='*80}")
        print(f"  {'Actividad':12s} | {'XGB BA':8s} | {'XGB AUC':8s} | {'KNN BA':8s} | {'KNN AUC':8s}")
        print("  " + "─" * 56)
        for act in all_acts:
            xba  = results_xgb.get(act, {}).get("external", {}).get("balanced_accuracy", float('nan'))
            xauc = results_xgb.get(act, {}).get("external", {}).get("roc_auc",           float('nan'))
            kba  = results_knn.get(act, {}).get("external", {}).get("balanced_accuracy", float('nan'))
            kauc = results_knn.get(act, {}).get("external", {}).get("roc_auc",           float('nan'))
            print(f"  {act:12s} | {xba:8.4f} | {xauc:8.4f} | {kba:8.4f} | {kauc:8.4f}")
        print(f"{'='*80}")

        print(f"\n  Completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Modelos  -> models/wav2vec/")
        print(f"  Figuras  -> reports/wav2vec/")
        print(f"  Log      -> {log_path}")


if __name__ == "__main__":
    main()
