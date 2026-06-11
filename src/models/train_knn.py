"""
train_knn.py — Entrenamiento KNN sobre 4 tipos de actividad de habla.

Actividades: vocal | frase | espontanea | all
Pipeline por actividad (identico a train_xgboost.py):
  1. Carga train_80.csv / holdout_20.csv  (split paciente-nivel precalculado)
  2. Filtra por activity_type — grabaciones individuales (sin agregar por paciente)
  3. Split interno de PACIENTES 80/20:
       - cv_patients  (80%) → StratifiedGroupKFold 5-fold para GridSearchCV
       - val_patients (20%) → validacion interna (evaluada a nivel de paciente)
  4. Umbral optimo: criterio de Youden sobre probabilidades agregadas por paciente (val interna)
  5. Evaluacion sobre holdout externo: probabilidades de grabaciones → media por paciente → metrica

Salidas:
  models/traditionals/KNN/{run}/{actividad}/    -> model.pkl, scaler.pkl, threshold.pkl, features.pkl
  reports/traditionals/KNN/{run}/{actividad}/   -> 6 figuras por modelo
  reports/traditionals/KNN/{run}/comparison_4models.png
  logs/traditionals/KNN/train_{run}_{timestamp}.log

Uso:
  uv run python -m src.models.train_knn --run sin_age
  uv run python -m src.models.train_knn --run con_age
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

from sklearn.model_selection import (
    train_test_split, StratifiedGroupKFold, GridSearchCV,
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

from src.config import SEED, PALETTE, apply_style
from src.utils.results_logger import save_run_json

warnings.filterwarnings('ignore')

# ─── Rutas ───────────────────────────────────────────────────────────────────
TRAIN_CSV    = Path("data/processed/combined/train_80.csv")
HOLDOUT_CSV  = Path("data/processed/combined/holdout_20.csv")
REPORTS_BASE = Path("reports/traditionals/KNN")
MODELS_BASE  = Path("models/traditionals/KNN")
LOG_DIR      = Path("logs/traditionals/KNN")

# ─── Features y actividades ──────────────────────────────────────────────────
FEATURES: list[str] = ['Age', 'Sex', 'ShimmerDb', 'ATRI', 'Hnr', 'CHNR', 'rPPQ']

# FEATURES_ALL (descomentar para experimentar con todas las features):
# FEATURES: list[str] = [
#     'Age', 'Sex',
#     'JITA', 'rJitter', 'RAP', 'rPPQ', 'rSPPQ',
#     'ShimmerDb', 'Shimmer', 'rAPQ', 'rSAPQ',
#     'Hnr', 'CHNR',
#     'FTRI', 'ATRI', 'FFTR', 'FATR',
# ]

ACTIVITIES: dict[str, list[str]] = {
    'vocal':      ['vocal'],
    'frase':      ['frase'],
    'espontanea': ['espontanea'],
    'all':        ['vocal', 'frase', 'espontanea'],
}


# =============================================================================
# TEE — duplica stdout a pantalla y fichero de log
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
# CARGA DE GRABACIONES INDIVIDUALES
# =============================================================================
def cargar_grabaciones(csv_path: Path, activity_types: list[str]) -> pd.DataFrame:
    """Devuelve grabaciones individuales (sin agregar). Mantiene ID_Paciente para GroupKFold."""
    df = pd.read_csv(csv_path)
    df = df[df['activity_type'].isin(activity_types)].copy()
    df = df.dropna(subset=FEATURES)
    return df.reset_index(drop=True)


def _agregar_proba_por_paciente(df: pd.DataFrame, y_proba: np.ndarray) -> pd.DataFrame:
    """Media de probabilidades de grabaciones → nivel de paciente."""
    tmp = df[['ID_Paciente', 'Target']].copy()
    tmp['proba'] = y_proba
    return tmp.groupby('ID_Paciente').agg(
        proba=('proba', 'mean'),
        Target=('Target', 'first'),
    ).reset_index()


# =============================================================================
# TRAINER
# =============================================================================
class KNNTrainer:

    def __init__(self, activity_name: str, cv_folds: int = 5):
        self.activity_name = activity_name
        self.cv_folds      = cv_folds

        self.models_dir  = MODELS_BASE  / activity_name
        self.reports_dir = REPORTS_BASE / activity_name
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.scaler            = None
        self.best_model        = None
        self.best_params_      = {}
        self.optimal_threshold = 0.5
        self.feature_names     = None
        self.metrics_internal_ = {}
        self.metrics_external_ = {}

        self._X_cv_sc     = None
        self._y_cv        = None
        self._y_val       = None
        self._y_pred_val  = None
        self._y_proba_val = None
        self._y_ext       = None
        self._y_pred_ext  = None
        self._y_proba_ext = None

    # ─── Entrenamiento principal ──────────────────────────────────────────────
    def fit(self, df_train: pd.DataFrame, df_holdout: pd.DataFrame) -> "KNNTrainer":
        """
        df_train / df_holdout: grabaciones individuales (sin agregar por paciente).
        CV usa StratifiedGroupKFold para evitar leakage entre grabaciones del mismo paciente.
        Evaluacion interna y externa a nivel de paciente (media de probabilidades por paciente).
        """
        n_pac_tr = df_train['ID_Paciente'].nunique()
        n_pac_ho = df_holdout['ID_Paciente'].nunique()

        print(f"\n{'='*60}")
        print(f"  ENTRENAMIENTO — KNN — [{self.activity_name.upper()}]")
        print(f"{'='*60}")
        print(f"  Grabaciones train:   {len(df_train):>5}  ({n_pac_tr} pacientes)")
        print(f"  Grabaciones holdout: {len(df_holdout):>5}  ({n_pac_ho} pacientes)")
        print(df_train.groupby(['Dataset', 'Target']).size()
              .rename('n').reset_index().to_string(index=False))

        self.feature_names = FEATURES[:]

        # ── Split de PACIENTES en cv_patients (80%) y val_patients (20%) ─
        unique_pats = df_train[['ID_Paciente', 'Target']].drop_duplicates()
        pat_cv_idx, pat_val_idx = train_test_split(
            unique_pats['ID_Paciente'],
            test_size=0.20,
            stratify=unique_pats['Target'],
            random_state=SEED,
        )
        df_cv  = df_train[df_train['ID_Paciente'].isin(pat_cv_idx)].copy()
        df_val = df_train[df_train['ID_Paciente'].isin(pat_val_idx)].copy()

        X_cv   = df_cv[FEATURES]
        y_cv   = df_cv['Target']
        groups = df_cv['ID_Paciente'].values

        print(f"\n  CV-train: {len(df_cv)} grab. / {pat_cv_idx.nunique()} pac.")
        print(f"  Val interna: {len(df_val)} grab. / {pat_val_idx.nunique()} pac.")

        # ── GridSearchCV con StratifiedGroupKFold ─────────────────────────
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn",   KNeighborsClassifier(n_jobs=-1)),
        ])
        param_grid = {
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'knn__weights':     ['uniform', 'distance'],
            'knn__metric':      ['euclidean', 'manhattan', 'minkowski'],
        }
        cv_strategy = StratifiedGroupKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=SEED
        )
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1,
        )
        print(f"\n  GridSearchCV ({self.cv_folds}-fold StratifiedGroupKFold)...")
        search.fit(X_cv, y_cv, groups=groups)

        self.best_params_ = {
            k.replace('knn__', ''): v for k, v in search.best_params_.items()
        }
        print("\n  Mejores hiperparametros:")
        for k, v in self.best_params_.items():
            print(f"    {k}: {v}")
        print(f"  Balanced Accuracy CV: {search.best_score_:.4f}")

        # ── Refit con scaler separado sobre todos los cv_patients ────────
        self.scaler  = StandardScaler()
        X_cv_sc      = self.scaler.fit_transform(X_cv)
        X_val_sc     = self.scaler.transform(df_val[FEATURES])

        self.best_model = KNeighborsClassifier(**self.best_params_, n_jobs=-1)
        self.best_model.fit(X_cv_sc, y_cv)

        self._X_cv_sc = X_cv_sc
        self._y_cv    = y_cv

        # ── Umbral optimo (Youden) — nivel PACIENTE en val interna ───────
        y_proba_val_rec = self.best_model.predict_proba(X_val_sc)[:, 1]
        pat_val_df = _agregar_proba_por_paciente(df_val, y_proba_val_rec)
        fpr, tpr, ths = roc_curve(pat_val_df['Target'], pat_val_df['proba'])
        finite = np.isfinite(ths)
        if finite.any():
            best_idx = np.argmax((tpr - fpr)[finite])
            self.optimal_threshold = float(ths[finite][best_idx])
        else:
            self.optimal_threshold = 0.5
        print(f"\n  Umbral optimo (Youden, paciente-nivel): {self.optimal_threshold:.4f}")

        # ── Evaluaciones a nivel de PACIENTE ─────────────────────────────
        self._evaluate_pac(df_val,     X_val_sc,
                           tag="VAL INTERNA (paciente)", internal=True)
        self._evaluate_pac(df_holdout, self.scaler.transform(df_holdout[FEATURES]),
                           tag="HOLDOUT EXTERNO (paciente)", internal=False)

        self._save()
        apply_style()
        self._plot_all(X_cv_sc, y_cv)
        return self

    def _evaluate_pac(self, df: pd.DataFrame, X_sc: np.ndarray, tag: str, internal: bool):
        """Predice a nivel de grabacion, agrega por paciente (media de prob.), calcula metricas."""
        y_proba_rec = self.best_model.predict_proba(X_sc)[:, 1]
        pat_df      = _agregar_proba_por_paciente(df, y_proba_rec)

        y_true  = pat_df['Target'].values
        y_proba = pat_df['proba'].values
        y_pred  = (y_proba >= self.optimal_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr_, tpr_, _ = roc_curve(y_true, y_proba)

        metrics = {
            "accuracy":          accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall":            recall_score(y_true, y_pred),
            "specificity":       spec,
            "precision":         precision_score(y_true, y_pred, zero_division=0),
            "f1":                f1_score(y_true, y_pred),
            "roc_auc":           auc(fpr_, tpr_),
        }

        if internal:
            self.metrics_internal_ = metrics
            self._y_val       = y_true
            self._y_pred_val  = y_pred
            self._y_proba_val = y_proba
        else:
            self.metrics_external_ = metrics
            self._y_ext       = y_true
            self._y_pred_ext  = y_pred
            self._y_proba_ext = y_proba

        n_pac = len(pat_df)
        print(f"\n  {'─'*52}")
        print(f"  METRICAS {tag}  ({n_pac} pac., umbral={self.optimal_threshold:.3f})")
        print(f"  {'─'*52}")
        for k, v in metrics.items():
            print(f"  {k:22s}: {v:.4f}")
        print(f"  {'─'*52}")

    def _save(self):
        base = self.models_dir
        joblib.dump(self.best_model,        base / "model.pkl")
        joblib.dump(self.scaler,            base / "scaler.pkl")
        joblib.dump(self.optimal_threshold, base / "threshold.pkl")
        joblib.dump(self.feature_names,     base / "features.pkl")
        print(f"\n  Artefactos guardados en: {base}/")

    # ─── Plots ────────────────────────────────────────────────────────────────
    def _plot_all(self, X_cv_sc: np.ndarray, y_cv: pd.Series):
        self._plot_confusion()
        self._plot_roc()
        self._plot_k_analysis(X_cv_sc, y_cv)
        self._plot_permutation_importance(X_cv_sc, y_cv)
        self._plot_learning_curve(X_cv_sc, y_cv)
        self._plot_summary()

    def _plot_confusion(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Matriz de Confusion — KNN [{self.activity_name}]",
            fontsize=14, fontweight="bold", y=1.02,
        )
        for ax, (y_true, y_pred, subtitle) in zip(axes, [
            (self._y_val, self._y_pred_val, "Val interna"),
            (self._y_ext, self._y_pred_ext, "Holdout externo"),
        ]):
            cm   = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                        xticklabels=["Control", "Parkinson"],
                        yticklabels=["Control", "Parkinson"],
                        cbar=False, annot_kws={"size": 12, "weight": "bold"})
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ax.set_title(f"{subtitle}\nSensib={sens:.3f} | Especif={spec:.3f}")
            ax.set_xlabel("Prediccion")
            ax.set_ylabel("Real")
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
            best_i  = np.argmax(tpr - fpr)
            ax.plot(fpr, tpr, color=color, lw=2.5,
                    label=f"{label} (AUC={auc_val:.4f})")
            ax.scatter(fpr[best_i], tpr[best_i], s=100, color=color, zorder=5)
        ax.plot([0, 1], [0, 1], ":", color=PALETTE["subtext"], lw=2)
        ax.set_xlabel("FPR (1-Especificidad)")
        ax.set_ylabel("TPR (Sensibilidad)")
        ax.set_title(f"Curva ROC — KNN [{self.activity_name}]", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        self._save_fig("2_roc_curve")

    def _plot_k_analysis(self, X_cv_sc: np.ndarray, y_cv: pd.Series):
        """Balanced Accuracy en CV para distintos valores de K."""
        k_range  = list(range(1, 22, 2))
        ba_means, ba_stds = [], []

        for k in k_range:
            model = KNeighborsClassifier(
                n_neighbors=k,
                weights=self.best_params_['weights'],
                metric=self.best_params_['metric'],
                n_jobs=-1,
            )
            scores = cross_val_score(
                model, X_cv_sc, y_cv,
                cv=self.cv_folds,
                scoring='balanced_accuracy',
                n_jobs=-1,
            )
            ba_means.append(scores.mean())
            ba_stds.append(scores.std())

        ba_means = np.array(ba_means)
        ba_stds  = np.array(ba_stds)
        best_k   = k_range[int(np.argmax(ba_means))]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(k_range, ba_means - ba_stds, ba_means + ba_stds,
                        alpha=0.15, color=PALETTE["accent1"])
        ax.plot(k_range, ba_means, "o-", color=PALETTE["accent1"], lw=2.5,
                label="Balanced Accuracy CV (media)")
        ax.axvline(best_k, color=PALETTE["accent3"], lw=2, ls="--",
                   label=f"Mejor K = {best_k}")
        ax.set_xlabel("Numero de vecinos (K)")
        ax.set_ylabel("Balanced Accuracy (CV)")
        ax.set_title(f"Analisis del parametro K — KNN [{self.activity_name}]",
                     fontweight="bold")
        ax.set_xticks(k_range)
        ax.legend()
        plt.tight_layout()
        self._save_fig("3_k_analysis")

    def _plot_permutation_importance(self, X_cv_sc: np.ndarray, y_cv: pd.Series):
        """Importancia por permutacion sobre el conjunto de CV-train."""
        result = permutation_importance(
            self.best_model, X_cv_sc, y_cv,
            n_repeats=15, random_state=SEED,
            scoring='balanced_accuracy', n_jobs=-1,
        )
        sorted_idx = result.importances_mean.argsort()
        names = np.array(self.feature_names)[sorted_idx]
        means = result.importances_mean[sorted_idx]
        stds  = result.importances_std[sorted_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(names, means, xerr=stds, color=PALETTE["accent1"],
                       alpha=0.9, capsize=4)
        for i in range(1, min(4, len(bars) + 1)):
            bars[-i].set_color(PALETTE["accent3"])
        ax.set_xlabel("Disminucion media de Balanced Accuracy al permutar")
        ax.set_title(f"Permutation Importance — KNN [{self.activity_name}]",
                     fontweight="bold")
        plt.tight_layout()
        self._save_fig("4_permutation_importance")

    def _plot_learning_curve(self, X_cv_sc: np.ndarray, y_cv: pd.Series):
        """Curva de aprendizaje (train sizes vs BA) sobre cv_patients."""
        model_lc = KNeighborsClassifier(**self.best_params_, n_jobs=-1)
        train_sizes, tr_scores, val_scores = learning_curve(
            model_lc, X_cv_sc, y_cv,
            cv=min(self.cv_folds, 3),
            scoring='balanced_accuracy',
            train_sizes=np.linspace(0.2, 1.0, 6),
            n_jobs=-1,
        )
        tr_mean, tr_std = tr_scores.mean(axis=1), tr_scores.std(axis=1)
        va_mean, va_std = val_scores.mean(axis=1), val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                        alpha=0.15, color=PALETTE["accent1"])
        ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std,
                        alpha=0.15, color=PALETTE["accent2"])
        ax.plot(train_sizes, tr_mean, "o-", color=PALETTE["accent1"],
                lw=2, label="Train (BA)")
        ax.plot(train_sizes, va_mean, "s-", color=PALETTE["accent2"],
                lw=2, label="Validacion CV (BA)")
        ax.set_xlabel("Grabaciones de entrenamiento")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(f"Curva de aprendizaje — KNN [{self.activity_name}]",
                     fontweight="bold")
        ax.legend()
        plt.tight_layout()
        self._save_fig("5_learning_curve")

    def _plot_summary(self):
        m_i = self.metrics_internal_
        m_e = self.metrics_external_
        fig = plt.figure(figsize=(13, 5.5))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.8])

        cats   = ["Recall\n(Sensib.)", "Especif.", "Precision", "F1", "AUC-ROC"]
        vals_i = [m_i["recall"], m_i["specificity"], m_i["precision"],
                  m_i["f1"], m_i["roc_auc"]]
        vals_e = [m_e["recall"], m_e["specificity"], m_e["precision"],
                  m_e["f1"], m_e["roc_auc"]]
        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
        angles += angles[:1]

        ax_r = fig.add_subplot(gs[0], polar=True)
        for vals, label, color in [
            (vals_i, "Val interna",     PALETTE["accent1"]),
            (vals_e, "Holdout externo", PALETTE["accent3"]),
        ]:
            vr = vals + [vals[0]]
            ax_r.plot(angles, vr, color=color, lw=2, label=label)
            ax_r.fill(angles, vr, color=color, alpha=0.1)
        ax_r.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=9)
        ax_r.set_ylim(0, 1)
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        ax_r.set_title(f"[{self.activity_name}]", fontweight="bold", pad=20)

        ax_t = fig.add_subplot(gs[1])
        ax_t.axis("off")
        rows = [
            ["Metrica",            "Val interna",                     "Holdout externo"],
            ["Balanced Accuracy",  f"{m_i['balanced_accuracy']:.4f}", f"{m_e['balanced_accuracy']:.4f}"],
            ["Recall (PD)",        f"{m_i['recall']:.4f}",            f"{m_e['recall']:.4f}"],
            ["Especificidad (HC)", f"{m_i['specificity']:.4f}",       f"{m_e['specificity']:.4f}"],
            ["F1-Score",           f"{m_i['f1']:.4f}",                f"{m_e['f1']:.4f}"],
            ["AUC-ROC",            f"{m_i['roc_auc']:.4f}",           f"{m_e['roc_auc']:.4f}"],
            ["Accuracy",           f"{m_i['accuracy']:.4f}",          f"{m_e['accuracy']:.4f}"],
            ["Umbral Youden",      f"{self.optimal_threshold:.4f}",   "—"],
            ["K optimo",           f"{self.best_model.n_neighbors}",  "—"],
        ]
        tbl = ax_t.table(cellText=rows, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.8)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(PALETTE["accent1"])
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor(PALETTE["panel"])
        plt.subplots_adjust(bottom=0.1)
        self._save_fig("6_metrics_summary")

    def _save_fig(self, name: str, dpi: int = 300):
        path = self.reports_dir / f"{name}.png"
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: {path}")


# =============================================================================
# COMPARACION FINAL — 4 MODELOS (holdout externo)
# =============================================================================
def plot_comparison(results: dict):
    apply_style()

    METRIC_LABELS = {
        "balanced_accuracy": "Balanced Acc.",
        "roc_auc":           "AUC-ROC",
        "f1":                "F1-Score",
        "recall":            "Recall (Sensib.)",
        "specificity":       "Especificidad",
    }
    models = list(results.keys())
    n_met  = len(METRIC_LABELS)
    x      = np.arange(len(models))
    width  = 0.14
    colors = [
        PALETTE["accent1"], PALETTE["accent3"],
        PALETTE["accent2"], PALETTE["accent4"], PALETTE["warn"],
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (key, label) in enumerate(METRIC_LABELS.items()):
        vals   = [results[m]["external"][key] for m in models]
        offset = (i - n_met / 2) * width + width / 2
        bars   = ax.bar(x + offset, vals, width, label=label,
                        color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold",
            )

    ax.set_xlabel("Tipo de actividad de habla")
    ax.set_ylabel("Puntuacion")
    ax.set_title("Comparacion KNN — Holdout externo (4 actividades)",
                 fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in models], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.5, color=PALETTE["subtext"], lw=1, ls="--", alpha=0.5)
    plt.tight_layout()

    out = REPORTS_BASE / "comparison_4models.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparacion guardada en: {out}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='default',
                        help='Nombre del run, define la subcarpeta de salida (ej: sin_age, con_age)')
    parser.add_argument('--no-age', action='store_true',
                        help='Excluye Age y Sex de las features')
    args = parser.parse_args()

    global FEATURES, REPORTS_BASE, MODELS_BASE
    if args.no_age:
        FEATURES[:] = [f for f in FEATURES if f not in ('Age', 'Sex')]
        print(f"[--no-age] Features activas: {FEATURES}")

    REPORTS_BASE = REPORTS_BASE / args.run
    MODELS_BASE  = MODELS_BASE  / args.run

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_BASE.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = LOG_DIR / f"train_{args.run}_{timestamp}.log"

    with Tee(log_path):
        print(f"Log: {log_path}\n")

        for path in [TRAIN_CSV, HOLDOUT_CSV]:
            if not path.exists():
                print(f"[!] No encontrado: {path}")
                print("    Ejecuta: uv run python -m src.features.split_dataset")
                return

        print(f"{'='*60}")
        print("  Cortex-AI — Entrenamiento KNN (4 actividades)")
        print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        df_train_raw   = pd.read_csv(TRAIN_CSV)
        df_holdout_raw = pd.read_csv(HOLDOUT_CSV)
        print(f"\nTrain raw:   {len(df_train_raw)} grabaciones, "
              f"{df_train_raw['ID_Paciente'].nunique()} pacientes")
        print(f"Holdout raw: {len(df_holdout_raw)} grabaciones, "
              f"{df_holdout_raw['ID_Paciente'].nunique()} pacientes")
        print("\nDistribucion (train):")
        print(df_train_raw
              .groupby(['Dataset', 'activity_type', 'Target']).size()
              .rename('n').reset_index().to_string(index=False))

        results    = {}
        thresholds = {}
        for activity_name, activity_types in ACTIVITIES.items():
            print(f"\n\n{'#'*60}")
            print(f"# ACTIVIDAD: {activity_name.upper()}")
            print(f"{'#'*60}")

            df_tr_act = cargar_grabaciones(TRAIN_CSV,   activity_types)
            df_ho_act = cargar_grabaciones(HOLDOUT_CSV, activity_types)

            n_pac_tr = df_tr_act['ID_Paciente'].nunique()
            if n_pac_tr < 20:
                print(f"  [!] Solo {n_pac_tr} pacientes en train — actividad omitida.")
                continue

            print(f"  Train:   {len(df_tr_act)} grab. / {n_pac_tr} pac.")
            print(f"  Holdout: {len(df_ho_act)} grab. / {df_ho_act['ID_Paciente'].nunique()} pac.")

            trainer = KNNTrainer(activity_name, cv_folds=5)
            trainer.fit(df_tr_act, df_ho_act)

            results[activity_name] = {
                "internal": trainer.metrics_internal_,
                "external": trainer.metrics_external_,
            }
            thresholds[activity_name] = trainer.optimal_threshold

        # ── Tabla resumen ─────────────────────────────────────────────────
        print(f"\n\n{'='*68}")
        print("  RESUMEN FINAL — HOLDOUT EXTERNO")
        print(f"{'='*68}")
        header = (f"  {'Actividad':12s} | {'Bal.Acc':8s} | {'AUC':8s} | "
                  f"{'F1':8s} | {'Recall':8s} | {'Specif.':8s}")
        print(header)
        print("  " + "─" * 64)
        for act, r in results.items():
            m = r["external"]
            print(f"  {act:12s} | {m['balanced_accuracy']:8.4f} | {m['roc_auc']:8.4f} | "
                  f"{m['f1']:8.4f} | {m['recall']:8.4f} | {m['specificity']:8.4f}")
        print(f"{'='*68}")

        if len(results) > 1:
            plot_comparison(results)

        save_run_json("KNN", args.run, results, thresholds)

        print(f"\n  Completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Modelos  -> models/traditionals/KNN/")
        print(f"  Figuras  -> reports/traditionals/KNN/")
        print(f"  Log      -> {log_path}")


if __name__ == "__main__":
    main()
