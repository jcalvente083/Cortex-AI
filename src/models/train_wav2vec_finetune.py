"""
train_wav2vec_finetune.py — Fine-tuning Wav2Vec2 end-to-end para clasificacion PD vs HC.

Modelo    : jonatasgrosman/wav2vec2-large-xlsr-53-spanish (24 capas transformer, 1024-dim)
Input     : audio crudo resamplado a 16 kHz, truncado a MAX_AUDIO_SECS segundos
Protocolo : identico a train_resnet.py
  - Split de PACIENTES 80/20 dentro del train (cv_patients / val_patients)
  - Evaluacion a nivel de paciente (media de probabilidades de todas sus grabaciones)
  - Umbral optimo: criterio de Youden sobre val interna a nivel de paciente

Estrategia de freeze:
  - Feature encoder CNN        : siempre congelado (freeze_feature_encoder())
  - Feature projection         : congelado
  - Transformer layers [0..N)  : congelados  (N = --freeze-layers, default 21 de 24)
  - Transformer layers [N..23] + clasificador: entrenables (~20M params)

Salidas:
  models/wav2vec_finetune/Wav2Vec2/{run}/{actividad}/   -> model.pth, threshold.pkl
  reports/wav2vec_finetune/Wav2Vec2/{run}/{actividad}/  -> 6 figuras por modelo
  reports/wav2vec_finetune/Wav2Vec2/{run}/comparison_4models.png
  logs/wav2vec_finetune/train_{run}_{timestamp}.log

Uso:
  uv run python -m src.models.train_wav2vec_finetune --run baseline
  uv run python -m src.models.train_wav2vec_finetune --run freeze18 --freeze-layers 18
  uv run python -m src.models.train_wav2vec_finetune --run run2 --epochs 40 --batch-size 8
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
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    recall_score, precision_score, f1_score,
    accuracy_score, balanced_accuracy_score,
)

from src.config import SEED, PALETTE, apply_style, SR

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
DATA_ROOT    = Path("/data1")
TRAIN_CSV    = DATA_ROOT / "data/processed/combined/train_80.csv"
HOLDOUT_CSV  = DATA_ROOT / "data/processed/combined/holdout_20.csv"
REPORTS_BASE = Path("reports/wav2vec_finetune/Wav2Vec2")
MODELS_BASE  = Path("models/wav2vec_finetune/Wav2Vec2")
LOG_DIR      = Path("logs/wav2vec_finetune")

# =============================================================================
# HIPERPARAMETROS
# =============================================================================
MODEL_NAME      = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
MAX_AUDIO_SECS  = 10
MAX_SAMPLES     = SR * MAX_AUDIO_SECS      # 160 000 samples
N_LAYERS_FROZEN = 21                        # Congelar [0..21), entrenar [21..23]
EPOCHS          = 30
BATCH_SIZE      = 4
LR              = 1e-5
WEIGHT_DECAY    = 1e-2
PATIENCE        = 8
LR_PATIENCE     = 3
GRAD_CLIP       = 1.0
DROPOUT         = 0.1
GRAD_ACCUM      = 4                         # batch efectivo = 4 × 4 = 16

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
# MODELO
# =============================================================================
class Wav2Vec2Classifier(nn.Module):
    """Wav2Vec2 backbone + mean pooling temporal + clasificador lineal."""

    def __init__(self, model_name: str, n_classes: int = 2, dropout: float = DROPOUT):
        super().__init__()
        self.backbone   = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size     = self.backbone.config.hidden_size  # 1024 en large
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_values)
        pooled  = outputs.last_hidden_state.mean(dim=1)  # (batch, 1024)
        return self.classifier(pooled)

    def freeze_layers(self, n_layers_frozen: int) -> None:
        """
        Congela: feature encoder CNN + feature projection + capas [0..n_layers_frozen).
        Entrena: capas [n_layers_frozen..23] + clasificador.
        """
        self.backbone.freeze_feature_encoder()

        for param in self.backbone.feature_projection.parameters():
            param.requires_grad = False

        total_layers = len(self.backbone.encoder.layers)
        n_freeze     = min(n_layers_frozen, total_layers)
        for i, layer in enumerate(self.backbone.encoder.layers):
            if i < n_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Freeze : feature_encoder + feature_proj + encoder.layers[0:{n_freeze}]")
        print(f"  Params entrenables: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


# =============================================================================
# DATASET
# =============================================================================
class AudioDataset(Dataset):
    """
    Carga audio crudo en tiempo real.
    Trunca/pad a MAX_SAMPLES. Normaliza con el feature extractor de Wav2Vec2.
    """

    def __init__(self, df: pd.DataFrame, feature_extractor, training: bool = False):
        self.df                = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.training          = training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            waveform, _ = librosa.load(row['audio_path'], sr=SR, mono=True)
        except Exception:
            waveform = np.zeros(MAX_SAMPLES, dtype=np.float32)

        if len(waveform) >= MAX_SAMPLES:
            if self.training:
                start = np.random.randint(0, len(waveform) - MAX_SAMPLES + 1)
                waveform = waveform[start:start + MAX_SAMPLES]
            else:
                waveform = waveform[:MAX_SAMPLES]
        else:
            waveform = np.pad(waveform, (0, MAX_SAMPLES - len(waveform)))

        inputs = self.feature_extractor(
            waveform.astype(np.float32),
            sampling_rate=SR,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.squeeze(0)   # (MAX_SAMPLES,)
        return input_values, int(row['Target']), str(row['ID_Paciente'])


# =============================================================================
# TRAINER
# =============================================================================
class Wav2Vec2FinetuneTrainer:

    def __init__(
        self,
        activity_name:    str,
        feature_extractor,
        epochs:           int = EPOCHS,
        batch_size:       int = BATCH_SIZE,
        n_layers_frozen:  int = N_LAYERS_FROZEN,
        grad_accum:       int = GRAD_ACCUM,
    ):
        self.activity_name    = activity_name
        self.feature_extractor = feature_extractor
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.n_layers_frozen  = n_layers_frozen
        self.grad_accum       = grad_accum

        self.models_dir  = MODELS_BASE  / activity_name
        self.reports_dir = REPORTS_BASE / activity_name
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()
        print(f"  Dispositivo: {self.device}  |  AMP: {self.use_amp}")

        self.model             = None
        self.optimal_threshold = 0.5
        self.history: dict     = {'train_loss': [], 'val_loss': [],
                                   'train_acc':  [], 'val_acc':  []}
        self.metrics_internal_ = {}
        self.metrics_external_ = {}

        self._y_val       = None
        self._y_pred_val  = None
        self._y_proba_val = None
        self._y_ext       = None
        self._y_pred_ext  = None
        self._y_proba_ext = None
        self._sample_waves: dict[int, list] = {0: [], 1: []}

    # ─── Modelo ──────────────────────────────────────────────────────────────
    def _build_model(self) -> Wav2Vec2Classifier:
        model = Wav2Vec2Classifier(MODEL_NAME).to(self.device)
        model.freeze_layers(self.n_layers_frozen)
        return model

    # ─── DataLoader ──────────────────────────────────────────────────────────
    def _make_loader(self, df: pd.DataFrame, shuffle: bool, training: bool = False) -> DataLoader:
        dataset = AudioDataset(df, self.feature_extractor, training=training)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    # ─── Epoch ───────────────────────────────────────────────────────────────
    def _run_epoch(self, loader: DataLoader, optimizer, criterion, scaler, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            if train:
                optimizer.zero_grad()
            for batch_idx, (inputs, targets, _) in enumerate(
                tqdm(loader, desc="    batch", leave=False, ncols=80)
            ):
                inputs  = inputs.to(self.device)
                targets = (targets if isinstance(targets, torch.Tensor)
                           else torch.tensor(targets)).long().to(self.device)

                if self.use_amp:
                    with torch.autocast(device_type='cuda'):
                        outputs = self.model(inputs)
                        loss    = criterion(outputs, targets) / self.grad_accum
                else:
                    outputs = self.model(inputs)
                    loss    = criterion(outputs, targets) / self.grad_accum

                if train:
                    if self.use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                total_loss += loss.item() * self.grad_accum * inputs.size(0)
                _, predicted = torch.max(outputs.detach(), 1)
                correct += (predicted == targets).sum().item()
                total   += targets.size(0)

                if train:
                    last_batch = (batch_idx == len(loader) - 1)
                    if (batch_idx + 1) % self.grad_accum == 0 or last_batch:
                        if self.use_amp:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                            optimizer.step()
                        optimizer.zero_grad()

        return total_loss / max(total, 1), correct / max(total, 1)

    # ─── Inferencia paciente-nivel ────────────────────────────────────────────
    def _infer_patient_level(self, loader: DataLoader) -> pd.DataFrame:
        self.model.eval()
        all_proba, all_targets, all_pids = [], [], []

        with torch.no_grad():
            for inputs, targets, pids in loader:
                inputs = inputs.to(self.device)
                if self.use_amp:
                    with torch.autocast(device_type='cuda'):
                        logits = self.model(inputs)
                else:
                    logits = self.model(inputs)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_proba.extend(probs.cpu().numpy().tolist())
                all_targets.extend(targets.tolist() if isinstance(targets, torch.Tensor)
                                   else list(targets))
                all_pids.extend(list(pids))

        df = pd.DataFrame({'ID_Paciente': all_pids, 'proba': all_proba, 'Target': all_targets})
        return df.groupby('ID_Paciente').agg(
            proba=('proba', 'mean'),
            Target=('Target', 'first'),
        ).reset_index()

    # ─── Ejemplos de waveform ─────────────────────────────────────────────────
    def _collect_samples(self, df: pd.DataFrame, n_per_class: int = 3):
        self._sample_waves = {0: [], 1: []}
        for _, row in df.iterrows():
            cls = int(row['Target'])
            if len(self._sample_waves[cls]) < n_per_class:
                try:
                    w, _ = librosa.load(row['audio_path'], sr=SR, mono=True)
                    w = (w[:MAX_SAMPLES] if len(w) >= MAX_SAMPLES
                         else np.pad(w, (0, MAX_SAMPLES - len(w))))
                    self._sample_waves[cls].append(w)
                except Exception:
                    pass
            if all(len(v) >= n_per_class for v in self._sample_waves.values()):
                break

    # ─── Entrenamiento principal ──────────────────────────────────────────────
    def fit(self, df_train: pd.DataFrame, df_holdout: pd.DataFrame) -> "Wav2Vec2FinetuneTrainer":
        n_pac_tr = df_train['ID_Paciente'].nunique()
        n_pac_ho = df_holdout['ID_Paciente'].nunique()

        print(f"\n{'='*60}")
        print(f"  ENTRENAMIENTO — Wav2Vec2 Fine-tuning — [{self.activity_name.upper()}]")
        print(f"{'='*60}")
        print(f"  Audios train  : {len(df_train):>5}  ({n_pac_tr} pacientes)")
        print(f"  Audios holdout: {len(df_holdout):>5}  ({n_pac_ho} pacientes)")
        print(f"  Audio max dur : {MAX_AUDIO_SECS}s ({MAX_SAMPLES:,} samples @ {SR} Hz)")

        # ── Split de PACIENTES: cv (80%) / val (20%) ──────────────────────
        unique_pats = df_train[['ID_Paciente', 'Target']].drop_duplicates()
        pat_cv_idx, pat_val_idx = train_test_split(
            unique_pats['ID_Paciente'],
            test_size=0.20,
            stratify=unique_pats['Target'],
            random_state=SEED,
        )
        df_cv  = df_train[df_train['ID_Paciente'].isin(pat_cv_idx)].copy()
        df_val = df_train[df_train['ID_Paciente'].isin(pat_val_idx)].copy()

        print(f"\n  CV-train  : {len(df_cv)} aud. / {pat_cv_idx.nunique()} pac.")
        print(f"  Val intern: {len(df_val)} aud. / {pat_val_idx.nunique()} pac.")

        self._collect_samples(df_val)

        # ── Loaders ──────────────────────────────────────────────────────
        train_loader   = self._make_loader(df_cv,      shuffle=True,  training=True)
        val_loader     = self._make_loader(df_val,     shuffle=False, training=False)
        holdout_loader = self._make_loader(df_holdout, shuffle=False, training=False)

        # ── Modelo, optimizador, scheduler ──────────────────────────────
        print("\n  Cargando Wav2Vec2 y configurando freeze...")
        self.model = self._build_model()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, min_lr=1e-7,
        )
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        best_val_loss = float('inf')
        epochs_no_imp = 0
        temp_path     = self.models_dir / "_best_temp.pth"

        eff_batch = self.batch_size * self.grad_accum
        print(f"\n  Entrenando {self.epochs} epocas "
              f"(patience={PATIENCE}, grad_accum={self.grad_accum}, "
              f"batch_efectivo={eff_batch})...")
        print(f"  {'Epoca':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
              f"{'Train Acc':>9} | {'Val Acc':>9} | {'LR':>8}")
        print(f"  {'─'*62}")

        for epoch in range(1, self.epochs + 1):
            tr_loss, tr_acc = self._run_epoch(
                train_loader, optimizer, criterion, scaler, train=True)
            vl_loss, vl_acc = self._run_epoch(
                val_loader,   optimizer, criterion, scaler, train=False)

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(vl_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_acc'].append(vl_acc)

            lr_now = optimizer.param_groups[0]['lr']
            print(f"  {epoch:>5} | {tr_loss:>10.4f} | {vl_loss:>10.4f} | "
                  f"{tr_acc:>9.4f} | {vl_acc:>9.4f} | {lr_now:>8.2e}")

            scheduler.step(vl_loss)

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                epochs_no_imp = 0
                torch.save(self.model.state_dict(), temp_path)
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= PATIENCE:
                    print(f"\n  Early stopping en epoca {epoch}. Restaurando mejor modelo.")
                    break

        if temp_path.exists():
            self.model.load_state_dict(
                torch.load(temp_path, map_location=self.device, weights_only=True)
            )
            temp_path.unlink()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Umbral Youden (nivel paciente, val interna) ───────────────────
        print("\n  Calculando umbral Youden (val interna, nivel paciente)...")
        pat_val_df = self._infer_patient_level(val_loader)
        fpr, tpr, ths = roc_curve(pat_val_df['Target'], pat_val_df['proba'])
        finite = np.isfinite(ths)
        if finite.any():
            best_idx = np.argmax((tpr - fpr)[finite])
            self.optimal_threshold = float(ths[finite][best_idx])
        else:
            self.optimal_threshold = 0.5
        print(f"  Umbral optimo: {self.optimal_threshold:.4f}")

        # ── Evaluaciones paciente-nivel ──────────────────────────────────
        pat_hol_df = self._infer_patient_level(holdout_loader)
        self._evaluate_pac(pat_val_df, "VAL INTERNA (paciente)",     internal=True)
        self._evaluate_pac(pat_hol_df, "HOLDOUT EXTERNO (paciente)", internal=False)

        self._save()
        apply_style()
        self._plot_all()
        return self

    # ─── Evaluacion ──────────────────────────────────────────────────────────
    def _evaluate_pac(self, pat_df: pd.DataFrame, tag: str, internal: bool):
        y_true  = pat_df['Target'].values
        y_proba = pat_df['proba'].values
        y_pred  = (y_proba >= self.optimal_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        try:
            fpr_, tpr_, _ = roc_curve(y_true, y_proba)
            auc_val = auc(fpr_, tpr_)
        except Exception:
            auc_val = 0.0

        metrics = {
            "accuracy":          accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall":            recall_score(y_true, y_pred, zero_division=0),
            "specificity":       spec,
            "precision":         precision_score(y_true, y_pred, zero_division=0),
            "f1":                f1_score(y_true, y_pred, zero_division=0),
            "roc_auc":           auc_val,
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

        print(f"\n  {'─'*52}")
        print(f"  METRICAS {tag}  ({len(pat_df)} pac., umbral={self.optimal_threshold:.3f})")
        print(f"  {'─'*52}")
        for k, v in metrics.items():
            print(f"  {k:22s}: {v:.4f}")
        print(f"  {'─'*52}")

    # ─── Guardado ─────────────────────────────────────────────────────────────
    def _save(self):
        torch.save(self.model.state_dict(), self.models_dir / "model.pth")
        joblib.dump(self.optimal_threshold, self.models_dir / "threshold.pkl")
        print(f"\n  Artefactos guardados en: {self.models_dir}/")

    # ─── Plots ────────────────────────────────────────────────────────────────
    def _plot_all(self):
        self._plot_confusion()
        self._plot_roc()
        self._plot_training_curves()
        self._plot_probability_distribution()
        self._plot_waveform_examples()
        self._plot_metrics_summary()

    def _plot_confusion(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Matriz de Confusion — Wav2Vec2 Fine-tuning [{self.activity_name}]",
            fontsize=14, fontweight="bold", y=1.02,
        )
        for ax, (y_true, y_pred, subtitle) in zip(axes, [
            (self._y_val, self._y_pred_val, "Val interna"),
            (self._y_ext, self._y_pred_ext, "Holdout externo"),
        ]):
            cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
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
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_val     = auc(fpr, tpr)
                best_i      = np.argmax(tpr - fpr)
                ax.plot(fpr, tpr, color=color, lw=2.5,
                        label=f"{label} (AUC={auc_val:.4f})")
                ax.scatter(fpr[best_i], tpr[best_i], s=100, color=color, zorder=5)
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], ":", color=PALETTE["subtext"], lw=2)
        ax.set_xlabel("FPR (1-Especificidad)")
        ax.set_ylabel("TPR (Sensibilidad)")
        ax.set_title(f"Curva ROC — Wav2Vec2 Fine-tuning [{self.activity_name}]",
                     fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        self._save_fig("2_roc_curve")

    def _plot_training_curves(self):
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs_range, self.history['train_loss'],
                     color=PALETTE["accent1"], lw=2, label="Train Loss")
        axes[0].plot(epochs_range, self.history['val_loss'],
                     color=PALETTE["accent3"], lw=2, label="Val Loss")
        axes[0].set_title(f"Historial de Perdida — [{self.activity_name}]", fontweight="bold")
        axes[0].set_xlabel("Epocas")
        axes[0].set_ylabel("Cross Entropy Loss")
        axes[0].legend()

        axes[1].plot(epochs_range, self.history['train_acc'],
                     color=PALETTE["accent1"], lw=2, label="Train Acc")
        axes[1].plot(epochs_range, self.history['val_acc'],
                     color=PALETTE["accent2"], lw=2, label="Val Acc")
        axes[1].set_title(f"Historial de Accuracy — [{self.activity_name}]", fontweight="bold")
        axes[1].set_xlabel("Epocas")
        axes[1].set_ylabel("Accuracy (audio-level)")
        axes[1].legend()

        plt.tight_layout()
        self._save_fig("3_training_curves")

    def _plot_probability_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Distribucion de Probabilidades — Wav2Vec2 [{self.activity_name}]",
            fontweight="bold", fontsize=13,
        )
        for ax, (y_true, y_proba, subtitle) in zip(axes, [
            (self._y_val, self._y_proba_val, "Val interna"),
            (self._y_ext, self._y_proba_ext, "Holdout externo"),
        ]):
            df_plot = pd.DataFrame({
                'Probabilidad PD': y_proba,
                'Clase': ['Parkinson' if t else 'Control' for t in y_true],
            })
            sns.violinplot(
                data=df_plot, x='Clase', y='Probabilidad PD', ax=ax,
                palette={'Control': PALETTE["accent2"], 'Parkinson': PALETTE["accent3"]},
                inner='box', cut=0,
            )
            ax.axhline(self.optimal_threshold, color=PALETTE["accent1"],
                       lw=2, ls='--', label=f'Umbral={self.optimal_threshold:.3f}')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(subtitle)
            ax.legend(fontsize=9)
        plt.tight_layout()
        self._save_fig("4_probability_distribution")

    def _plot_waveform_examples(self):
        if not self._sample_waves or not any(self._sample_waves.values()):
            return

        n_cols = max(len(v) for v in self._sample_waves.values())
        if n_cols == 0:
            return

        labels = {0: "Control (HC)", 1: "Parkinson (PD)"}
        colors = {0: PALETTE["accent2"], 1: PALETTE["accent3"]}
        t_axis = np.linspace(0, MAX_AUDIO_SECS, MAX_SAMPLES)

        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 7))
        fig.suptitle(
            f"Ejemplos de Waveform — Wav2Vec2 [{self.activity_name}]",
            fontweight="bold", fontsize=13,
        )
        if n_cols == 1:
            axes = np.array(axes).reshape(2, 1)

        for row_idx, cls in enumerate([0, 1]):
            waves = self._sample_waves.get(cls, [])
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                if col_idx < len(waves):
                    ax.plot(t_axis, waves[col_idx], color=colors[cls], lw=0.6, alpha=0.85)
                    ax.set_title(f"{labels[cls]} #{col_idx + 1}",
                                 color=colors[cls], fontweight="bold")
                    ax.set_xlabel("Tiempo (s)")
                    ax.set_ylabel("Amplitud")
                    ax.set_xlim(0, MAX_AUDIO_SECS)
                else:
                    ax.axis('off')

        plt.tight_layout()
        self._save_fig("5_waveform_examples")

    def _plot_metrics_summary(self):
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
        eff_batch = self.batch_size * self.grad_accum
        rows = [
            ["Metrica",            "Val interna",                     "Holdout externo"],
            ["Balanced Accuracy",  f"{m_i['balanced_accuracy']:.4f}", f"{m_e['balanced_accuracy']:.4f}"],
            ["Recall (PD)",        f"{m_i['recall']:.4f}",            f"{m_e['recall']:.4f}"],
            ["Especificidad (HC)", f"{m_i['specificity']:.4f}",       f"{m_e['specificity']:.4f}"],
            ["F1-Score",           f"{m_i['f1']:.4f}",                f"{m_e['f1']:.4f}"],
            ["AUC-ROC",            f"{m_i['roc_auc']:.4f}",           f"{m_e['roc_auc']:.4f}"],
            ["Accuracy",           f"{m_i['accuracy']:.4f}",          f"{m_e['accuracy']:.4f}"],
            ["Umbral Youden",      f"{self.optimal_threshold:.4f}",   "—"],
            ["Epocas entrenadas",  f"{len(self.history['train_loss'])}", "—"],
            ["Freeze layers",      f"[0:{self.n_layers_frozen}] de 24", "—"],
            ["Batch efectivo",     f"{self.batch_size}x{self.grad_accum}={eff_batch}", "—"],
        ]
        tbl = ax_t.table(cellText=rows, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.65)
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
# COMPARACION FINAL — 4 ACTIVIDADES
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
    colors = [PALETTE["accent1"], PALETTE["accent3"],
               PALETTE["accent2"], PALETTE["accent4"], PALETTE["warn"]]

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
    ax.set_title(
        "Comparacion Wav2Vec2 Fine-tuning — Holdout externo (4 actividades)",
        fontweight="bold", fontsize=13,
    )
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
# AGE-MATCHING
# =============================================================================
def _age_match_neurovoz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retiene solo pacientes de NeuroVoz emparejados 1:1 por edad (±5 años).
    PC-GITA pasa sin cambios.

    Algoritmo greedy: itera sobre los PD ordenados por edad y asigna el HC
    más cercano disponible dentro de la ventana. Emparejamiento a nivel de
    paciente; todas las grabaciones del par quedan incluidas.
    """
    pcgita = df[df['Dataset'] != 'NeuroVoz'].copy()
    nv     = df[df['Dataset'] == 'NeuroVoz'].copy()

    if 'Age' not in nv.columns or nv['Age'].isna().all():
        print("  [!] Age no disponible en NeuroVoz — age-match omitido.")
        return df

    pats    = nv[['ID_Paciente', 'Target', 'Age']].drop_duplicates('ID_Paciente')
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

    kept       = set(matched_pd) | set(matched_hc)
    nv_matched = nv[nv['ID_Paciente'].isin(kept)]

    n_removed = pats.shape[0] - len(kept)
    print(f"  Age-match NeuroVoz: {len(matched_pd)} pares emparejados "
          f"({n_removed} pacientes descartados de NeuroVoz)")
    hc_ages = pats[pats['ID_Paciente'].isin(matched_hc)]['Age']
    pd_ages = pats[pats['ID_Paciente'].isin(matched_pd)]['Age']
    print(f"    HC edad: {hc_ages.mean():.1f} ± {hc_ages.std():.1f}  |  "
          f"PD edad: {pd_ages.mean():.1f} ± {pd_ages.std():.1f}")

    return pd.concat([nv_matched, pcgita], ignore_index=True)


# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning Wav2Vec2 end-to-end para clasificacion Parkinson vs Control"
    )
    parser.add_argument('--run',           default='baseline',
                        help='Nombre del run (define subcarpeta de salida)')
    parser.add_argument('--epochs',        type=int, default=EPOCHS,
                        help=f'Epocas de entrenamiento (default: {EPOCHS})')
    parser.add_argument('--batch-size',    type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--freeze-layers', type=int, default=N_LAYERS_FROZEN,
                        help=f'Numero de capas transformer a congelar (default: {N_LAYERS_FROZEN})')
    parser.add_argument('--grad-accum',    type=int, default=GRAD_ACCUM,
                        help=f'Pasos de gradient accumulation (default: {GRAD_ACCUM})')
    parser.add_argument('--age-match',     action='store_true',
                        help='Age-matching 1:1 (±5 años) en NeuroVoz para eliminar confound demográfico')
    args = parser.parse_args()

    global REPORTS_BASE, MODELS_BASE
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

        eff_batch = args.batch_size * args.grad_accum
        print(f"{'='*60}")
        print("  Cortex-AI — Wav2Vec2 Fine-tuning (4 actividades)")
        print(f"  Inicio     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Modelo     : {MODEL_NAME}")
        print(f"  Epocas     : {args.epochs} | Batch: {args.batch_size}")
        print(f"  Freeze     : capas [0:{args.freeze_layers}] de 24")
        print(f"  Grad Accum : {args.grad_accum}  (batch efectivo: {eff_batch})")
        print(f"  Audio max  : {MAX_AUDIO_SECS}s | SR: {SR} Hz")
        print(f"  Device     : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
        print(f"{'='*60}")

        df_train   = pd.read_csv(TRAIN_CSV)
        df_holdout = pd.read_csv(HOLDOUT_CSV)

        # Convertir audio_path relativas a absolutas bajo DATA_ROOT
        for df in [df_train, df_holdout]:
            df['audio_path'] = df['audio_path'].apply(
                lambda p: str(DATA_ROOT / p) if not Path(p).is_absolute() else p
            )

        if args.age_match:
            print("\n  [Age-match activo]")
            df_all     = pd.concat([df_train, df_holdout], ignore_index=True)
            df_all     = _age_match_neurovoz(df_all)
            kept       = set(df_all['ID_Paciente'])
            df_train   = df_train[df_train['ID_Paciente'].isin(kept)].copy()
            df_holdout = df_holdout[df_holdout['ID_Paciente'].isin(kept)].copy()
            print(f"  Tras age-match — Train: {len(df_train)} grab. / "
                  f"{df_train['ID_Paciente'].nunique()} pac. | "
                  f"Holdout: {len(df_holdout)} grab. / "
                  f"{df_holdout['ID_Paciente'].nunique()} pac.")

        print(f"\nTrain  : {len(df_train)} grab. / {df_train['ID_Paciente'].nunique()} pac.")
        print(f"Holdout: {len(df_holdout)} grab. / {df_holdout['ID_Paciente'].nunique()} pac.")
        print("\nDistribucion (train):")
        print(df_train.groupby(['Dataset', 'activity_type', 'Target']).size()
              .rename('n').reset_index().to_string(index=False))

        print(f"\nCargando feature extractor ({MODEL_NAME})...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        print("  Feature extractor listo.\n")

        results = {}
        for activity_name, activity_types in ACTIVITIES.items():
            print(f"\n\n{'#'*60}")
            print(f"# ACTIVIDAD: {activity_name.upper()}")
            print(f"{'#'*60}")

            df_tr = df_train[df_train['activity_type'].isin(activity_types)].copy()
            df_ho = df_holdout[df_holdout['activity_type'].isin(activity_types)].copy()

            n_pac_tr = df_tr['ID_Paciente'].nunique()
            if n_pac_tr < 20:
                print(f"  [!] Solo {n_pac_tr} pacientes en train — actividad omitida.")
                continue

            print(f"  Train  : {len(df_tr)} aud. / {n_pac_tr} pac.")
            print(f"  Holdout: {len(df_ho)} aud. / {df_ho['ID_Paciente'].nunique()} pac.")

            trainer = Wav2Vec2FinetuneTrainer(
                activity_name    = activity_name,
                feature_extractor = feature_extractor,
                epochs           = args.epochs,
                batch_size       = args.batch_size,
                n_layers_frozen  = args.freeze_layers,
                grad_accum       = args.grad_accum,
            )
            trainer.fit(df_tr, df_ho)

            if trainer.metrics_internal_ and trainer.metrics_external_:
                results[activity_name] = {
                    "internal": trainer.metrics_internal_,
                    "external": trainer.metrics_external_,
                }

            # Liberar VRAM antes de la siguiente actividad
            if trainer.model is not None:
                del trainer.model
                trainer.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Tabla resumen ─────────────────────────────────────────────────
        if results:
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

        print(f"\n  Completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Modelos -> {MODELS_BASE}/")
        print(f"  Figuras -> {REPORTS_BASE}/")
        print(f"  Log     -> {log_path}")


if __name__ == "__main__":
    main()
