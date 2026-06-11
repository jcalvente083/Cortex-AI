"""
train_resnet10.py — Fine-tuning ResNet10t sobre mel-espectrogramas (4 actividades de habla).

Arquitectura: resnet10t (timm) — ~5.4M params, pretrained ImageNet.
Sirve como ablación frente a ResNet18 para evaluar si la mayor capacidad
del modelo justifica el coste computacional adicional.

Actividades: vocal | frase | espontanea | all
Pipeline por actividad (idéntico a train_resnet.py):
  1. Carga spectrograms_meta.csv alineado con train_80.csv / holdout_20.csv
  2. AudioDataset: pre-carga espectrogramas en memoria, frames de 1.0s / hop 0.5s
  3. Split interno de PACIENTES 80/20:
       - cv_patients  (80%) -> entrenamiento con early stopping (val_loss)
       - val_patients (20%) -> umbral Youden a nivel de paciente
  4. Umbral optimo: criterio de Youden sobre probabilidades promediadas por paciente
  5. Evaluacion sobre holdout externo: probabilidades de frames -> media por paciente

Salidas:
  models/resnet/ResNet10/{run}/{actividad}/    -> model.pth, threshold.pkl
  reports/resnet/ResNet10/{run}/{actividad}/   -> 6 figuras por modelo
  reports/resnet/ResNet10/{run}/comparison_4models.png
  logs/resnet/ResNet10/train_{run}_{timestamp}.log

Uso:
  uv run python -m src.models.train_resnet10 --run baseline
  uv run python -m src.models.train_resnet10 --run specaugment
  uv run python -m src.models.train_resnet10 --run specaugment_freeze --freeze
  uv run python -m src.models.train_resnet10 --run age_matched --age-match
  uv run python -m src.models.train_resnet10 --run age_matched_freeze --age-match --freeze
"""

import argparse
import random
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
from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    recall_score, precision_score, f1_score,
    accuracy_score, balanced_accuracy_score,
)

from src.config import SEED, PALETTE, apply_style, SR
from src.features.spectrograms import (
    FRAME_LEN_S, HOP_LEN_S, N_MELS, N_FFT, N_FFT_HOP_S,
)
from src.utils.results_logger import save_run_json

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
SPECTRO_CSV  = Path("data/processed/combined/spectrograms_meta.csv")
TRAIN_CSV    = Path("data/processed/combined/train_80.csv")
HOLDOUT_CSV  = Path("data/processed/combined/holdout_20.csv")
REPORTS_BASE = Path("reports/resnet/ResNet10")
MODELS_BASE  = Path("models/resnet/ResNet10")
LOG_DIR      = Path("logs/resnet/ResNet10")

# =============================================================================
# HIPERPARAMETROS — idénticos a ResNet18 para comparación justa
# =============================================================================
RESIZE       = 224
EPOCHS       = 30
BATCH_SIZE   = 32
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 7
LR_PATIENCE  = 3

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
    def isatty(self):
        return False

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
# DATASET
# =============================================================================
class AudioDataset(Dataset):
    def __init__(self, df_meta: pd.DataFrame, transform=None, training: bool = False):
        self.transform   = transform
        self.training    = training
        self.frames      = []
        self.targets     = []
        self.patient_ids = []

        frame_len = int(SR * FRAME_LEN_S)
        hop_len   = int(SR * HOP_LEN_S)
        n_fft_hop = int(SR * N_FFT_HOP_S)
        n_skipped = 0

        for _, row in tqdm(df_meta.iterrows(), total=len(df_meta),
                           desc="Cargando espectrogramas", leave=False, ncols=100, ascii=True):
            try:
                y, _ = librosa.load(row['audio_path'], sr=SR, mono=True)

                if len(y) < int(SR * 0.1) or np.max(np.abs(y)) < 1e-4:
                    n_skipped += 1
                    continue

                if len(y) >= frame_len:
                    y_framed = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
                else:
                    y_pad    = np.concatenate([y, np.zeros(frame_len - len(y))])
                    y_framed = librosa.util.frame(y_pad, frame_length=frame_len, hop_length=hop_len)

                mels      = librosa.feature.melspectrogram(
                    y=y_framed.T, sr=SR, n_fft=N_FFT, hop_length=n_fft_hop, n_mels=N_MELS,
                )
                mels_db   = librosa.power_to_db(mels, ref=np.max)
                mels_norm = (mels_db - mels_db.mean()) / (mels_db.std() + 1e-8)

                target     = int(row['Target'])
                patient_id = str(row['ID_Paciente'])

                for i in range(mels_norm.shape[0]):
                    self.frames.append(mels_norm[i].astype(np.float32))
                    self.targets.append(target)
                    self.patient_ids.append(patient_id)

            except Exception:
                n_skipped += 1

        if n_skipped:
            print(f"  [!] Audios omitidos en carga: {n_skipped}")

    def _spec_augment(self, spec: torch.Tensor,
                      freq_mask_param: int = 15,
                      time_mask_param: int = 20,
                      n_freq_masks: int = 2,
                      n_time_masks: int = 2) -> torch.Tensor:
        _, n_mels, n_time = spec.shape
        for _ in range(n_freq_masks):
            f  = random.randint(0, freq_mask_param)
            f0 = random.randint(0, max(0, n_mels - f))
            spec[:, f0:f0 + f, :] = 0
        for _ in range(n_time_masks):
            t  = random.randint(0, min(time_mask_param, n_time))
            t0 = random.randint(0, max(0, n_time - t))
            spec[:, :, t0:t0 + t] = 0
        return spec

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.frames[idx]).unsqueeze(0)
        if self.transform:
            tensor = self.transform(tensor)
        if self.training:
            tensor = self._spec_augment(tensor)
        return tensor, self.targets[idx], self.patient_ids[idx]


# =============================================================================
# TRAINER
# =============================================================================
class ResNet10Trainer:

    def __init__(self, activity_name: str, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                 freeze: bool = False):
        self.activity_name = activity_name
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.freeze        = freeze

        self.models_dir  = MODELS_BASE  / activity_name
        self.reports_dir = REPORTS_BASE / activity_name
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Dispositivo: {self.device}")

        self.model             = None
        self.optimal_threshold = 0.5
        self.history           = {'train_loss': [], 'val_loss': [],
                                   'train_acc':  [], 'val_acc':  []}
        self.metrics_internal_ = {}
        self.metrics_external_ = {}

        self._y_val       = None
        self._y_pred_val  = None
        self._y_proba_val = None
        self._y_ext       = None
        self._y_pred_ext  = None
        self._y_proba_ext = None
        self._sample_specs = {}

    # ─── Modelo ──────────────────────────────────────────────────────────────
    def _build_model(self) -> nn.Module:
        # in_chans=1 lets timm adapt the tiered stem (Sequential) natively
        model = timm.create_model("resnet10t", pretrained=True, num_classes=2, in_chans=1)
        if self.freeze:
            for module in [model.conv1, model.layer1, model.layer2]:
                for param in module.parameters():
                    param.requires_grad = False
            if hasattr(model, "bn1"):
                for param in model.bn1.parameters():
                    param.requires_grad = False
        return model.to(self.device)

    # ─── DataLoader ──────────────────────────────────────────────────────────
    def _make_loader(self, df_meta: pd.DataFrame, shuffle: bool, training: bool = False):
        transform = transforms.Resize((RESIZE, RESIZE), antialias=True)
        dataset   = AudioDataset(df_meta, transform=transform, training=training)
        loader    = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        return loader, dataset

    # ─── Epoch ───────────────────────────────────────────────────────────────
    def _run_epoch(self, loader: DataLoader, optimizer, criterion, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for inputs, targets, _ in loader:
                inputs  = inputs.to(self.device)
                targets = torch.tensor(targets, dtype=torch.long).to(self.device) \
                          if not isinstance(targets, torch.Tensor) \
                          else targets.to(torch.long).to(self.device)

                if train:
                    optimizer.zero_grad()

                outputs = self.model(inputs)
                loss    = criterion(outputs, targets)

                if train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total   += targets.size(0)

        return total_loss / total, correct / total

    # ─── Inferencia paciente-nivel ────────────────────────────────────────────
    def _infer_patient_level(self, dataset: AudioDataset) -> pd.DataFrame:
        loader = DataLoader(dataset, batch_size=self.batch_size * 2,
                            shuffle=False, num_workers=0)
        self.model.eval()
        all_proba, all_targets, all_pids = [], [], []

        with torch.no_grad():
            for inputs, targets, pids in loader:
                inputs = inputs.to(self.device)
                probs  = F.softmax(self.model(inputs), dim=1)[:, 1]
                all_proba.extend(probs.cpu().numpy().tolist())
                all_targets.extend(targets.tolist() if isinstance(targets, torch.Tensor)
                                   else list(targets))
                all_pids.extend(list(pids))

        df = pd.DataFrame({
            'ID_Paciente': all_pids,
            'proba':       all_proba,
            'Target':      all_targets,
        })
        return df.groupby('ID_Paciente').agg(
            proba=('proba', 'mean'),
            Target=('Target', 'first'),
        ).reset_index()

    # ─── Entrenamiento principal ──────────────────────────────────────────────
    def fit(self, df_train: pd.DataFrame, df_holdout: pd.DataFrame) -> "ResNet10Trainer":
        n_pac_tr = df_train['ID_Paciente'].nunique()
        n_pac_ho = df_holdout['ID_Paciente'].nunique()

        print(f"\n{'='*60}")
        print(f"  ENTRENAMIENTO — ResNet10 — [{self.activity_name.upper()}]")
        print(f"{'='*60}")
        print(f"  Audios train:   {len(df_train):>5}  ({n_pac_tr} pacientes)")
        print(f"  Audios holdout: {len(df_holdout):>5}  ({n_pac_ho} pacientes)")

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

        print("\n  Generando datasets (pre-carga en memoria)...")
        train_loader, _            = self._make_loader(df_cv,      shuffle=True,  training=True)
        val_loader,   val_dataset  = self._make_loader(df_val,     shuffle=False, training=False)
        _,         holdout_dataset = self._make_loader(df_holdout, shuffle=False, training=False)

        if len(train_loader.dataset) == 0:
            print("  [!] No hay frames de entrenamiento. Abortando.")
            return self

        print(f"  Frames CV     : {len(train_loader.dataset):,}")
        print(f"  Frames Val    : {len(val_dataset):,}")
        print(f"  Frames Holdout: {len(holdout_dataset):,}")

        self._collect_samples(val_dataset)

        self.model = self._build_model()
        n_params   = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Parametros entrenables: {n_params:,}")

        criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer  = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY,
        )
        scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, min_lr=1e-6,
        )

        best_val_loss = float('inf')
        epochs_no_imp = 0
        temp_path     = self.models_dir / "_best_temp.pth"

        print(f"\n  Entrenando {self.epochs} epocas (early stopping patience={PATIENCE})...")
        print(f"  {'Epoca':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
              f"{'Train Acc':>9} | {'Val Acc':>9} | {'LR':>8}")
        print(f"  {'─'*62}")

        for epoch in range(1, self.epochs + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, optimizer, criterion, train=True)
            vl_loss, vl_acc = self._run_epoch(val_loader,   optimizer, criterion, train=False)

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(vl_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_acc'].append(vl_acc)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"  {epoch:>5} | {tr_loss:>10.4f} | {vl_loss:>10.4f} | "
                  f"{tr_acc:>9.4f} | {vl_acc:>9.4f} | {current_lr:>8.2e}")

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
            self.model.load_state_dict(torch.load(temp_path, map_location=self.device,
                                                   weights_only=True))
            temp_path.unlink()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n  Calculando umbral Youden (val interna, nivel paciente)...")
        pat_val_df = self._infer_patient_level(val_dataset)
        fpr, tpr, ths = roc_curve(pat_val_df['Target'], pat_val_df['proba'])
        finite = np.isfinite(ths)
        if finite.any():
            best_idx = np.argmax((tpr - fpr)[finite])
            self.optimal_threshold = float(ths[finite][best_idx])
        else:
            self.optimal_threshold = 0.5
        print(f"  Umbral optimo: {self.optimal_threshold:.4f}")

        pat_hol_df = self._infer_patient_level(holdout_dataset)
        self._evaluate_pac(pat_val_df, "VAL INTERNA (paciente)",    internal=True)
        self._evaluate_pac(pat_hol_df, "HOLDOUT EXTERNO (paciente)", internal=False)

        self._save()
        apply_style()
        self._plot_all()
        return self

    def _collect_samples(self, dataset: AudioDataset, n_per_class: int = 3):
        self._sample_specs = {0: [], 1: []}
        for i in range(len(dataset)):
            spec, target, _ = dataset[i]
            cls = int(target)
            if len(self._sample_specs[cls]) < n_per_class:
                self._sample_specs[cls].append(spec.squeeze(0).numpy())
            if all(len(v) >= n_per_class for v in self._sample_specs.values()):
                break

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
        torch.save(self.model.state_dict(),       self.models_dir / "model.pth")
        joblib.dump(self.optimal_threshold,        self.models_dir / "threshold.pkl")
        print(f"\n  Artefactos guardados en: {self.models_dir}/")

    # ─── Plots ────────────────────────────────────────────────────────────────
    def _plot_all(self):
        self._plot_confusion()
        self._plot_roc()
        self._plot_training_curves()
        self._plot_probability_distribution()
        self._plot_spectrogram_examples()
        self._plot_metrics_summary()

    def _plot_confusion(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Matriz de Confusion — ResNet10 [{self.activity_name}]",
                     fontsize=14, fontweight="bold", y=1.02)
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
                auc_val = auc(fpr, tpr)
                best_i  = np.argmax(tpr - fpr)
                ax.plot(fpr, tpr, color=color, lw=2.5,
                        label=f"{label} (AUC={auc_val:.4f})")
                ax.scatter(fpr[best_i], tpr[best_i], s=100, color=color, zorder=5)
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], ":", color=PALETTE["subtext"], lw=2)
        ax.set_xlabel("FPR (1-Especificidad)")
        ax.set_ylabel("TPR (Sensibilidad)")
        ax.set_title(f"Curva ROC — ResNet10 [{self.activity_name}]", fontweight="bold")
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
        axes[1].set_ylabel("Accuracy (frame-level)")
        axes[1].legend()

        plt.tight_layout()
        self._save_fig("3_training_curves")

    def _plot_probability_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Distribucion de Probabilidades — ResNet10 [{self.activity_name}]",
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

    def _plot_spectrogram_examples(self):
        if not self._sample_specs or not any(self._sample_specs.values()):
            return

        n_cols  = max(len(v) for v in self._sample_specs.values())
        labels  = {0: "Control (HC)", 1: "Parkinson (PD)"}
        colors  = {0: PALETTE["accent2"], 1: PALETTE["accent3"]}

        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
        fig.suptitle(
            f"Ejemplos de Mel-Espectrograma — ResNet10 [{self.activity_name}]",
            fontweight="bold", fontsize=13,
        )
        for row_idx, cls in enumerate([0, 1]):
            specs = self._sample_specs.get(cls, [])
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx] if n_cols > 1 else axes[row_idx]
                if col_idx < len(specs):
                    ax.imshow(specs[col_idx], aspect='auto', origin='lower',
                              cmap='magma', interpolation='nearest')
                    ax.set_title(f"{labels[cls]} #{col_idx+1}",
                                 color=colors[cls], fontweight="bold")
                    ax.set_xlabel("Tiempo (frames)")
                    ax.set_ylabel("Mel bins")
                else:
                    ax.axis('off')

        plt.tight_layout()
        self._save_fig("5_spectrogram_examples")

    def _plot_metrics_summary(self):
        m_i = self.metrics_internal_
        m_e = self.metrics_external_
        fig = plt.figure(figsize=(13, 5.5))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.8])

        cats   = ["Recall\n(Sensib.)", "Especif.", "Precision", "F1", "AUC-ROC"]
        vals_i = [m_i["recall"], m_i["specificity"], m_i["precision"], m_i["f1"], m_i["roc_auc"]]
        vals_e = [m_e["recall"], m_e["specificity"], m_e["precision"], m_e["f1"], m_e["roc_auc"]]
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
            ["Metrica",            "Val interna",                      "Holdout externo"],
            ["Balanced Accuracy",  f"{m_i['balanced_accuracy']:.4f}",  f"{m_e['balanced_accuracy']:.4f}"],
            ["Recall (PD)",        f"{m_i['recall']:.4f}",             f"{m_e['recall']:.4f}"],
            ["Especificidad (HC)", f"{m_i['specificity']:.4f}",        f"{m_e['specificity']:.4f}"],
            ["F1-Score",           f"{m_i['f1']:.4f}",                 f"{m_e['f1']:.4f}"],
            ["AUC-ROC",            f"{m_i['roc_auc']:.4f}",            f"{m_e['roc_auc']:.4f}"],
            ["Accuracy",           f"{m_i['accuracy']:.4f}",           f"{m_e['accuracy']:.4f}"],
            ["Umbral Youden",      f"{self.optimal_threshold:.4f}",    "—"],
            ["Epocas entrenadas",  f"{len(self.history['train_loss'])}", "—"],
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
# COMPARACION FINAL — 4 ACTIVIDADES (holdout externo)
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
    ax.set_title("Comparacion ResNet10 — Holdout externo (4 actividades)",
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
# AGE-MATCHING
# =============================================================================
def _age_match_neurovoz(df: pd.DataFrame) -> pd.DataFrame:
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
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',        default='baseline',
                        help='Nombre del run (define subcarpeta de salida)')
    parser.add_argument('--epochs',     type=int, default=EPOCHS,
                        help=f'Numero de epocas (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--age-match',  action='store_true',
                        help='Age-matching 1:1 (±5 años) en NeuroVoz para eliminar confound demográfico')
    parser.add_argument('--freeze',     action='store_true',
                        help='Congela conv1, bn1, layer1, layer2')
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

        for path, hint in [
            (SPECTRO_CSV,  "uv run python -m src.features.spectrograms"),
            (TRAIN_CSV,    "uv run python -m src.features.split_dataset"),
            (HOLDOUT_CSV,  "uv run python -m src.features.split_dataset"),
        ]:
            if not path.exists():
                print(f"[!] No encontrado: {path}")
                print(f"    Ejecuta: {hint}")
                return

        print(f"{'='*60}")
        print("  Cortex-AI — Entrenamiento ResNet10 (4 actividades)")
        print(f"  Inicio  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Epochs  : {args.epochs} | Batch: {args.batch_size} | LR: {LR}")
        print(f"  Device  : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"{'='*60}")

        df_meta    = pd.read_csv(SPECTRO_CSV)
        df_train   = pd.read_csv(TRAIN_CSV)
        df_holdout = pd.read_csv(HOLDOUT_CSV)

        if args.age_match:
            print("\n  [Age-match activo]")
            df_meta = _age_match_neurovoz(df_meta)
            print(f"  Tras age-match: {len(df_meta)} audios | "
                  f"{df_meta['ID_Paciente'].nunique()} pacientes")

        train_paths   = set(df_train['audio_path'].tolist())
        holdout_paths = set(df_holdout['audio_path'].tolist())

        df_train_meta   = df_meta[df_meta['audio_path'].isin(train_paths)].copy()
        df_holdout_meta = df_meta[df_meta['audio_path'].isin(holdout_paths)].copy()

        print(f"\nMeta train  : {len(df_train_meta)} audios, "
              f"{df_train_meta['ID_Paciente'].nunique()} pacientes")
        print(f"Meta holdout: {len(df_holdout_meta)} audios, "
              f"{df_holdout_meta['ID_Paciente'].nunique()} pacientes")

        results    = {}
        thresholds = {}
        for activity_name, activity_types in ACTIVITIES.items():
            print(f"\n\n{'#'*60}")
            print(f"# ACTIVIDAD: {activity_name.upper()}")
            print(f"{'#'*60}")

            df_tr = df_train_meta[
                df_train_meta['activity_type'].isin(activity_types)
            ].copy()
            df_ho = df_holdout_meta[
                df_holdout_meta['activity_type'].isin(activity_types)
            ].copy()

            n_pac_tr = df_tr['ID_Paciente'].nunique()
            if n_pac_tr < 20:
                print(f"  [!] Solo {n_pac_tr} pacientes en train — actividad omitida.")
                continue

            print(f"  Train  : {len(df_tr)} aud. / {n_pac_tr} pac.")
            print(f"  Holdout: {len(df_ho)} aud. / {df_ho['ID_Paciente'].nunique()} pac.")

            trainer = ResNet10Trainer(
                activity_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                freeze=args.freeze,
            )
            trainer.fit(df_tr, df_ho)

            if trainer.metrics_internal_ and trainer.metrics_external_:
                results[activity_name] = {
                    "internal": trainer.metrics_internal_,
                    "external": trainer.metrics_external_,
                }
                thresholds[activity_name] = trainer.optimal_threshold

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

            save_run_json("ResNet10", args.run, results, thresholds)

        print(f"\n  Completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Modelos -> {MODELS_BASE}/")
        print(f"  Figuras -> {REPORTS_BASE}/")
        print(f"  Log     -> {log_path}")


if __name__ == "__main__":
    main()
