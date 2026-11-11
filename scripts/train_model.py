import os
import shutil
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

from LUMIA.ml.models import ResNetLiteClassifier

# ==========================================
# OPTIMIERTE TRAININGSPARAMETER
# ==========================================
INITIAL_LR = 1e-4
EPOCHS = 12
BATCH_SIZE = 1024
TOP_K_MODELS = 5

# Gewichtung fuer kombinierten Tomo-e Loss (BCE + Pairwise AUC/Hinge Loss)
LAMBDA_CE = 1.0
LAMBDA_AUC = 0.5

H5_PATH = "data/ground_truth/tomoe_training_dataset.h5"
MODEL_DIR = "data/models"
DEFAULT_BEST_NAME = "resnet_lite_tomoe.pth"
# ==========================================


class TomoeHybridLoss(nn.Module):
    """Kombiniert BCEWithLogitsLoss mit dem paarweisen exp-Cross-Hinge Loss (Takahashi et al. 2022)."""
    def __init__(self, lambda_ce=1.0, lambda_auc=0.5, margin=1.0, gamma=1.0):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_auc = lambda_auc
        self.margin = margin
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)

        loss_ce = self.bce(logits_flat, targets_flat)

        pos_mask = (targets_flat == 1)
        neg_mask = (targets_flat == 0)

        if self.lambda_auc > 0 and pos_mask.any() and neg_mask.any():
            pos_logits = logits_flat[pos_mask].unsqueeze(1)  # (N_pos, 1)
            neg_logits = logits_flat[neg_mask].unsqueeze(0)  # (1, N_neg)

            # Paarweise Differenz: s_pos - s_neg soll groesser als margin sein
            diff = pos_logits - neg_logits
            hinge = torch.clamp(self.margin - diff, min=0.0)
            # Clamp auf 10.0 verhindert float16 Overflow unter AMP autocast
            loss_ech = torch.exp(-self.gamma * torch.clamp(hinge, max=10.0)).mean()
            return self.lambda_ce * loss_ce + self.lambda_auc * loss_ech

        return loss_ce


class TopKModelManager:
    """Verwaltet die besten K Modelle auf der Festplatte und benennt sie eindeutig."""
    def __init__(self, model_dir: Path, k: int = 5, primary_metric: str = "f1", maximize: bool = True):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.primary_metric = primary_metric
        self.maximize = maximize
        self.top_models = []  # Liste aus dicts: {'score', 'epoch', 'path', 'metrics'}

    def update(self, model: nn.Module, epoch: int, metrics: dict, default_best_name: str):
        score = metrics[self.primary_metric]
        unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model

        # Pruefen, ob das Modell in die Top K gehoert
        qualifies = len(self.top_models) < self.k
        if not qualifies:
            worst_score = self.top_models[-1]["score"]
            qualifies = (score > worst_score) if self.maximize else (score < worst_score)

        if not qualifies:
            return False

        # Temporaer speichern, danach alle Top-K nach neuem Rang umbenennen
        temp_path = self.model_dir / f"temp_ep{epoch:02d}.pth"
        torch.save(unwrapped.state_dict(), temp_path)

        self.top_models.append({
            "score": score,
            "epoch": epoch,
            "path": temp_path,
            "metrics": metrics
        })

        # Sortieren nach Score
        self.top_models.sort(key=lambda x: x["score"], reverse=self.maximize)

        # Ueberzaehlige Modelle (Platz 6+) von der Festplatte loeschen
        while len(self.top_models) > self.k:
            removed = self.top_models.pop()
            if removed["path"].exists():
                removed["path"].unlink()

        # Eindeutige Dateinamen mit Rang, Epoche und Kern-Metriken vergeben
        for idx, item in enumerate(self.top_models):
            rank = idx + 1
            m = item["metrics"]
            ep = item["epoch"]
            new_name = (
                f"rank{rank}_ep{ep:02d}_"
                f"f1-{m['f1']:.4f}_"
                f"fpr90-{m['fpr_90']:.4f}_"
                f"auc-{m['roc_auc']:.4f}.pth"
            )
            new_path = self.model_dir / new_name
            if item["path"] != new_path:
                if new_path.exists():
                    new_path.unlink()
                item["path"].rename(new_path)
                item["path"] = new_path

        # Platz 1 zusaetzlich als Standard-Modell kopieren
        best_default_path = self.model_dir / default_best_name
        shutil.copy(self.top_models[0]["path"], best_default_path)
        return True


class TomoeHDF5Dataset(Dataset):
    def __init__(self, h5_path, split="train"):
        self.h5_path = h5_path
        self.split = split
        self.file = None

        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f[self.split]['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        cutout = torch.tensor(self.file[self.split]['cutouts'][idx], dtype=torch.float32)
        label = torch.tensor(self.file[self.split]['labels'][idx], dtype=torch.float32)

        if self.split == "train":
            if torch.rand(1) > 0.5:
                cutout = torch.flip(cutout, dims=[2])
            if torch.rand(1) > 0.5:
                cutout = torch.flip(cutout, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                cutout = torch.rot90(cutout, k, dims=[1, 2])

        return cutout, label


def evaluate_validation(model, val_loader, device):
    """Berechnet alle relevanten Astrophysik- und ML-Metriken."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for cutouts, labels in val_loader:
            cutouts = cutouts.to(device)
            logits = model.forward_logits(cutouts).squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    # 1. FPR bei festem Recall (TPR = 0.80 und TPR = 0.90)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fpr_80 = float(fpr[np.searchsorted(tpr, 0.80, side="left")])
    fpr_90 = float(fpr[np.searchsorted(tpr, 0.90, side="left")])

    # 2. PR-Kurve fuer optimalen F1, F0.5 und MCC
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    p = precisions[:-1]
    r = recalls[:-1]

    f1_scores = (2 * p * r) / (p + r + 1e-10)
    f05_scores = (1.25 * p * r) / (0.25 * p + r + 1e-10)

    best_f1_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_f1_idx])
    best_f05 = float(np.max(f05_scores))
    best_thresh = float(thresholds[best_f1_idx])
    best_prec = float(p[best_f1_idx])
    best_rec = float(r[best_f1_idx])

    # 3. Matthews Correlation Coefficient (MCC) am besten F1-Punkt
    preds = (all_probs >= best_thresh).astype(int)
    tp = float(np.sum((preds == 1) & (all_labels == 1)))
    tn = float(np.sum((preds == 0) & (all_labels == 0)))
    fp = float(np.sum((preds == 1) & (all_labels == 0)))
    fn = float(np.sum((preds == 0) & (all_labels == 1)))
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float(((tp * tn) - (fp * fn)) / mcc_denom) if mcc_denom > 0 else 0.0

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": best_f1,
        "f05": best_f05,
        "mcc": mcc,
        "prec": best_prec,
        "rec": best_rec,
        "fpr_80": fpr_80,
        "fpr_90": fpr_90,
        "thresh": best_thresh,
    }


def train_model():
    model_dir_path = Path(MODEL_DIR)
    keeper = TopKModelManager(model_dir_path, k=TOP_K_MODELS, primary_metric="f1", maximize=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starte Training auf Device: {DEVICE}")

    train_dataset = TomoeHDF5Dataset(H5_PATH, split="train")
    test_dataset = TomoeHDF5Dataset(H5_PATH, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=14, pin_memory=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=14, pin_memory=True, prefetch_factor=4
    )

    model = ResNetLiteClassifier().to(DEVICE)
    print("Kompiliere Modell mit torch.compile()...")
    model = torch.compile(model)

    criterion = TomoeHybridLoss(lambda_ce=LAMBDA_CE, lambda_auc=LAMBDA_AUC)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoche {epoch}/{EPOCHS}")

            for cutouts, labels in progress_bar:
                cutouts, labels = cutouts.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    logits = model.forward_logits(cutouts)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            scheduler.step()
            avg_loss = running_loss / len(train_loader)

            print(f"\nEvaluiere Epoche {epoch} auf Test-Set...")
            m = evaluate_validation(model, val_loader, DEVICE)

            print(
                f"Epoche {epoch:02d} | Loss: {avg_loss:.4f} | "
                f"F1: {m['f1']:.4f} (P: {m['prec']:.4f}, R: {m['rec']:.4f}) | "
                f"F0.5: {m['f05']:.4f} | MCC: {m['mcc']:.4f} | "
                f"PR-AUC: {m['pr_auc']:.4f} | ROC-AUC: {m['roc_auc']:.4f} | "
                f"FPR@90: {m['fpr_90']:.4f} | FPR@80: {m['fpr_80']:.4f}"
            )

            saved = keeper.update(model, epoch, m, DEFAULT_BEST_NAME)
            if saved:
                print(f" -> In Top-{TOP_K_MODELS} aufgenommen! Aktuelle Rangliste:")
                for idx, entry in enumerate(keeper.top_models):
                    print(f"    #{idx + 1}: {entry['path'].name}")

    except KeyboardInterrupt:
        print("\n[!] Training durch Benutzer (Strg+C) sanft abgebrochen!")

    print("\nTraining beendet! Top-Modelle liegen unter:")
    for idx, entry in enumerate(keeper.top_models):
        print(f"  #{idx + 1}: {entry['path']}")


if __name__ == "__main__":
    train_model()