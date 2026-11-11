import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from tqdm import tqdm

from LUMIA.ml.models import ResNetLiteClassifier

# ==========================================
# KONFIGURATION & PFADE
# ==========================================
H5_PATH = "data/ground_truth/tomoe_training_dataset.h5"
WEIGHTS_PATH = "data/models/resnet_lite_tomoe.pth"
PLOT_SAVE_DIR = "data/models"
BATCH_SIZE = 1024
# ==========================================


class TomoeHDF5Dataset(Dataset):
    def __init__(self, h5_path, split="test"):
        self.h5_path = h5_path
        self.split = split
        self.file = None

        with h5py.File(self.h5_path, 'r') as f:
            if self.split not in f:
                raise ValueError(f"Split '{self.split}' nicht in HDF5 gefunden.")
            self.length = len(f[self.split]['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        cutout = self.file[self.split]['cutouts'][idx]
        label = self.file[self.split]['labels'][idx]

        return torch.tensor(cutout, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def plot_metrics(all_labels, all_probs, roc_auc, pr_auc, optimal_threshold, save_dir):
    """Erstellt professionelle ROC- und PR-Plots und speichert sie."""
    os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: ROC Curve
    ax1.plot(fpr, tpr, color='cyan', lw=2, label=f'ResNet-Lite (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Zufall (AUC = 0.5000)')
    ax1.axhline(0.90, color='yellow', lw=1, linestyle=':', label='Ziel-TPR = 0.90')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=15)
    ax1.set_xlabel('False Positive Rate (Rauschen als Transiente)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (Transienten gefunden)', fontsize=12)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(alpha=0.2)

    # Plot 2: Precision-Recall Curve
    ax2.plot(recalls, precisions, color='magenta', lw=2, label=f'PR-Kurve (AUC = {pr_auc:.4f})')

    if len(thresholds) > 0:
        f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        opt_idx = np.argmax(f1_scores)
        opt_rec = recalls[opt_idx]
        opt_prec = precisions[opt_idx]

        ax2.plot(opt_rec, opt_prec, marker='o', color='white', markersize=8,
                 label=f'Optimal (Thresh={optimal_threshold:.4f}\nF1={f1_scores[opt_idx]:.4f})')

    ax2.set_title('Precision-Recall Curve (PRC)', fontsize=14, pad=15)
    ax2.set_xlabel('Recall (Vollstaendigkeit)', fontsize=12)
    ax2.set_ylabel('Precision (Reinheit)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nVisuelle Evaluierung gespeichert unter: {plot_path}")
    plt.close()


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for cutouts, labels in tqdm(dataloader, desc="Evaluierung"):
            cutouts, labels = cutouts.to(device), labels.to(device)

            logits = model.forward_logits(cutouts).squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    pos_mask = (all_labels == 1)
    neg_mask = (all_labels == 0)
    print("\n--- DIAGNOSE-CHECK ---")
    print(f"Gesamt Samples: {len(all_labels)} | Positive: {pos_mask.sum()} | Negative: {neg_mask.sum()}")
    if pos_mask.sum() > 0:
        print(f"Mittlere Vorhersage fuer ECHTE Transienten: {all_probs[pos_mask].mean():.6f}")
    if neg_mask.sum() > 0:
        print(f"Mittlere Vorhersage fuer RAUSCHEN: {all_probs[neg_mask].mean():.6f}")

    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
    idx_90 = np.searchsorted(tpr, 0.90, side="left")
    idx_80 = np.searchsorted(tpr, 0.80, side="left")
    fpr_90 = fpr[idx_90]
    fpr_80 = fpr[idx_80]

    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    p, r = precisions[:-1], recalls[:-1]
    f1_scores = (2 * p * r) / (p + r + 1e-10)
    f05_scores = (1.25 * p * r) / (0.25 * p + r + 1e-10)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    best_f05 = np.max(f05_scores)

    preds = (all_probs >= optimal_threshold).astype(int)
    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)

    tp = int(np.sum((preds == 1) & (all_labels == 1)))
    tn = int(np.sum((preds == 0) & (all_labels == 0)))
    fp = int(np.sum((preds == 1) & (all_labels == 0)))
    fn = int(np.sum((preds == 0) & (all_labels == 1)))
    mcc_denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0

    print("\n" + "=" * 52)
    print("              EVALUIERUNGSERGEBNISSE")
    print("=" * 52)
    print(f"Optimaler Threshold (Max-F1): {optimal_threshold:.6f}")
    print(f"Konfusionsmatrix:             TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print("-" * 52)
    print(f"Accuracy:                     {acc:.4f}")
    print(f"Precision (Reinheit):         {prec:.4f}")
    print(f"Recall (Vollstaendigkeit):    {rec:.4f}")
    print(f"F1-Score (Ausgewogen):        {f1:.4f}")
    print(f"F0.5-Score (Precision-Fokus): {best_f05:.4f}")
    print(f"MCC (Korrelations-Score):     {mcc:.4f}")
    print(f"PR-AUC (Average Precision):   {pr_auc:.4f}")
    print(f"ROC-AUC:                      {roc_auc:.4f}")
    print("-" * 52)
    print(f"FPR @ TPR=0.90 (90% Recall):  {fpr_90:.5f} (~{int(fpr_90 * neg_mask.sum())} Fehlalarme)")
    print(f"FPR @ TPR=0.80 (80% Recall):  {fpr_80:.5f} (~{int(fpr_80 * neg_mask.sum())} Fehlalarme)")
    print("=" * 52)

    plot_metrics(all_labels, all_probs, roc_auc, pr_auc, optimal_threshold, PLOT_SAVE_DIR)


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Nutze Compute Device: {DEVICE}")

    test_dataset = TomoeHDF5Dataset(H5_PATH, split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=14, pin_memory=True, prefetch_factor=4
    )

    model = ResNetLiteClassifier().to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        print(f"Lade Gewichte: {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))

    evaluate_model(model, test_loader, DEVICE)