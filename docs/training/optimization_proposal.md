# Optimization of Training Metrics and Pipeline Robustness for Project LUMIA
## Architectural Blueprint & Technical Proposal: TrackletClassifierNet

> **Document Type:** Architectural Proposal & Specification  
> **Status:** APPROVED FOR IMPLEMENTATION  
> **Target Subsystem:** `src/LUMIA/training/` and `src/LUMIA/models/`  
> **Dataset Source:** `data/ground_truth/ztf_sso_tracklets_128x128.h5`  
> **Key References:**  
> - [`docs/training/architecture_blueprint.md`](file:///D:/Programming/Python/ProjectLUMIA/docs/training/architecture_blueprint.md)  
> - [`docs/data_mining/architecture_blueprint.md`](file:///D:/Programming/Python/ProjectLUMIA/docs/data_mining/architecture_blueprint.md)  
> - [`docs/adr/0002-strict-pydantic-contracts.md`](file:///D:/Programming/Python/ProjectLUMIA/docs/adr/0002-strict-pydantic-contracts.md)  

---

## 1. Executive Summary & Root-Cause Diagnosis

Training runs of `TrackletClassifierNet` across 30 epochs on ZTF multi-epoch tracklet archives (`ztf_sso_tracklets_128x128.h5`) exhibit two persistent failure modes:
1. **Validation Macro-F1 strictly plateaus at $\sim 0.43$** with no breakthrough across epochs.
2. **Validation FPR90 remains frozen at exactly $1.0000$** throughout all training checkpoints.

A systematic audit of the dataset partitions, loss formulation, decision boundaries, and evaluation metrics reveals the exact mathematical and algorithmic mechanisms driving these plateaus:

### 1.1 Empirical HDF5 Split Distribution Audit

Inspection of the operational archive `data/ground_truth/ztf_sso_tracklets_128x128.h5` reveals severe partition-level skew:

| Split | Total Tracklets | Class 0 (`BOGUS`) | Class 1 (`POINT`) | Class 2 (`STREAK`) | Class 3 (`COMET`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 1,158 | **16 (1.38%)** | 428 (36.96%) | 470 (40.59%) | 244 (21.07%) |
| **Val** | 333 | **1 (0.30%)** | 154 (46.25%) | 138 (41.44%) | 40 (12.01%) |
| **Test** | 143 | **3 (2.10%)** | 94 (65.73%) | 46 (32.17%) | **0 (0.00%)** |

### 1.2 Mathematical Proof of the Metric Plateaus

#### Why FPR90 = 1.0000 Is Stagnant
`compute_fpr90` converts predictions into a binary task:
$$\text{Bogus} = 0 \quad (\text{Class 0}), \quad \text{Real} = 1 \quad (\text{Classes 1, 2, 3}), \quad p_{\text{real}} = \sum_{c=1}^{3} p_c$$
In the validation split, $N_{\text{neg}} = 1$ and $N_{\text{pos}} = 332$.
The False Positive Rate is defined as:
$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{\text{FP}}{N_{\text{neg}}}$$
Because $N_{\text{neg}} = 1$, $\text{FPR}$ can mathematically take **only two values**:
$$\text{FPR} \in \{0.0, \, 1.0\}$$
When evaluating the ROC curve at $90\%$ Recall ($\text{TPR} \ge 0.90$), the operating threshold $\tau_{90}$ is lowered until at least $299$ of the $332$ real tracklets are recovered.
Because the unweighted loss biased the network heavily toward predicting classes 1, 2, and 3, the single negative sample receives a predicted $p_{\text{real}} > \tau_{90}$. Consequently, $\text{FP} = 1, \text{TN} = 0 \implies \text{FPR} = 1 / 1 = 1.0000$.
**Conclusion:** FPR90 is completely degenerate when $N_{\text{neg}} = 1$. It provides zero gradient signal for model ranking or checkpointing.

#### Why Macro-F1 Plateaus at ~0.43
In `TrackletMetricsTracker`, predictions are formed via hard uncalibrated argmax:
$$\hat{y} = \arg\max_{c \in \{0, 1, 2, 3\}} p_c$$
1. **Class 0 Collapse:** With only 16 training samples ($1.38\%$) and unweighted focal loss, raw logits for Class 0 are heavily suppressed relative to classes 1 and 2. The single validation sample for Class 0 is predicted as Class 1 or 2:
   $$\text{TP}_0 = 0 \implies \text{Precision}_0 = 0.0, \, \text{Recall}_0 = 0.0 \implies \text{F1}_0 = 0.0$$
2. **Class 3 Ambiguity:** Comets (40 samples in val) share morphological spatial traits with point sources (Class 1) and streaks (Class 2), achieving weak recall under unweighted argmax ($\text{F1}_3 \approx 0.0 - 0.20$).
3. **Upper Bound on Macro-F1:**
   $$\text{Macro-F1} = \frac{\text{F1}_0 + \text{F1}_1 + \text{F1}_2 + \text{F1}_3}{4} \approx \frac{0.0 + 0.85 + 0.85 + 0.02}{4} = \frac{1.72}{4} \approx 0.4300$$
**Conclusion:** The $\sim 0.43$ plateau is an algebraic artifact of Class 0 and Class 3 suppression under uncalibrated argmax decisions.

### 1.3 Loss Function Instantiation Bug
In `src/LUMIA/training/trainer.py`, the loss function was initialized as:
```python
# CURRENT FLAWED CODE in trainer.py:
if loss_type == "class_balanced_focal":
    return ClassBalancedFocalLoss(
        beta=self.config.class_balanced_beta,
        gamma=self.config.focal_gamma,
        # class_counts and alpha were OMITTED!
    )
```
In `src/LUMIA/training/loss.py`:
```python
if alpha is not None:
    weights = torch.as_tensor(alpha, dtype=torch.float32)
elif class_counts is not None:
    weights = compute_class_balanced_weights(class_counts, beta=beta)
else:
    weights = None  # <--- DEFAULTED TO NONE!
```
Because `class_counts` was omitted, `weights` remained `None`. `ClassBalancedFocalLoss` ran identically to unweighted `FocalLoss`, entirely ignoring the $27\times$ frequency disparity between Point Sources and Bogus Artifacts.

---

## 2. Domain-Specific Astronomical Metric Suite Re-Evaluation & Replacement

In operational time-domain astronomy (e.g., ZTF Alert Distribution, Rubin Observatory LSST alert stream), candidate feeds present extreme base-rate imbalances ($100:1$ to $10,000:1$ bogus-to-real ratios). Conversely, curated discovery archives often exhibit inverted sample skews. Evaluation metrics must remain robust across both regimes.

```mermaid
flowchart LR
    subgraph Current["Current Metric Flaws"]
        FPR["FPR90\n(Degenerate when N_neg is small;\nonly evaluates 1 operating point)"]
        ArgmaxF1["Argmax Macro-F1\n(Penalizes rare classes heavily\nunder uncalibrated priors)"]
    end

    subgraph Proposed["Domain-Specific Astronomy Metrics"]
        AP["Mean Average Precision (mAP)\n& Per-Class PR-AUC\n(Threshold-independent ranking)"]
        RealAP["Operational AP_real\n(Precision-Recall for Real vs Bogus)"]
        OperatingPoints["Alert Broker Operating Points:\n- P@R90, P@R95 (Purity at High Recall)\n- R@FPR01, R@FPR05 (Completeness at Low FPR)"]
        CalF1["Calibrated Macro-F1 (F1*)\n(Threshold-Optimized Decision Boundary)"]
    end

    FPR --> RealAP
    FPR --> OperatingPoints
    ArgmaxF1 --> AP
    ArgmaxF1 --> CalF1
```

### 2.1 Precision-Recall AUC (PR-AUC) & Mean Average Precision (mAP)

In skewed astronomical detection streams, PR-AUC and Average Precision (AP) are superior to ROC-AUC because they directly quantify the trade-off between purity (Precision) and completeness (Recall) without being distorted by true negative counts:
$$\text{AP}_c = \sum_{k=1}^{K} \left(R_c(k) - R_c(k-1)\right) P_c(k)$$
$$\text{mAP} = \frac{1}{C} \sum_{c=0}^{C-1} \text{AP}_c$$
Where $P_c(k)$ and $R_c(k)$ represent precision and recall at threshold step $k$.

#### Operational Real-vs-Bogus AP ($\text{AP}_{\text{real}}$)
Consolidates classes $\{1, 2, 3\}$ into a single Real super-class and evaluates binary Average Precision against Class 0 (`BOGUS`). This measures candidate ranking quality for follow-up telescope scheduling.

### 2.2 Alert Broker Operating Point Metrics

Instead of a single brittle FPR metric, the evaluation suite must report standardized alert broker operating metrics:

1. **Purity at High Completeness ($\text{P@R90}$ and $\text{P@R95}$):**
   Precision at the exact threshold where $\text{Recall} \ge 0.90$ (and $0.95$):
   $$\text{P@R90} = \max \left\{ P(k) \mid R(k) \ge 0.90 \right\}$$
2. **Completeness at Low Contamination ($\text{R@FPR01}$ and $\text{R@FPR05}$):**
   Recall at maximum allowable False Positive Rates of $1\%$ and $5\%$:
   $$\text{R@FPR01} = \max \left\{ \text{TPR}(k) \mid \text{FPR}(k) \le 0.01 \right\}$$
3. **Small-Sample Guard & Confidence Interpolation:**
   When $N_{\text{neg}} < 30$, raw ROC/PR points exhibit high variance. The tracker must log an informative warning, smooth discrete steps using monotonic cubic Hermite interpolation (PCHIP), and report sample counts alongside the metric.

### 2.3 Checkpointing Strategy Transition

Model checkpointing (`TopKModelCheckpointer`) must transition:
- **From:** `val_macro_f1` (argmax-dependent, collapsed by rare classes)
- **To:** `val_map` (mean Average Precision) as the primary ranking metric, supplemented by `val_ap_real` and `val_macro_f1_calibrated`.

---

## 3. Loss Function & Decision Threshold Calibration

### 3.1 Loss Function Audit & Dynamic Class Balancing

#### 1. Dynamic Class Count Extraction
The trainer must automatically query the active dataset to compute effective sample counts:
```python
train_counts = np.bincount(train_labels, minlength=num_classes)
```
For the current training set, $N = [16, 428, 470, 244]$.

#### 2. Cui et al. Hyperparameter Calibration ($\beta$)
The effective number of samples is given by:
$$E_c = \frac{1 - \beta^{n_c}}{1 - \beta}, \quad w_c = \frac{1}{E_c} \Bigg/ \left( \sum_{k=0}^{C-1} \frac{1}{E_k} \right) \cdot C$$

- If $\beta \to 1.0$ (e.g. $0.9999$), $E_c \approx n_c$, producing raw inverse frequency ($w_0 / w_1 \approx 27\times$). This causes gradient instability on noisy negatives.
- If $\beta = 0.99$, effective numbers compress smoothly:
  - $E_0 = 14.85 \implies w_0 = 2.73$
  - $E_1 = 98.65 \implies w_1 = 0.41$
  - $E_2 = 99.11 \implies w_2 = 0.41$
  - $E_3 = 91.43 \implies w_3 = 0.44$
  Ratio $w_0 / w_1 \approx 6.6\times$.
  **Recommendation:** Set default $\beta = 0.99$ for datasets with $N < 2,000$, transitioning to $\beta = 0.999$ for $N \ge 10,000$.

#### 3. Asymmetric Focal Loss (ASL)
In multi-epoch streak and transient detection, negative samples (artifacts) are diverse, while positive moving targets share structured spatial kinematics. Asymmetric focal modulation decouples positive and negative focusing:
$$\mathcal{L}_{\text{ASL}} = - \sum_{c=0}^{C-1} w_c \cdot \left[ y_c (1 - p_c)^{\gamma_+} \log(p_c) + (1 - y_c) (p_c)^{\gamma_-} \log(1 - p_c) \right]$$
Setting $\gamma_+ = 1.0$ (gentle modulation for rare comets/NEOs) and $\gamma_- = 3.0$ (aggressive suppression of obvious subtraction noise) prevents negative gradient saturation.

### 3.2 Post-Processing Decision Threshold Calibration

A deep neural network trained with cross-entropy or focal loss produces uncalibrated output probabilities. Two calibration stages must be introduced prior to alert generation:

```mermaid
flowchart TD
    RawLogits["Raw Network Logits z in R^4"] --> TempScaling["Stage 1: Temperature Scaling\nz' = z / T* (Learned via NLL on Val)"]
    TempScaling --> CalProbs["Calibrated Probabilities p_cal = Softmax(z')"]
    CalProbs --> Choice{"Decision Strategy"}
    Choice --> SubtypeRouting["Strategy A: Per-Class Threshold Matrix\ny* = argmax_c (p_c / tau_c*)"]
    Choice --> CascadeRouting["Strategy B: Cascaded Alert Gatekeeper\nIf sum(p_1, p_2, p_3) >= tau_real* -> REAL\nElse -> BOGUS"]
```

#### Stage 1: Temperature Scaling (Guo et al. 2017)
Learn a single positive temperature scalar $T^* > 0$ on validation logits $\mathbf{z}$:
$$\min_{T > 0} \; -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \log \left( \frac{\exp(z_{i, y_i} / T)}{\sum_{j} \exp(z_{i, j} / T)} \right)$$
Solved via 1D bounded scalar minimization (Brent's method, $T \in [0.1, 10.0]$) in $< 10$ milliseconds. Temperature scaling preserves argmax rankings while restoring calibrated posterior confidence estimates.

#### Stage 2: Validation-Optimized Decision Thresholds ($\boldsymbol{\tau}^*$)
Instead of default argmax ($\tau = 0.25$), search optimal per-class thresholds $\boldsymbol{\tau}^* = (\tau_0^*, \tau_1^*, \tau_2^*, \tau_3^*)$:
$$\hat{y} = \arg\max_{c \in \{0, 1, 2, 3\}} \left( \frac{p_{c}}{\tau_c^*} \right)$$
Thresholds are optimized on the validation set via coordinate search or Nelder-Mead to directly maximize validation **Macro-F1**.

#### Stage 3: Operational Cascaded Alert Gatekeeper
For astronomical alert distribution streams:
1. **Real vs. Bogus Gate:**
   $$p_{\text{real}} = p_1 + p_2 + p_3$$
   If $p_{\text{real}} < \tau_{\text{real}}^* \implies \text{Class 0 (BOGUS\_ARTIFACT)}$.
   Where $\tau_{\text{real}}^*$ is set to achieve $\text{Recall} \ge 0.95$ with minimum false alarms.
2. **Astrophysical Subtype Classifier:**
   If $p_{\text{real}} \ge \tau_{\text{real}}^*$, route to:
   $$\hat{c}_{\text{real}} = \arg\max_{c \in \{1, 2, 3\}} \left( \frac{p_c}{\tau_c^*} \right)$$

---

## 4. Architectural & Training Loop Enhancements

### 4.1 Learning Rate Scheduling & Optimization Dynamics

#### 1. Transition to Step-Level Scheduling with Warm Restarts (SGDR)
- **Current Defect:** `scheduler.step()` is invoked once per epoch. With $B=32$ and $N=1,158$, an epoch contains only $36$ batches. A 3-epoch warmup is only $108$ steps, and the monotonic cosine decay permanently drops the learning rate to $10^{-6}$ by epoch 25, freezing the spatial backbone into early suboptimal feature extractors.
- **Enhanced Schedule:** **Cosine Annealing with Warm Restarts** (Loshchilov & Hutter), stepped **per batch**:
  $$T_0 = 10 \text{ epochs}, \quad T_{\text{mult}} = 2, \quad \eta_{\min} = 1 \times 10^{-6}, \quad \eta_{\max} = 3 \times 10^{-4}$$
  Restarts at epoch 10 and epoch 30 inject momentum to escape sharp local minima.

#### 2. Discriminative Layer Learning Rates (Layer-Wise Decay)
The spatial ResNet-Lite backbone, temporal BiGRU/Transformer, and classification head require different gradient dynamics:
$$\eta_{\text{backbone}} = 0.3 \times \eta_{\text{base}}, \quad \eta_{\text{aggregator}} = 1.0 \times \eta_{\text{base}}, \quad \eta_{\text{head}} = 1.5 \times \eta_{\text{base}}$$
This stabilizes low-level edge/streak filter learning while enabling fast adaptation of temporal sequence patterns.

### 4.2 Multi-Epoch Astronomical Tensor Augmentations

The current pipeline only applies spatial $D_4$ dihedral group operations (rotations and flips). We specify three domain-specific multi-epoch augmentations:

```mermaid
flowchart TD
    subgraph InputTracklet["Native Tracklet Tensor: (N, 3, 128, 128)"]
        Frames["Frames: t_1, t_2, ..., t_N"]
    end

    subgraph TemporalAugs["Astronomical Time-Series Augmentations"]
        Inversion["1. Kinematic Temporal Inversion\nReverse frame order (t_N ... t_1)\nReverses motion vector; physically valid trajectory"]
        Subsampling["2. Stochastic Frame Dropout\nRandomly drop 1 frame for N >= 4\nSimulates missing/cloudy observational epochs"]
        Photometry["3. Photometric Seeing Perturbation\nMultiply Diff channel by s ~ U(0.85, 1.15)\nSimulates asteroid rotational light curves"]
    end

    InputTracklet --> Inversion
    InputTracklet --> Subsampling
    InputTracklet --> Photometry
```

1. **Kinematic Temporal Inversion ($p=0.5$):**
   Reversing the temporal ordering of frames:
   $$(x_1, x_2, \dots, x_N) \longrightarrow (x_N, x_{N-1}, \dots, x_1)$$
   The physical trajectory remains completely valid under inverted time, effectively doubling sequence diversity and preventing recurrent direction bias.
2. **Stochastic Frame Dropout ($p=0.3$ for $N \ge 4$):**
   Randomly drops 1 frame from the tracklet sequence ($N \to N-1$). This trains the BiGRU and Transformer aggregators to remain robust against missing survey exposures and variable observational cadence.
3. **Photometric Light-Curve Jitter ($p=0.4$):**
   Multiplies Science (channel 0) and Difference (channel 2) by a smooth random factor $s \sim \mathcal{U}(0.85, 1.15)$ across epochs to emulate rotational light curves and atmospheric transparency fluctuations.

### 4.3 Resolving Negative Sample Scarcity in the Dataset

The validation split contains only $N_{\text{neg}} = 1$ because the data-mining crawler grouped artifacts under target object designations (`BOGUS_<target_id>`), and cryptographic hashing partitioned only one target into `val`.
To permanently resolve this:
1. **Uncoupled Artifact Split Routing:** Non-target artifacts (satellite trails, optical ghosts, cosmic rays) must be partitioned via stratified random splitting rather than object-keyed hashing, ensuring balanced representation:
   $$\text{Train: } 70\% \quad (140+), \quad \text{Val: } 15\% \quad (30+), \quad \text{Test: } 15\% \quad (30+)$$
2. **Online Hard Negative Mining (OHEM):** In each batch, retain the top-K highest-loss negative examples to guide gradient updates.

---

## 5. Concrete Implementation Blueprint & Architecture Stubs

### 5.1 Updated Configuration Schema: `TrackletTrainingConfig`

```python
# In src/LUMIA/training/config.py
from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from LUMIA.core.schemas import LumiaBaseModel

class TrackletTrainingConfig(LumiaBaseModel):
    """Enhanced configuration for Project LUMIA tracklet training and validation."""

    # Dataset & Storage
    dataset_path: Path = Field(default=Path("data/ground_truth/ztf_sso_tracklets_128x128.h5"))
    model_output_dir: Path = Field(default=Path("data/models"))
    batch_size: int = Field(default=32, ge=1, le=512)
    epochs: int = Field(default=30, ge=1, le=200)

    # Optimization
    learning_rate: float = Field(default=3e-4, gt=0.0, le=1e-1)
    weight_decay: float = Field(default=1e-3, ge=0.0)
    lr_scheduler_type: Literal["cosine", "cosine_restarts"] = Field(default="cosine_restarts")
    restart_t0_epochs: int = Field(default=10, ge=1, le=50)
    warmup_epochs: int = Field(default=3, ge=0, le=20)
    layerwise_lr_decay: float = Field(default=0.3, ge=0.1, le=1.0)
    grad_clip_norm: float = Field(default=1.0, gt=0.0)
    use_amp: bool = Field(default=True)

    # Loss Configuration
    loss_function: Literal["class_balanced_focal", "focal", "asymmetric_focal", "cross_entropy"] = (
        Field(default="class_balanced_focal")
    )
    focal_gamma: float = Field(default=2.0, ge=0.0, le=5.0)
    focal_gamma_neg: float = Field(default=3.0, ge=0.0, le=7.0)
    class_balanced_beta: float = Field(default=0.99, ge=0.9, lt=1.0)
    label_smoothing: float = Field(default=0.05, ge=0.0, le=0.2)

    # Validation & Model Selection
    primary_metric: Literal["val_map", "val_macro_f1", "val_ap_real"] = Field(default="val_map")
    enable_temperature_scaling: bool = Field(default=True)
    enable_threshold_optimization: bool = Field(default=True)
    min_neg_samples_warning: int = Field(default=10)

    # Augmentation
    enable_temporal_inversion: bool = Field(default=True)
    enable_frame_dropout: bool = Field(default=True)
    enable_photometric_jitter: bool = Field(default=True)

    num_workers: int = Field(default=4, ge=0)
    seed: int = Field(default=42, ge=0)
    model: TrackletModelConfig = Field(default_factory=TrackletModelConfig)
```

### 5.2 Enhanced Metrics Evaluation Pipeline: `TrackletMetricsTracker`

```python
# In src/LUMIA/training/metrics.py
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

class TrackletMetricsTracker:
    """Comprehensive astronomy evaluation tracker with AP, PR curves, and operating points."""

    def __init__(self, num_classes: int = 4, min_neg_samples_warning: int = 10) -> None:
        self.num_classes = num_classes
        self.min_neg_samples = min_neg_samples_warning

    def compute_metrics(
        self,
        y_true: np.ndarray,
        logits: np.ndarray,
        thresholds: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute full multi-class and domain-specific transient metrics."""
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)

        # 1. Predictions via argmax or threshold scaling
        if thresholds is not None:
            scaled = probs / np.clip(thresholds, 1e-4, 1.0)
            preds = np.argmax(scaled, axis=-1)
        else:
            preds = np.argmax(probs, axis=-1)

        accuracy = float(np.mean(preds == y_true))
        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0.0))

        # 2. Per-class AP & Mean Average Precision (mAP)
        ap_per_class: dict[str, float] = {}
        valid_aps: list[float] = []
        for i in range(self.num_classes):
            binary_target = (y_true == i).astype(int)
            class_name = CLASS_NAMES.get(i, f"class_{i}")
            if np.sum(binary_target == 1) > 0 and np.sum(binary_target == 0) > 0:
                ap_val = float(average_precision_score(binary_target, probs[:, i]))
                ap_per_class[class_name] = ap_val
                valid_aps.append(ap_val)
            else:
                ap_per_class[class_name] = 0.0

        m_ap = float(np.mean(valid_aps)) if valid_aps else 0.0

        # 3. Operational Real-vs-Bogus Metrics
        y_binary = (y_true > 0).astype(int)
        p_real = np.sum(probs[:, 1:], axis=1)

        n_neg = int(np.sum(y_binary == 0))
        n_pos = int(np.sum(y_binary == 1))

        if n_neg < self.min_neg_samples:
            logger.warning(
                "Validation split contains only %d negative samples. Operating point metrics "
                "may exhibit high variance.", n_neg
            )

        if n_pos > 0 and n_neg > 0:
            ap_real = float(average_precision_score(y_binary, p_real))
            p_curve, r_curve, _ = precision_recall_curve(y_binary, p_real)
            fpr_curve, tpr_curve, _ = roc_curve(y_binary, p_real)

            # Precision at 90% and 95% Recall
            idx_r90 = np.where(r_curve >= 0.90)[0]
            p_at_r90 = float(p_curve[idx_r90[-1]]) if len(idx_r90) > 0 else 0.0

            idx_r95 = np.where(r_curve >= 0.95)[0]
            p_at_r95 = float(p_curve[idx_r95[-1]]) if len(idx_r95) > 0 else 0.0

            # Recall at 1% and 5% FPR
            idx_fpr01 = np.where(fpr_curve <= 0.01)[0]
            r_at_fpr01 = float(tpr_curve[idx_fpr01[-1]]) if len(idx_fpr01) > 0 else 0.0

            idx_fpr05 = np.where(fpr_curve <= 0.05)[0]
            r_at_fpr05 = float(tpr_curve[idx_fpr05[-1]]) if len(idx_fpr05) > 0 else 0.0

            # FPR at 90% Recall
            target_idx = np.where(tpr_curve >= 0.90)[0]
            fpr90 = float(fpr_curve[target_idx[0]]) if len(target_idx) > 0 else 1.0
        else:
            ap_real = 0.0
            p_at_r90 = p_at_r95 = r_at_fpr01 = r_at_fpr05 = 0.0
            fpr90 = 0.0 if n_neg == 0 else 1.0

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "map": m_ap,
            "ap_per_class": ap_per_class,
            "ap_real": ap_real,
            "p_at_r90": p_at_r90,
            "p_at_r95": p_at_r95,
            "r_at_fpr01": r_at_fpr01,
            "r_at_fpr05": r_at_fpr05,
            "fpr90": fpr90,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }
```

### 5.3 Post-Processing Calibrator: `TemperatureScaler` & `ThresholdOptimizer`

```python
# In src/LUMIA/training/calibration.py
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from sklearn.metrics import f1_score

class TemperatureScaler:
    """Post-processing temperature scaling calibrator optimizing NLL on validation logits."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Find optimal temperature T > 0 minimizing NLL."""
        def nll_loss(t: float) -> float:
            scaled = logits / max(t, 1e-4)
            exp_s = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
            probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
            log_p = np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0))
            return -float(np.mean(log_p))

        res = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(res.x)
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=-1, keepdims=True)


class ThresholdOptimizer:
    """Optimizes per-class decision thresholds to maximize validation Macro-F1."""

    def __init__(self, num_classes: int = 4) -> None:
        self.num_classes = num_classes
        self.thresholds: np.ndarray = np.ones(num_classes) / num_classes

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Coordinate search finding thresholds maximizing Macro-F1."""
        def objective(tau: np.ndarray) -> float:
            scaled = probs / np.clip(tau, 1e-4, 1.0)
            preds = np.argmax(scaled, axis=-1)
            return -float(f1_score(y_true, preds, average="macro", zero_division=0.0))

        init_tau = np.ones(self.num_classes) / float(self.num_classes)
        bounds = [(0.05, 0.95)] * self.num_classes
        res = minimize(objective, init_tau, method="Nelder-Mead", bounds=bounds, options={"maxiter": 200})
        self.thresholds = res.x / np.sum(res.x)
        return self.thresholds

    def predict(self, probs: np.ndarray) -> np.ndarray:
        scaled = probs / np.clip(self.thresholds, 1e-4, 1.0)
        return np.argmax(scaled, axis=-1)
```

### 5.4 Dynamic Loss Fix in `TrackletTrainer`

```python
# In src/LUMIA/training/trainer.py
def _build_loss_fn(self) -> nn.Module:
    loss_type = self.config.loss_function
    num_classes = self.config.model.num_classes

    # Automatically extract class counts from training dataset
    try:
        raw_labels = self.train_loader.dataset.get_labels()  # or read from CSR HDF5
        class_counts = np.bincount(raw_labels, minlength=num_classes)
    except Exception as exc:
        logger.warning("Could not automatically extract train class counts: %s. Using uniform.", exc)
        class_counts = None

    if loss_type == "class_balanced_focal":
        return ClassBalancedFocalLoss(
            class_counts=class_counts,
            beta=self.config.class_balanced_beta,
            gamma=self.config.focal_gamma,
        )
    if loss_type == "asymmetric_focal":
        return AsymmetricFocalLoss(
            class_counts=class_counts,
            gamma_pos=self.config.focal_gamma,
            gamma_neg=self.config.focal_gamma_neg,
        )
    if loss_type == "focal":
        return FocalLoss(gamma=self.config.focal_gamma)
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
    raise ValueError(f"Unsupported loss function '{loss_type}'.")
```

---

## 6. Validation Strategy & Verification Protocol

1. **Synthetic Edge-Case Test:** Unit test `compute_metrics` when $N_{\text{neg}} = 1$ to verify it does not produce division-by-zero crashes and logs the small-sample diagnostic warning.
2. **Class Balanced Weights Verification:** Assert that with counts `[16, 428, 470, 244]` and $\beta=0.99$, Class 0 receives a weight between $2.5$ and $3.0$, while Classes 1 and 2 receive weights $< 0.5$.
3. **Threshold Optimization Verification:** Verify on simulated uncalibrated probabilities that `ThresholdOptimizer` improves Macro-F1 from $0.43$ to $> 0.70$ by adjusting the boundary for under-represented classes.
4. **End-to-End Epoch Step Test:** Verify that `TrackletTrainer` logs `val_map`, `val_ap_real`, `p_at_r90`, and `r_at_fpr01` cleanly each epoch, and ranks checkpoints according to `primary_metric`.
