# LUMIA Models

## Overview
LUMIA Models implements deep neural network topologies for multi-epoch astronomical candidate and tracklet classification. It provides spatial convolutional feature backbones (ResNet-Lite) alongside swappable temporal sequence aggregators (Masked Pooling, Bidirectional GRU, and Transformer encoders) to classify moving-object trajectories across varying epoch lengths.

## Interface
- `BasicResidualBlock`: Standard 2-layer residual convolutional block with BatchNorm, ReLU, and 1x1 projection shortcut.
- `ResNetLite`: Compact 3-stage convolutional backbone mapping 3-channel cutout triplets to 128-dimensional spatial feature vectors.
- `MaskedPoolAggregator`: Zero-parameter baseline aggregator executing valid-frame masked mean and max pooling along the temporal axis.
- `BiGRUAggregator`: Multi-layer bidirectional GRU processing variable-length sequences using PyTorch packed sequence utilities.
- `TransformerAggregator`: Multi-head self-attention encoder utilizing sinusoidal temporal positional embeddings and attention masks.
- `TrackletClassifierNet`: End-to-end neural network combining `ResNetLite` spatial feature extraction with a selected temporal aggregator and an MLP classification head.
- `build_tracklet_model`: Factory function constructing a `TrackletClassifierNet` from a `TrackletModelConfig`.

## Invariants
- Inputs to temporal aggregators require a token tensor `(B, N_max, D_spatial)` and a corresponding boolean `padding_mask` `(B, N_max)` where `True` denotes valid observation epochs.
- Network outputs are raw classification logits of shape `(B, num_classes)` without implicit softmax activation.
- Default category dimensionality is frozen at 4 classes (`0: Bogus`, `1: Point Source`, `2: Streak`, `3: Comet Coma`).

## Side Effects
- Allocates CPU and GPU tensor memory during forward and backward passes.
- Initializes model weights using PyTorch default initializers based on global RNG state.

## Verification
- Run: `uv run pytest tests/test_training/test_tracklet_models.py`
