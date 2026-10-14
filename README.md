# A Comparative Study of Graph Neural Network Layer Selection for Interaction Modelling in Driving Trajectory Prediction

**George Daoud, Mohamed El-Darieby**

*IEEE Intelligent Vehicles Symposium (IV 2026) Workshops*

[![arXiv](https://img.shields.io/badge/arXiv-2606.14956-b31b1b.svg)](https://doi.org/10.48550/arXiv.2606.14956)
[![IEEE](https://img.shields.io/badge/IEEE-10.1109%2FIVWorkshops71735.2026.11624111-00629B.svg)](https://doi.org/10.1109/IVWorkshops71735.2026.11624111)

* arXiv: https://doi.org/10.48550/arXiv.2606.14956
* IEEE: https://doi.org/10.1109/IVWorkshops71735.2026.11624111

## Overview

GNN-based trajectory predictors use many different graph layers, and there is little guidance on which layers best capture **spatial interactions** between road agents and the **temporal dynamics** of each agent. This paper compares **19 graph convolution layer (GCL) types** as the spatial and temporal layers of a spatio-temporal GNN trajectory predictor. It tests 55 spatial/temporal combinations and identifies the strongest ones.

## Framework

- **Graph construction**
  - *Spatial edges* connect agents in the same frame that are within 30 m of each other.
  - *Temporal edges* connect consecutive observations of the same agent.
- **Node features**: agent dimensions, heading and type, combined with a 200-d map embedding from a ResNet-18 map encoder. The map is centred but not rotated.
- **Message passing**: paired spatial and temporal GCLs, with 100-d embeddings, 4 GCL layers and skip connections to reduce over-smoothing.
- **Output**: an MLP head predicts a single (unimodal) 25-step trajectory per agent.
- **Horizon**: 1 s of observation (5 frames at 5 Hz) to predict 5 s into the future.

## Dataset

The experiments use the [rounD](https://levelxdata.com/round-dataset/) dataset, drone recordings of German roundabouts. Only the third recording scenario is used, downsampled from 25 FPS to 5 FPS.

| Property | Value |
|---|---|
| Frames / nodes | 5.4K / 54.4K |
| Spatial / temporal edges | 203.9K / 53.7K |
| Valid (predictable) nodes | 34K (62.5%) |
| Split | 70% train / 20% val / 10% test |

## Layers Compared

| Category | Layers |
|---|---|
| Traditional convolutions | GCN (baseline), GraphSAGE, HoGraph |
| Single-head attention | AGNN, FA, GAT, LEConv |
| Multi-head attention | EGC, Graph Transformer, SuperGAT |
| Topology-based | SGC, S2GC, MixHop, TAGCN, MF (Molecular Fingerprints) |
| Recurrent | Gated GCN (GRU), Residual Gated GCN (ResGRU) |
| Specialized filters | ARMA, Chebyshev |

## Training

- Adam optimizer for 60 epochs with MSE loss.
- Learning rate starts at 1e-3 (1e-4 for EGC). It is divided by 10 after epoch 30 and again 20 epochs later.
- The checkpoint with the best validation ADE is kept.
- Hardware: one RTX 4090 GPU, about 10 hours per model.

## Results

### Top five spatial/temporal combinations

| Spatial GCL | Temporal GCL | ADE @3s | ADE @5s | FDE @3s | FDE @5s |
|---|---|---|---|---|---|
| **Chebyshev** | **Chebyshev** | **0.65** | **1.10** | **1.19** | **2.45** |
| MF | GCN | 0.67 | 1.15 | 1.25 | 2.54 |
| TAGCN | TAGCN | 0.66 | 1.19 | 1.25 | 2.80 |
| ARMA | ARMA | 0.67 | 1.19 | 1.28 | 2.74 |
| Chebyshev | GCN | 0.72 | 1.20 | 1.30 | 2.63 |
| *GCN (baseline)* | *GCN* | *1.08* | *1.89* | *2.06* | *4.14* |

For comparison, a CNN+LSTM baseline reports ADE 3.57 and FDE 10.29 at 5 s.

### Key findings

- **Chebyshev, ARMA and topology-aware layers (TAGCN, MF)** perform best, with the clearest gains at longer horizons.
- **Sum aggregation outperforms mean aggregation** in the traditional and topology-based layers.
- **Single-head attention** works better when embeddings are linearly transformed before attention (GAT, LEConv).
- **Multi-head attention** benefits from a distinct transformation per head (Transformer, SuperGAT). EGC's basis decomposition does not.
- **Hop-specific weighting** (TAGCN, MixHop) beats shared weights across hops (SGC, S2GC).
- **Recurrent layers** become competitive when combined with attention (ResGRU).
- **The spatial layer matters more than the temporal one**, and gains from improving each are not additive.

### Limitations

- Only 55 of the possible layer combinations were evaluated.
- Only roundabout scenarios were tested.
- The model predicts a single trajectory, so comparisons with multimodal methods (minADE/minFDE) are not like-for-like.
- Computational cost was not reported.

## Citation

```bibtex
@inproceedings{daoud2026gnnlayer,
  title     = {A Comparative Study of Graph Neural Network Layer Selection for Interaction Modelling in Driving Trajectory Prediction},
  author    = {Daoud, George and El-Darieby, Mohamed},
  booktitle = {2026 IEEE Intelligent Vehicles Symposium Workshops (IV Workshops)},
  year      = {2026},
  doi       = {10.1109/IVWorkshops71735.2026.11624111},
  eprint    = {2606.14956},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

This project is licensed under the GNU GPL v3.0. See [LICENSE](LICENSE).
