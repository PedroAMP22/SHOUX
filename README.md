# SSNN-XCBR: A Siamese-Spiking Neural Network Architecture for Audio Case-Based Explanations

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of a **Siamese Spiking Neural Network (SSNN)** designed for robust audio retrieval and the generation of faithful factual and counterfactual explanations. Unlike traditional acoustic-based methods (e.g., MFCC), this model learns a **semantic latent space** optimized for class separability, temporal dynamics, and computational efficiency.

## Key Features

- **Explainable-by-Design Backbone:** `PopNetAudio` architecture utilizing **Population Coding** (class-specific expert neurons) to ensure intrinsic transparency.
- **Temporal Siamese Head:** A learned convolutional projection head that emulates biological spike-train metrics (Van Rossum distance) while maintaining Euclidean efficiency.
- **Hierarchical Representation Learning:** A custom **Hierarchical Contrastive Loss** that organizes the latent space into semantic macro-clusters (digits) and acoustic micro-clusters (speakers).
- **Advanced xAI Evaluation Framework:**
    - **Semantic Fidelity:** Uses **Wav2Vec 2.0** as an impartial "Gold Standard" judge to measure Factual and Counterfactual RMSE.
    - **Causal Validation:** Implements the **ROAD (Remove and Debias)** benchmark to verify that the model relies on causal phonetic features rather than background noise.
    - **Latent Topology Metrics:** Quantitative assessment of Inter-class Diversity and Intra-class Compactness.
- **High Efficiency:** Real-time inference and retrieval in **~2ms** (over 800x faster than raw biological metrics).

## Architecture

The system consists of two critical components:

1.  **Backbone (SNN):** Processes raw 1D audio waveforms into binary spike trains using LIF (*Leaky Integrate-and-Fire*) neurons. It features a hidden layer of 250 neurons divided into 10 specialized expert groups.
2.  **Siamese Projection Head:** A high-resolution temporal head that applies learned grouped convolutions ($K=25$) and adaptive pooling to generate a 512-dimensional semantic embedding.

## Benchmark Results (AudioMNIST)

| Method | Time/Query | Recall@5 | F. RMSE (↓) | CF. RMSE (↑) | CF. Div (↑) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Siamese SNN (Ours)** | **2.10 ms** | **93.1%** | 0.143 | 0.181 | **5.10** |
| MFCC (SOTA) | 2.72 ms | 88.2% | 0.134 | 0.177 | 5.03 |
| Van Rossum | 1709 ms | 94.4% | 0.139 | 0.181 | 5.10 |

### Latent Space Topology (t-SNE)
Our Siamese SNN generates clearly segregated semantic "islands," significantly reducing the boundary overlap observed in MFCC. This ensures that counterfactual explanations (nearest enemies) are safe, distinct, and unambiguous.

### Causal Fidelity (ROAD)
The ROAD benchmark confirms the model's robustness: the latent representation remains unchanged when up to 50% of irrelevant noise is removed (LeRF $\approx$ 0), while it collapses immediately when key phonetic segments are occluded (MoRF).


## 📝 Citation
If you use this work in your research, please cite:
@misc{martin2026ssnn,
  title={SSNN-XCBR: A Siamese-Spiking Neural Network Architecture for Audio Case-Based Explanations},
  author={Mart{\'i}n-Pel{\'a}ez, Pedro A. and Caro-Mart{\'i}nez, Marta},
  year={2026},
  howpublished={Technical Report, Project SHOUX, GAIA Group},
  institution={Department of Software Engineering and Artificial Intelligence, Universidad Complutense de Madrid},
  address={Madrid, Spain}
}
---
Developed as part of the research by pedro Antonio Martín Pelaéz under the supervision of marta.
