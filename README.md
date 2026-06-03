# HCAT-Pain: Hierarchical Cross-Attention Transformer for Non-contact Multimodal Pain Classification

<p align="center">
  <img src="assets/Proposed_Method.gif" alt="Overview of the proposed HCAT-Pain framework" width="80%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/FG-2026-315A89" alt="FG 2026">
  <img src="https://img.shields.io/badge/task-pain%20classification-2A9D8F" alt="Pain Classification">
  <img src="https://img.shields.io/badge/modality-facial%20video-E76F51" alt="Facial Video">
  <img src="https://img.shields.io/badge/framework-multimodal%20transformer-8E7AB5" alt="Multimodal Transformer">
  <img src="https://img.shields.io/badge/code-coming%20soon-D9A441" alt="Code Coming Soon">
</p>

This is the official repository for our FG 2026 paper:

> **Hierarchical Cross-Attention Transformer for Non-contact Multimodal Pain Classification using Remote Physiological Signals and Visual Features**  
> Anup Kumar Gupta, Puneet Gupta, and Abhinav Dhall  
> IEEE International Conference on Automatic Face and Gesture Recognition, 2026

---

## Overview

Automatic pain assessment is important for healthcare, but conventional physiological monitoring often depends on contact-based sensors. Such sensors may be uncomfortable or impractical in several settings, including neonatal care, geriatric care, burn care, and continuous monitoring.

**HCAT-Pain** explores a fully non-contact alternative for pain classification from facial videos. The framework jointly models:

- remote pulse signals extracted using an Eulerian rPPG pipeline,
- remote respiratory signals extracted using Lagrangian motion analysis,
- visual facial behavior represented using Action Unit intensities.

These complementary cues are fused using a hierarchical Transformer architecture with modality-specific temporal encoding, symmetric cross-attention, and a fusion Transformer.

---

## Method

<!-- <p align="center">
  <img src="assets/proposed_architecture.png" alt="Abstract overview of the proposed architecture" width="95%">
</p> -->

HCAT-Pain consists of four main stages:

| Stage | Description |
|---|---|
| **Remote physiological extraction** | Extracts pulse and respiratory signals from facial videos without contact sensors. |
| **Visual feature extraction** | Uses Action Unit intensities to capture facial muscle activity related to pain expression. |
| **Modality-specific encoding** | Processes physiological and visual streams using separate Transformer encoders. |
| **Cross-modal fusion** | Uses symmetric cross-attention and a fusion Transformer to combine complementary information. |

A higher-quality method animation is available here: [`assets/Proposed_Method.mp4`](assets/Proposed_Method.mp4)

---

## Key contributions

- We propose a non-contact multimodal framework for pain classification from facial videos.
- We jointly use remotely extracted pulse and respiratory signals, instead of relying only on cardiac cues.
- We introduce a hierarchical Transformer architecture with modality-specific Transformers, symmetric cross-attention, and a fusion Transformer.
- We employ an Ordinal Weighted Cross-Entropy loss to account for both class imbalance and the ordered nature of pain intensities.
- The framework does not require contact-based physiological ground truth during training.
- Experiments show consistent improvements over existing methods in both multiclass and binary pain classification settings.

---

## Results

### Multiclass pain classification

| Method | Modality | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| Prior video + rPPG method | Visual + rPPG | 61.94 | 63.73 | 62.04 | 61.10 |
| **HCAT-Pain** | **Visual + remote pulse + remote respiration** | **64.72** | **65.92** | **65.51** | **65.69** |

### Binary pain classification

| Setting | Task | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| B1 | No Pain vs. Pain | **88.33** | 83.99 | 77.08 | 79.80 |
| B2 | No Pain vs. High Pain | **86.11** | 84.17 | 85.07 | 84.58 |
| B3 | Low Pain vs. High Pain | **73.61** | 74.84 | 73.61 | 73.28 |

### Fusion ablation

| Fusion strategy | Accuracy | F1-score |
|---|---:|---:|
| Early Sum | 48.61 | 48.12 |
| Early Concatenation | 55.83 | 56.37 |
| Modality Transformer without cross-modality loss | 56.11 | 55.91 |
| Modality Transformer with cross-modality loss | 57.34 | 57.36 |
| Cross-Attention Concatenation | 61.94 | 60.94 |
| **HCAT-Pain** | **64.72** | **65.69** |

---

## Dataset

The experiments are conducted on the [AI4Pain 2024 dataset](https://sites.google.com/view/ai4pain/challenge-details). The dataset contains multimodal recordings from 65 participants under experimentally induced pain conditions.

The original dataset includes facial videos and contact-based physiological signals. In this work, we use only the facial video modality and extract remote physiological and visual features from it.

Please follow the original dataset access procedure and EULA. The dataset is not redistributed in this repository.

---

## Repository status

This repository currently serves as the public project page for the paper.

The implementation code is **not uploaded yet**. The planned release includes:

- preprocessing scripts,
- remote pulse extraction pipeline,
- remote respiratory extraction pipeline,
- Action Unit feature processing interface,
- training and evaluation scripts,
- configuration files,
- reproducibility instructions.

The code will be released after cleanup, documentation, and dependency verification.

---

##  Developer setup

For environment creation, library installations and common setup issues, please see:
- [Installation Guide](docs/SETUP.md)

---
## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{gupta2026hcatpain,
  title     = {Hierarchical Cross-Attention Transformer for Non-contact Multimodal Pain Classification using Remote Physiological Signals and Visual Features},
  author    = {Gupta, Anup Kumar and Gupta, Puneet and Dhall, Abhinav},
  booktitle = {Proceedings of the IEEE International Conference on Automatic Face and Gesture Recognition},
  year      = {2026}
}
```
For citation metadata, see also [`CITATION.cff`](CITATION.cff).
