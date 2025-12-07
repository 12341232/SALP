# SAPL: Structure-Aware Adaptive Pseudo-Labeling for Semi-Supervised Partial Label Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper:

**Structure-Aware Adaptive Pseudo-Labeling for Semi-Supervised Partial Label Learning**

> **Abstract:** *Semi-Supervised Partial Label Learning (SS-PLL) deals with data where each labeled instance is associated with an ambiguous candidate label set, while many unlabeled samples remain unused. Although recent deep SS and PL methods have achieved progress, they often suffer from unreliable pseudo-labels and unstable training caused by noisy predictions. Motivated by multi-view consistency learning, we propose a novel framework named Structure-Aware Adaptive Pseudo-Labeling (SAPL) for robust SS-PLL. Specifically, SAPL computes four weak-view predictions from dual students and their EMA teachers, and assigns each sample a unified consistency-based score. An adaptive quantile threshold is then used to select a small number of reliable pseudo-labeled instances, while rejecting inconsistent ones. To further enhance pseudo-label quality, SAPL introduces a Class Transition Graph (CTG) to rectify fused probabilities with dynamic inter-class structural priors. Extensive experiments show that SAPL consistently improves SS-PLL performance, especially under high ambiguity settings.*

## 🚀 Framework

![SAPL Framework](assets/framework.png)
*Fig 1. Overview of SAPL. (a) The framework integrates multi-view predictions from two student models and their EMA teachers. (b) A unified scoring mechanism evaluates consistency, inter-model agreement, and confidence. (c) Selected samples are rectified using a Class Transition Graph (CTG).*

## ✨ Key Features

* **Unified Reliability Scoring:** Evaluates samples based on three metrics: *Consistency* (divergence among views), *Agreement* (consensus on hard labels), and *Confidence* (max probability).
* **Adaptive Sample Selection:** Uses a dynamic quantile threshold ($q^t$) that evolves with training dynamics to filter reliable pseudo-labels.
* **Class Transition Graph (CTG):** Models inter-class prediction transitions across epochs to rectify pseudo-labels using structural priors.
* **Dual-Student + EMA:** Leverages two student networks and their Exponential Moving Average (EMA) teachers to generate robust weak-view predictions.

## 🛠️ Requirements

The code is implemented using PyTorch.

1. Clone this repository:
   ```bash
   git clone [https://github.com/12341232/SALP.git](https://github.com/12341232/SALP.git)
   cd SALP
