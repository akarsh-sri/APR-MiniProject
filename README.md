# RNN/LSTM for Sequential Data Classification (APR Mini-Project)

This repository contains the code for a mini-project on Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models for classifying sequential data. The project implements and evaluates these models on two distinct datasets: handwritten characters and speech segments, using PyTorch.

## Project Overview

The objective is to build, tune, and compare the performance of simple RNN and LSTM networks for two different sequence classification tasks:
1.  **Handwritten Character Recognition:** Classifying characters based on a sequence of 2D (x, y) coordinates.
2.  **Consonant-Vowel (CV) Segment Identification:** Classifying speech segments based on a sequence of 39-dimensional MFCC features.

## Features

* **Data Preprocessing:**
    * Custom PyTorch `Dataset` classes for both handwriting and CV datasets.
    * **Handwriting:** Per-sample normalization of (x, y) coordinates to a `[0, 1]` range.
    * **CV (MFCC):** Global mean and standard deviation normalization computed across the entire training set.
* **Models:**
    * A flexible `SequenceClassifier` model supporting `RNN`, `LSTM`, and `GRU` layers.
    * Uses `pack_padded_sequence` to efficiently handle variable-length sequences.
    * The model uses the final hidden state from the last layer for classification.
* **Training:**
    * A training loop with a specific early stopping criterion: convergence is declared when the difference in average training loss between two successive epochs falls below a threshold of `1e-4`.
* **Evaluation:**
    * Experiment helpers to run multiple configurations, save results, and plot:
        * Average Training Loss vs. Epochs
        * Confusion Matrices for test set performance.

## Datasets

The notebook expects a `data.zip` file to be uploaded. This zip file should have the following internal structure:
data.zip

```bash
└── data/
    ├── handwriting/
    │   ├── a.zip
    │   ├── ai.zip
    │   ├── bA.zip
    │   ├── chA.zip
    │   └── dA.zip
    └── cv/
        ├── ba.zip
        ├── bhA.zip
        ├── ka.zip
        ├── ni.zip
        └── tA.zip
```
Each of the inner `.zip` files (e.g., `a.zip`, `ba.zip`) contains the `train/dev` (or `Train/Test`) folders, which in turn hold the `.txt` or `.mfcc` data files.

## How to Run

1.  **Environment:** This project is best run in Google Colab (it includes `from google.colab import files`). It can also be adapted for a local environment.
2.  **Requirements:**
    * `Python 3.x`
    * `PyTorch`
    * `NumPy`
    * `Matplotlib`
3.  **Execution:**
    1.  Open `APR_Miniproject.ipynb` in Google Colab.
    2.  Ensure the runtime is set to **GPU** (recommended).
    3.  Run the cells sequentially.
    4.  You will be prompted to **upload your `data.zip` file** in the "Data Setup" section (Cell 4).
    5.  The final cell will display the resulting loss and confusion matrix plots.

## Results Summary

The following results were obtained from the experiments defined in the notebook.

### Handwriting Dataset

| Model | Hidden Size | Layers | Dropout | Train Accuracy | Test Accuracy | Epochs to Converge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RNN | 64 | 1 | 0.0 | 84.35% | 85.00% | 200 |
| **LSTM** | **64** | **1** | **0.0** | **86.96%** | **88.00%** | **56** |

### CV Segment Dataset

| Model | Hidden Size | Layers | Dropout | Train Accuracy | Test Accuracy | Epochs to Converge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RNN | 128 | 1 | 0.0 | 98.77% | 88.80% | 42 |
| **LSTM** | **128** | **1** | **0.0** | **100.00%** | **92.35%** | **25** |

In both cases, the **LSTM** model achieved higher test accuracy and converged in significantly fewer epochs.
