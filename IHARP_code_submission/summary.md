# UGosaVGosaSLA VAE Model Summary

This document summarizes the architecture and functionality of the UGosaVGosaSLA VAE Model, which is designed for multi-label classification on oceanographic data.

## Overview

The model is developed to predict anomalies for 12 geographic regions using two primary data sources:
- **Ocean Currents:** Previous day’s **ugosa** and **vgosa** data, each of spatial size 100×160.
- **Sea Level Anomaly (SLA):** Current day’s SLA data with spatial dimensions 100×160.

The overall workflow consists of:
1. Extracting an embedding from the ugosa/vgosa data using a Convolutional Neural Network (CNN).
2. Concatenating this embedding with a flattened SLA vector.
3. Feeding the combined vector into 12 separate VAE-based classifier branches—one per label.

## Model Architecture

### 1. CNN Feature Extractor: `UgVgCNN`

- **Input:** Tensor of shape `[B, 2, 100, 160]`  
  *B* = batch size; 2 channels correspond to ugosa and vgosa.
  
- **Layers:**
  - **Convolution 1:** 
    - Input Channels: 2  
    - Output Channels: 8  
    - Kernel: 3×3 (with padding=1)  
    - Activation: ReLU
  - **Convolution 2:** 
    - Input Channels: 8  
    - Output Channels: 8  
    - Kernel: 3×3 (with padding=1)  
    - Activation: ReLU
  - **Adaptive Average Pooling:**  
    - Reduces the spatial dimensions to a fixed size (e.g., 5×5)
  - **Flattening and Linear Layer:**  
    - The pooled output is flattened and passed through a fully connected layer to produce an embedding of a configurable size (`out_dim`).

### 2. Single VAE Classifier Branch: `SingleVAEClassifier`

Each branch is a mini variational autoencoder (VAE) that processes the combined features to output one logit for classification. Its components include:

- **Encoder:**
  - A linear layer maps the input vector (the concatenated CNN embedding and flattened SLA) to a hidden dimension.
  - Two parallel linear layers generate:
    - **mu:** The mean of the latent distribution.
    - **logvar:** The log-variance of the latent distribution.
  
- **Reparameterization:**  
  - Uses the “reparameterization trick” to sample a latent vector \( z \) from the distribution \( \mathcal{N}(\mu, \exp(0.5 \times \text{logvar})) \).

- **Decoder:**
  - A linear layer maps the latent vector \( z \) to a hidden dimension.
  - A final linear layer produces a single logit (raw score).

### 3. Multi-Label VAE Classifier: `MultiVAEClassifier`

- **Structure:**  
  - Contains 12 instances of `SingleVAEClassifier` (one per target label).
  
- **Operation:**  
  - The same combined input vector is passed through each of the 12 VAE branches.
  - Each branch outputs a single logit.
  - The outputs are concatenated to form a tensor of shape `[B, 12]`.

### 4. Overall Model: `UGosaVGosaSLA_VAE_Model`

- **Inputs:**
  - **ugvg:** Previous day’s ugosa/vgosa data with shape `[B, 2, 100, 160]`.
  - **sla:** Current day’s SLA data with shape `[B, 100, 160]`.
  
- **Processing Steps:**
  1. **CNN Embedding:**  
     - The `ugvg` input is processed by the CNN to yield an embedding.
  2. **Concatenation:**  
     - The SLA data is flattened and concatenated with the CNN embedding.
  3. **VAE Branches:**  
     - The resulting vector is fed into the 12 VAE branches.
  
- **Outputs:**
  - A tensor of logits with shape `[B, 12]` (one logit per label).
  - Lists of latent parameters (`mu` and `logvar`) for each branch (used to compute the KL divergence).

## Training and Loss

- **Loss Function:**
  - **Classification Loss:**  
    - Based on a “soft” F1 score computed from the sigmoid-activated logits.
  - **KL Divergence Loss:**  
    - Computed from the latent parameters (mu and logvar) of each VAE branch, scaled by a hyperparameter (`vae_beta`).

- **Optimization:**  
  - The model is optimized using the Adam optimizer.
  
- **Training Strategy:**  
  - Uses a sliding-window approach over time-series data files.
  - Constructs day pairs where the model uses day *t-1*’s ugosa/vgosa and day *t*’s SLA, along with corresponding labels.

## Data Handling

- **Data Sources:**
  - **NetCDF Files:**  
    - Provide the ugosa, vgosa, and SLA data.
  - **CSV Files:**  
    - Contain anomaly labels for 12 regions.
  
- **Preprocessing:**
  - Files are parsed to extract relevant variables.
  - Data is paired by days to align previous day’s ocean current data with current day’s SLA.
  
- **Prediction:**
  - After training, the model predicts binary outcomes (via a threshold on sigmoid outputs) for a given date range.
  - Predictions and evaluation metrics (macro F1 score) are saved and reported.

## Summary

The UGosaVGosaSLA VAE model combines convolutional feature extraction with multi-branch variational autoencoder classification to tackle a complex spatiotemporal prediction task in oceanography. Its modular design allows for:
- **Effective feature extraction** from spatial data.
- **Uncertainty modeling** via VAE branches.
- **Multi-label predictions** for anomaly detection across different regions.

This architecture is well suited for environmental monitoring applications where both spatial features and latent uncertainty are important.

