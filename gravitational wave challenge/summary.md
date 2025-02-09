# Binary Classifier Summary
This document is for summarizing the method of the binary classifier I used.
## Overview
The model is developed for detect the anomalies of gravitational wave signal by using thsee datas:
* Background
* Binary black holes
* Sine Gaussian <br>
Besides provided data, we can also create more datas from 1. Add white noise to original data, 2. Add Sine Gaussian noise to background data
* Background w/ white noise
* Binary black holes w/ white noise
* Synthetic Sine Gaussian into original background
The workflow in a nutshell: <br>
1. Preprocess data:
   - Standardize data: data/ std(data)
   - Flatten data
   - Run Fast Fourier Transform
   - Standardize data again: (data-mean(data))/ std(data)
2. Feed processed data into DNN
## Model Architechure
* Inputs: [N, 200] ; N: batch size
* Layers:
  - Linear layer:
    - input: 200
    - Output: 128
    - Activation: ReLU
  - Dropout(p=0.3)
  - Linear layer:
    - input: 128
    - Output: 128
    - Activation: ReLU
  - Dropout(p=0.3)
  - Linear layer:
    - input: 128
    - Output: 1
    - Activation: ReLU
## Training 
* Loss function: BCEwithlogitloss
* Oprimizer: Adam
* Training process:
  - Use Supervised Learning, background label=1, others=0
  - Train every kinds of datasets for 200 epochs w/ 100000 batch size
  - Curriculum learning schedule: original datasets w/ lr=1e-3, original datasets with noise w/ lr=1e-3, synthetic data with stronger signal ratio w/ lr=1e-4, synthetic data with weaker signal ratio w/ lr=1e-4
  - 
