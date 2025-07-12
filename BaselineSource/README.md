# Baseline NN Training Pipeline

This script provides a pipeline for training and evaluating fully-connected neural networks serving as baselines for DEBI-NNs with a variety of regularization techniques. It supports Monte Carlo cross-validation (MCCV), hyperparameter tuning, early stopping, and a variety of regularization techniques.

## Features

- **Regularization**
  - L1
  - L2
  - DropConnect (DC)
  - Weight Standardization (WS)
  - Group Normalization (GN)
  - Shift-Scale Optimization (SCO)

- **Training utilities**  
  - Hyperparameter grid search (learning rate, weight decay)  
  - Early stopping  
  - Class-balanced CrossEntropy loss  
  - Automatic CSV logging of metrics (ACC, BACC, SNS, SPC, PPV, NPV)  
  - Model checkpointing

- **Data handling**  
  - One-hot encoding
  - Data cleaning & key-based matching
  - Directory parsing for MCCV folds

## Installation

1. **Clone this repo**
2. **Install dependencies using environment.yml**

## Usage

1. **Prepare your data and configure paths:** ensure every CSV includes a Key column to align features and labels

2. **Configure main.py**
    - data_dir: list of dataset paths
    - experiment_id: matching list of IDs
    - layouts & hidden_neuron_count: network architectures to test
    - Set regularization flags and hyperparameters in calls to run_comparison / run_all_configurations

3. **Run experiments:** for a single configuration or all configurations defined

## Outputs

- `results/{exp_id}_{layout}_results.csv` — metric summaries  
- `results/{exp_id}/trained_models/*.pt` — best model weights