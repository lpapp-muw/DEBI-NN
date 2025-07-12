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

1. **Access data and configure paths:** access data from
   
   DEBI-NN/
   
   ├── Datasets/

   and configure data source paths

3. **Configure main.py**
    - data_dir: list of dataset paths
    - experiment_id: matching list of IDs
    - layouts & hidden_neuron_count: network architectures to test
    - Set regularization flags and hyperparameters in calls to run_comparison / run_all_configurations

4. **Run experiments:** for a single configuration or all configurations defined

## Outputs

If no output directories are scpecified, a `results` directory will be created and all results will be saved inside such as:

- `results/{exp_id}_{layout}_results.csv` — metric summaries  
- `results/{exp_id}/trained_models/*.pt` — best model weights
