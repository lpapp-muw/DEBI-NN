import os
from os.path import join
import csv
from datetime import timedelta
import itertools
from time import perf_counter
import numpy as np
import pandas as pd
from natsort import natsorted
import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def onehot_encoder(feature):
    """
    Encode categorical features as a one-hot numeric array

    Args:
        feature (array-like): Input categorical data to encode

    Returns:
        np.ndarray: One-hot encoded feature array
    """
    ohe = OneHotEncoder(sparse_output=False)
    feature_vector = ohe.fit_transform(feature)

    return feature_vector

def listdir_nohidden(path):
    """
    List files in directory ignoring hidden files (starting with a dot)

    Args:
        path (str): Path to the directory
    """
    for file in natsorted(os.listdir(path)):
        if not file.startswith('.'):
            yield file

def load_data_as_dict(data_dir):
    """
    Loads Monte Carlo cross-validation (MCCV) data into a data dictionary

    Args:
        data_dir (str): Path to directory containing all MCCV folds

    Returns:
        dictionary: Data dictionary containing the fold path, fold number,
                    training dataset path (trainF), training label dataset
                    path (trainL), validation dataset path (validateF) and
                    validation label dataset path (validateL), test dataset
                    path (testF) and test label dataset path (testL).
    """
    
    data_dict = {'fold_path': [], 'fold': [], 'trainF': [], 'trainL': [], 'testF': [],
                 'testL': [], 'validateF': [], 'validateL': []}

    for fold in listdir_nohidden(data_dir):
        data_dict['fold'].append(fold)
        data_dict['fold_path'].append(join(data_dir, fold))

        for dataset in listdir_nohidden(join(data_dir, fold)):
            if dataset == "TrainF.csv":
                data_dict['trainF'].append(
                    join(data_dir, fold, dataset))
            if dataset == "TrainL.csv":
                data_dict['trainL'].append(
                    join(data_dir, fold, dataset))
            if dataset == "TestF.csv":
                data_dict['testF'].append(
                    join(data_dir, fold, dataset))
            if dataset == "TestL.csv":
                data_dict['testL'].append(
                    join(data_dir, fold, dataset))
            if dataset == "ValidateF.csv":
                data_dict['validateF'].append(
                    join(data_dir, fold, dataset))
            if dataset == "ValidateL.csv":
                data_dict['validateL'].append(
                    join(data_dir, fold, dataset))

    return data_dict

class ShiftScaleLayer(nn.Module):
    """
    Custom shift-scale layer that applies learnable affine transformation to groups of neurons
    """
    
    def __init__(self, group_count):
        super(ShiftScaleLayer, self).__init__()
        self.group_count = group_count
        self.gamma = nn.Parameter(torch.ones(group_count, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(group_count, 1), requires_grad=True)

    def forward(self, x):
        batch_size, num_neurons = x.size()
        group_size = num_neurons // self.group_count
        x_reshaped = x[:, :group_size * self.group_count].view(batch_size, self.group_count, group_size)

        #print(f"Scales (gamma): {self.gamma}, shifts (beta): {self.beta}")
        
        # Apply scaling (gamma) and shifting (beta)
        x_scaled_shifted = self.gamma * x_reshaped + self.beta

        # If there are remaining neurons, handle them separately
        if num_neurons % self.group_count != 0:
            remainder_neurons = num_neurons - (group_size * self.group_count)
            remainder_x = x[:, -remainder_neurons:].view(batch_size, 1, remainder_neurons)
            remainder_scaled_shifted = self.gamma[-1:] * remainder_x + self.beta[-1:]
            x_scaled_shifted = torch.cat((x_scaled_shifted.view(batch_size, -1), remainder_scaled_shifted.view(batch_size, -1)), dim=1)
        else:
            x_scaled_shifted = x_scaled_shifted.view(batch_size, -1)

        return x_scaled_shifted

class CustomGroupNorm(nn.Module):
    """
    Implements custom group normalization allowing flexible group assignments
    """
    
    def __init__(self, num_neurons, group_count):
        super(CustomGroupNorm, self).__init__()
        self.group_count = group_count
        self.weight = nn.Parameter(torch.ones(num_neurons), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_neurons), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x):
        batch_size, num_neurons = x.size()
        group_size = num_neurons // self.group_count

        x_reshaped = x[:, :group_size * self.group_count].view(batch_size, self.group_count, group_size)

        mean = x_reshaped.mean(dim=2, keepdim=True)
        variance = x_reshaped.var(dim=2, keepdim=True, unbiased=False)

        x_normalized = (x_reshaped - mean) / torch.sqrt(variance + self.eps)

        # Apply weights and biases
        weight_reshaped = self.weight[:group_size * self.group_count].view(1, self.group_count, group_size)
        bias_reshaped = self.bias[:group_size * self.group_count].view(1, self.group_count, group_size)

        x_scaled_shifted = weight_reshaped * x_normalized + bias_reshaped

        if num_neurons % self.group_count != 0:
            remainder_neurons = num_neurons - (group_size * self.group_count)
            remainder_x = x[:, -remainder_neurons:].view(batch_size, 1, remainder_neurons)
            remainder_mean = remainder_x.mean(dim=2, keepdim=True)
            remainder_variance = remainder_x.var(dim=2, keepdim=True, unbiased=False)
            remainder_normalized = (remainder_x - remainder_mean) / torch.sqrt(remainder_variance + self.eps)

            remainder_weight = self.weight[-remainder_neurons:].view(1, 1, remainder_neurons)
            remainder_bias = self.bias[-remainder_neurons:].view(1, 1, remainder_neurons)

            remainder_scaled_shifted = remainder_weight * remainder_normalized + remainder_bias

            x_scaled_shifted = torch.cat((x_scaled_shifted.view(batch_size, -1), remainder_scaled_shifted.view(batch_size, -1)), dim=1)
        else:
            x_scaled_shifted = x_scaled_shifted.view(batch_size, -1)

        return x_scaled_shifted

class DropConnectLinear(nn.Linear):
    """
    Linear layer with DropConnect regularization applied to weights during training
    """
    
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super(DropConnectLinear, self).__init__(in_features, out_features, bias=True)
        self.p = p  # Probability of keeping a connection

    def forward(self, input):
        if self.training:
            # Generate mask for weights
            weight_mask = torch.bernoulli(torch.full_like(self.weight, self.p)) / self.p
            weight = self.weight * weight_mask
            bias = self.bias # we don't apply dropconnect on bias term
        else:
            weight = self.weight
            bias = self.bias

        return F.linear(input, weight, bias)

class WSLinear(nn.Linear):
    """
    Linear layer with Weight Standardization applied to weights
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(WSLinear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        weight = self.weight

        # Compute mean and standard deviation
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1, keepdim=True) + 1e-5
        weight = weight / std

        return F.linear(input, weight, self.bias)

class WS_DropConnectLinear(nn.Linear):
    """
    Linear layer combining Weight Standardization and DropConnect
    """
    
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super(WS_DropConnectLinear, self).__init__(in_features, out_features, bias=bias)
        self.p = p  # Probability of keeping a connection

    def forward(self, input):
        weight = self.weight

        # Apply weight standardization
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1, keepdim=True) + 1e-5
        weight = weight / std

        if self.training:
            # Apply DropConnect
            weight_mask = torch.bernoulli(torch.full_like(weight, self.p)) / self.p
            weight = weight * weight_mask
            bias = self.bias
        else:
            bias = self.bias

        return F.linear(input, weight, bias)

class Net(nn.Module):
    """
    Fully connected neural network with configurable regularization strategies
    """
    
    def __init__(self, input_size, hidden_neurons, output_size=2,
                 use_group_norm=False,
                 use_shift_scale=False,
                 use_weight_standardization=False, use_dropconnect=False, dropconnect_p=0.5):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.use_group_norm = use_group_norm
        self.use_shift_scale = use_shift_scale
        self.use_weight_standardization = use_weight_standardization
        self.use_dropconnect = use_dropconnect
        self.dropconnect_p = dropconnect_p
        in_features = input_size

        if self.use_group_norm:
            self.group_norms = nn.ModuleList()  # Initialize only if group norm is used

        if self.use_shift_scale: # Initialize only if shift-scale is used
            self.shift_scale_layers = nn.ModuleList()
            
        for i, size in enumerate(hidden_neurons):
            layer = self._create_layer(in_features, size)
            self.layers.append(layer)
            
            group_count = max(1, size // 8)
            remainder = size % (group_count * 8)

            if remainder > 0:
                group_count += 1
                
            if self.use_group_norm:
                self.group_norms.append(CustomGroupNorm(num_neurons=size, group_count=group_count)) # NOTE: use custom group norm implementation to allow for flexible group assignments

            if self.use_shift_scale:
                self.shift_scale_layers.append(ShiftScaleLayer(group_count=group_count))

            in_features = size

        self.output_layer = nn.Linear(in_features, output_size, bias=True)
        self.activation = nn.LeakyReLU()

    def _create_layer(self, in_features, out_features):
        if self.use_dropconnect:
            if self.use_weight_standardization:
                return WS_DropConnectLinear(in_features, out_features, p=self.dropconnect_p)
            else:
                return DropConnectLinear(in_features, out_features, p=self.dropconnect_p)
        else:
            if self.use_weight_standardization:
                return WSLinear(in_features, out_features)
            else:
                return nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.use_group_norm:
                x = self.group_norms[i](x)
                
            if self.use_shift_scale:
                x = self.shift_scale_layers[i](x)
            
            x = self.activation(x)
            
        x = self.output_layer(x)
        
        return x

def create_optimizer(model, learning_rate=0.001, weight_decay=0.0):
    """
    Creates an Adam optimizer for the model

    Args:
        model (nn.Module): The model to optimize
        learning_rate (float): Learning rate for the optimizer
        weight_decay (float): L2 weight decay factor

    Returns:
        torch.optim.Optimizer: Configured Adam optimizer
    """
    
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def calculate_regularization_loss(model, l1=False, l2=False, l1_lambda=0.01, l2_lambda=0.01):
    """
    Computes L1 and/or L2 regularization loss over model parameters

    Args:
        model (nn.Module): The model being trained
        l1 (bool): Whether to include L1 loss
        l2 (bool): Whether to include L2 loss
        l1_lambda (float): L1 regularization strength
        l2_lambda (float): L2 regularization strength

    Returns:
        float: The total regularization loss
    """
    
    reg_loss = 0
    if l1:
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        reg_loss += l1_lambda * l1_norm
    if l2:
        l2_norm = sum((p ** 2).sum() for p in model.parameters())
        reg_loss += l2_lambda * l2_norm
    return reg_loss

def clean_datasets(x_train_file, y_train_file, x_test_file, y_test_file):
    """
    Cleans and aligns training and test datasets based on 'Key' column to ensure correct pairing

    Args:
        x_train_file (str): Path to training feature CSV
        y_train_file (str): Path to training labels CSV
        x_test_file (str): Path to test feature CSV
        y_test_file (str): Path to test labels CSV

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Clean train/test features and labels
    """
    
    X_train = pd.read_csv(x_train_file, sep=';')
    y_train = pd.read_csv(y_train_file, sep=';')
    X_test = pd.read_csv(x_test_file, sep=';')
    y_test = pd.read_csv(y_test_file, sep=';')

    # Ensure 'Key' column is included
    if 'Key' not in X_train.columns or 'Key' not in y_train.columns or 'Key' not in X_test.columns or 'Key' not in y_test.columns:
        raise ValueError("Each file must have a 'Key' column to match samples.")

    # Filter X_train to keep only the rows that have corresponding entries in y_train
    X_train_cleaned = X_train[X_train['Key'].isin(y_train['Key'])].reset_index(drop=True)

    # Filter X_test to keep only the rows that have corresponding entries in y_test
    X_test_cleaned = X_test[X_test['Key'].isin(y_test['Key'])].reset_index(drop=True)

    # Making sure that the keys are matching pairwise
    matching_train_keys = (X_train_cleaned['Key'] == y_train['Key']).all()
    matching_test_keys = (X_test_cleaned['Key'] == y_test['Key']).all()
    if matching_train_keys and matching_test_keys:
        print("Train and test splits are successfully loaded.")
    else:
        raise KeyError("The entries in the train and/or test splits do not match pairwise!")

    # Make sure the features and labels are in the right format (values only, labels flattened)
    X_train_cleaned = X_train_cleaned.drop('Key', axis=1).values
    X_test_cleaned = X_test_cleaned.drop('Key', axis=1).values
    y_train_cleaned = y_train.drop('Key', axis=1).values.flatten()
    y_test_cleaned = y_test.drop('Key', axis=1).values.flatten()

    return X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned

def run_comparison(data_dir, experiment_id, output_dir, hidden_neuron_count,
                   layouts, l1=False, l2=False, l1_lambda=0.01, l2_lambda=0.01,
                   use_group_norm=False, use_shift_scale=False, use_weight_standardization=False,
                   use_dropconnect=False, dropconnect_p=0.5, idx=0, total_configs=1):
    """
    Runs training, validation, testing, hyperparameter tuning, and evaluation
    for a given dataset and model configuration. Saves best-performing model and logs metrics to CSV

    Args:
        data_dir (str): Path to dataset
        experiment_id (str): Unique ID for experiment
        output_dir (str): Directory to save results and models
        hidden_neuron_count (dict): Dict of layout names to hidden layer sizes
        layouts (list): Layout names to evaluate
        l1 (bool): Whether to use L1 regularization
        l2 (bool): Whether to use L2 regularization
        l1_lambda (float): L1 regularization strength
        l2_lambda (float): L2 regularization strength
        use_group_norm (bool): Whether to use Group Normalization regularization
        use_shift_scale (bool): Whether to use Shift-Scale regularization
        use_weight_standardization (bool): Whether to use Weight Standardization regularization
        use_dropconnect (bool): Whether to use DropConnect regularization
        dropconnect_p (float): DropConnect probability of keeping a connection 
        idx (int): Unique ID for current configuration
        total_configs (int): Total number of configurations
    """
    
    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Start counter
    start_time = perf_counter()

    # Make output directories
    experiment_dir = join(output_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)

    # Load data paths
    data_dict = load_data_as_dict(data_dir)

    # Set up classifier parameters
    num_epochs = 500
    validation_split = 0.2
    patience = 300
    min_delta = 0.05
    batch_size = 32  # Adjust as needed
    
    # Define default hyperparameters
    default_params = {'learning_rate': 0.001, 'weight_decay': 0.0}
    
    # Define hyperparameter search space
    search_config = {
        'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001],
        'weight_decay': [0.0, 0.01, 0.001],
    }
    
    # Define order of hyperparameters to tune
    order_of_processing_params = ['learning_rate', 'weight_decay']
    
    # Initialize variable to store best performance
    best_val_loss = np.inf
    best_params = default_params.copy()
    best_model_state = None
    
    # Read and assign data (cleaned version, matching F-L pairs)
    X_train, X_test, y_train, y_test = clean_datasets(data_dict['trainF'][0], data_dict['trainL'][0], data_dict['testF'][0], data_dict['testL'][0])

    # Read and assign validation data if present
    if data_dict['validateF'] and data_dict['validateL']:
        X_val = pd.read_csv(data_dict['validateF'][0], delimiter=";", index_col=0).values
        y_val = pd.read_csv(data_dict['validateL'][0], delimiter=";", index_col=0).values.flatten()
    else:
        # Split training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, stratify=y_train, random_state=42)

    # Note: if the data is not already z-score standardized, uncomment the following section
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_val = scaler.transform(X_val)
    #X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors and move to device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu()), y=y_train.cpu().numpy())
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Define loss function with class weights
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    for layout in layouts:
        result_file_path = os.path.join(output_dir, f"{experiment_id}_{layout}_results.csv")
        if not os.path.exists(result_file_path):
            with open(result_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Config Idx", "Layout", "L1", "L2", "DC", "GN", 
                    "WS", "SCO", "ACC", "BACC", "SNS", "SPC", "PPV", "NPV"
                ])
        print(f"Using layout: {layout}")
        if layout not in hidden_neuron_count:
            print(f"Layout {layout} not found in hidden_neuron_count. Skipping.")
            continue
        hidden_neurons = hidden_neuron_count[layout]
        input_size = X_train.shape[1]

        # Initialize best parameters and model state for this layout
        best_val_loss = np.inf
        best_params = default_params.copy()
        best_model_state = None
        
        # Hyperparameter tuning
        for param in order_of_processing_params:
            best_loss_for_param = np.inf
            print(f"\nTuning hyperparameter: {param}")
            for value in search_config[param]:
                instance_params = best_params.copy()
                instance_params[param] = value
                
                # Train and evaluate model with current parameters
                val_loss, current_model_state = train_and_evaluate(
                    input_size, hidden_neurons, instance_params, train_loader, val_loader,
                    loss_fn, device, num_epochs, patience, min_delta,
                    l1, l2, l1_lambda, l2_lambda, use_group_norm, use_shift_scale, use_weight_standardization,
                    use_dropconnect, dropconnect_p
                )

                print(f"Value: {param}={value}, Validation Loss: {val_loss:.4f}")

                if val_loss < best_loss_for_param:
                    best_loss_for_param = val_loss
                    best_params[param] = value
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = current_model_state
                else:
                    # Early break if no improvement
                    break
            print(f"Best {param}: {best_params[param]} with Validation Loss: {best_loss_for_param:.4f}")

        # After tuning, evaluate the best model on the test set
        print("\nBest Hyperparameters:")
        print(best_params)
        # Load the best model state
        model = Net(input_size=input_size,
                    hidden_neurons=hidden_neurons,
                    use_group_norm=use_group_norm,
                    use_shift_scale=use_shift_scale,
                    use_weight_standardization=use_weight_standardization,
                    use_dropconnect=use_dropconnect,
                    dropconnect_p=dropconnect_p).to(device)
        print(model)
        model.load_state_dict(best_model_state)
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for test_inputs, test_targets in test_loader:
                test_inputs = test_inputs.to(device)
                test_targets = test_targets.to(device)
                y_pred_raw = model(test_inputs)
                _, y_pred = torch.max(y_pred_raw, 1)
                all_preds.append(y_pred.cpu())
                all_targets.append(test_targets.cpu())

        y_pred = torch.cat(all_preds)
        y_test_cpu = torch.cat(all_targets)
        
        # Calculate metrics
        acc = metrics.accuracy(y_test_cpu.numpy(), y_pred.numpy())
        bacc = metrics.balanced_accuracy(y_test_cpu.numpy(), y_pred.numpy())
        sns_value = metrics.sensitivity(y_test_cpu.numpy(), y_pred.numpy())
        spc_value = metrics.specificity(y_test_cpu.numpy(), y_pred.numpy())
        ppv_value = metrics.positive_predictive_value(y_test_cpu.numpy(), y_pred.numpy())
        npv_value = metrics.negative_predictive_value(y_test_cpu.numpy(), y_pred.numpy())

        # Print results
        print(f"\nTest Performance with Best Hyperparameters:")
        print(f"ACC: {acc:.4f}")
        print(f"BACC: {bacc:.4f}")
        print(f"SNS: {sns_value:.4f}")
        print(f"SPC: {spc_value:.4f}")
        print(f"PPV: {ppv_value:.4f}")
        print(f"NPV: {npv_value:.4f}")
        
        with open(result_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                idx + 1, layout, l1, l2, use_dropconnect, use_group_norm,
                use_weight_standardization, use_shift_scale, acc, bacc,
                sns_value, spc_value, ppv_value, npv_value
            ])

        # Save the best model
        trained_models_dir = join(experiment_dir, 'trained_models/')
        os.makedirs(trained_models_dir, exist_ok=True)

        # Create model filename including exp_id, layout, and best hyperparameters
        hyperparams_str = "_".join([f"{k}_{v}" for k, v in best_params.items()])
        reg_config = f"L1_{l1}_L2_{l2}_GN_{use_group_norm}_WS_{use_weight_standardization}_DC_{use_dropconnect}"
        model_filename = f"{exp_id}_{layout}_{reg_config}_{hyperparams_str}.pt"
        model_path = join(trained_models_dir, model_filename)

        # Save the model state dict
        torch.save(best_model_state, model_path)
        print(f"Best model saved to {model_path}")
        
    # Report run time
    end_time = perf_counter()
    run_time = end_time - start_time
    print("-" * 40)
    print(f"Run time in hh:mm:ss.us: {timedelta(seconds=run_time)}")


def train_and_evaluate(input_size, hidden_neurons, params, train_loader, val_loader,
                       loss_fn, device, num_epochs, patience, min_delta,
                       l1, l2, l1_lambda, l2_lambda, use_group_norm, use_shift_scale, use_weight_standardization,
                       use_dropconnect, dropconnect_p):
    """
    Trains a neural network and evaluates it on a validation set with early stopping

    This function builds a model based on the given architecture and regularization
    configuration, trains it using the provided training DataLoader, and evaluates
    validation performance at each epoch. The best-performing model (lowest validation loss)
    is saved and returned
    
    Args:
        input_size (int): Number of input features
        hidden_neurons (list of int): Sizes of hidden layers
        params (dict): Dictionary containing hyperparameters such as 'learning_rate' and 'weight_decay'
        train_loader (DataLoader): PyTorch DataLoader for the training data
        val_loader (DataLoader): PyTorch DataLoader for the validation data
        loss_fn (callable): Loss function
        device (torch.device): Device to run the model on ('cpu' or 'cuda')
        num_epochs (int): Maximum number of training epochs
        patience (int): Number of epochs to wait for improvement before early stopping
        min_delta (float): Minimum improvement in validation loss to reset patience
        l1 (bool): Whether to use L1 regularization
        l2 (bool): Whether to use L2 regularization
        l1_lambda (float): L1 regularization strength
        l2_lambda (float): L2 regularization strength
        use_group_norm (bool): Whether to use Group Normalization regularization
        use_shift_scale (bool): Whether to use Shift-Scale regularization
        use_weight_standardization (bool): Whether to use Weight Standardization regularization
        use_dropconnect (bool): Whether to use DropConnect regularization
        dropconnect_p (float): DropConnect probability of keeping a connection 

    Returns:
        Tuple[float, dict]: Best validation loss and best model state dict
    """
    
    model = Net(input_size=input_size,
                hidden_neurons=hidden_neurons,
                use_group_norm=use_group_norm,
                use_shift_scale=use_shift_scale,
                use_weight_standardization=use_weight_standardization,
                use_dropconnect=use_dropconnect,
                dropconnect_p=dropconnect_p).to(device)

    optimizer = create_optimizer(model, learning_rate=params['learning_rate'], weight_decay=params['weight_decay'])

    best_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            reg_loss = calculate_regularization_loss(model, l1, l2, l1_lambda, l2_lambda)
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss = loss_fn(val_outputs, val_targets)
                val_running_loss += val_loss.item() * val_inputs.size(0)

        val_loss_value = val_running_loss / len(val_loader.dataset)
        
        #print(f"Epoch {epoch} - Train loss: {epoch_loss:.4f} - Val loss: {val_loss_value:.4f}")

        # Early stopping
        if val_loss_value + min_delta < best_loss:
            best_loss = val_loss_value
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}!")
            break

    return best_loss, best_model_state

def run_all_configurations(path_to_dataset, exp_id, output_dir, hidden_neuron_count, layouts):
    """
    Runs the full pipeline across all combinations of regularization settings.
    This is used to benchmark all architecture + regularizer configurations.
    Note: we have a total of 128 configurations in DEBI-NN (2^7 configs for 7 REG params)
    For Baseline NN (this model), we don't have SD, so we only have 2^6 = 64 configs:
        L1 (l1): 0 or 0.01
        L2 (l2): 0 or 0.01
        DC (use_dropconnect): True or False
        GN (use_group_norm): True or False
        WS (use_weight_standardization): True or False
        SCO (use_shift_scale): True or False

    Args:
        path_to_dataset (str): Path to dataset folder
        exp_id (str): Experiment identifier
        output_dir (str): Output directory for logs and models
        hidden_neuron_count (dict): Layout to hidden neuron count mapping
        layouts (list): Layouts to evaluate
    """
    
    # Set possible values for each regulaization parameter
    l1_options = [0, 0.01]
    l2_options = [0, 0.01]
    dropconnect_options = [True, False]
    group_norm_options = [True, False]
    weight_standardization_options = [True, False]
    shift_scale_options = [True, False]

    # Generate all possible configurations
    configurations = list(itertools.product(
        l1_options,
        l2_options,
        dropconnect_options,
        group_norm_options,
        weight_standardization_options,
        shift_scale_options
    ))

    # Iterate through each configuration
    for idx, (l1, l2, use_dropconnect, use_group_norm, use_weight_standardization, use_shift_scale) in enumerate(configurations):
        print(f"\nRunning configuration {idx + 1}/{len(configurations)}:")
        print(f"L1: {l1}, L2: {l2}, DC: {use_dropconnect}, GN: {use_group_norm}, WS: {use_weight_standardization}, SCO: {use_shift_scale}")

        run_comparison(
            path_to_dataset, 
            exp_id, 
            output_dir,
            hidden_neuron_count=hidden_neuron_count,
            layouts=layouts,
            l1=(l1 != 0),  # Set to True if l1 is non-zero
            l2=(l2 != 0),  # Set to True if l2 is non-zero
            l1_lambda=l1,
            l2_lambda=l2,
            use_group_norm=use_group_norm,
            use_shift_scale=use_shift_scale,
            use_weight_standardization=use_weight_standardization,
            use_dropconnect=use_dropconnect,
            dropconnect_p=0.5,
            idx=idx, 
            total_configs=64
        )

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = str(42)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = [
        'data/DLBCL/Dataset-1Fold',
        'data/HECKTOR-LITE/Dataset-1Fold',
        'data/PROSTATE/APR-Dataset-1Fold',
        'data/PROSTATE/PRA-Dataset-1Fold',
        'data/PROSTATE/PRS-Dataset-1Fold',
        'data/PROSTATE/SPR-Dataset-1Fold'
    ]
    experiment_id = [
        'Dlbcl',
        'HecktorLite',
        'ProstateAPR',
        'ProstatePRA',
        'ProstatePRS',
        'ProstateSPR'
    ]

    # possible layouts: ["1H", "2H-Block", "2H-Hinton", "3H-Block", "3H-Hinton"]
    layouts = ["2H-Block"]
    path_to_dataset = 'data/DLBCL/Dataset-1Fold'
    exp_id = 'Dlbcl'
    hidden_neuron_count = {"1H": [85], "2H-Block": [34, 34], "2H-Hinton": [85, 64],
                                   "3H-Block": [34, 34, 34], "3H-Hinton": [85, 64, 48]}
    
    # Note: to run a single configuration manually, uncomment the following function call, and comment out sequential run below
    # run_comparison(path_to_dataset, exp_id, output_dir,
    #                    hidden_neuron_count=hidden_neuron_count,
    #                    layouts=layouts,
    #                    l1=False,    # Set to True to use L1 regularization
    #                    l2=False,     # Set to True to use L2 regularization
    #                    l1_lambda=0, #0.01,
    #                    l2_lambda=0, #0.01,
    #                    use_group_norm=False,   # Set to True to use group normalization
    #                    use_shift_scale=False,  # Set to True to use shift-scale optimization
    #                    use_weight_standardization=True,  # Enable weight standardization
    #                    use_dropconnect=True,  # Enable DropConnect
    #                    dropconnect_p=0.5)      # Set DropConnect probability
    
    
    # Run all configurations sequentially
    run_all_configurations(
        path_to_dataset=path_to_dataset,
        exp_id=exp_id,
        output_dir=output_dir,
        hidden_neuron_count=hidden_neuron_count,
        layouts=layouts
    )

    # Run all experiments sequentially (for all datasets)
    # for path_to_dataset, exp_id in zip(data_dir, experiment_id):
    #     if exp_id == "Dlbcl":
    #         hidden_neuron_count = {"1H": [85], "2H-Block": [34, 34], "2H-Hinton": [85, 64],
    #                                "3H-Block": [34, 34, 34], "3H-Hinton": [85, 64, 48]}
    #     elif exp_id == "HecktorLite":
    #         hidden_neuron_count = {"1H": [180], "2H-Block": [72, 72], "2H-Hinton": [180, 135],
    #                                "3H-Block": [72, 72, 72], "3H-Hinton": [180, 135, 100]}
    #     elif "Prostate" in exp_id:
    #         hidden_neuron_count = {"1H": [150], "2H-Block": [60, 60], "2H-Hinton": [150, 113],
    #                                "3H-Block": [60, 60, 60], "3H-Hinton": [150, 113, 85]}
    #     else:
    #         raise ValueError("Invalid hidden neuron counts")

    #     print(exp_id, hidden_neuron_count)
    #     run_comparison(path_to_dataset, exp_id, output_dir,
    #                    hidden_neuron_count=hidden_neuron_count,
    #                    layouts=layouts,
    #                    l1=False,    # Set to True to use L1 regularization
    #                    l2=True,     # Set to True to use L2 regularization
    #                    l1_lambda=0.01,
    #                    l2_lambda=0.01,
    #                    use_group_norm=False,   # Set to True to use group normalization
    #                    use_weight_standardization=False,  # Enable weight standardization
    #                    use_dropconnect=False,  # Enable DropConnect
    #                    dropconnect_p=0.5)      # Set DropConnect probability