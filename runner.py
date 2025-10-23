"""
Runner script that loads parameter files and executes simulations.

Usage:
    python runner.py ./params/simulation_01/exp_0.yaml
"""

import sys
import os
import yaml
import torch
import torch.nn as nn
import time


class FullyConnectedNetwork(nn.Module):
    """
    Fully connected neural network with configurable width and depth.
    """
    def __init__(self, input_size, width, depth, output_size):
        """
        Args:
            input_size: Dimension of input features
            width: Dimension of each hidden layer
            depth: Number of hidden layers
            output_size: Dimension of output
        """
        super(FullyConnectedNetwork, self).__init__()

        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, width))
        layers.append(nn.ReLU())

        # Hidden layers (all have the same width dimension)
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # Final hidden layer to output layer
        layers.append(nn.Linear(width, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_param(yaml_path):
    """
    Load a single parameter configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML parameter file

    Returns:
        Dictionary containing the parameter configuration
    """
    with open(yaml_path, 'r') as f:
        param = yaml.safe_load(f)
    return param

def build_network(param, input_size=784, output_size=10):
    """
    Build a fully connected neural network based on parameters.

    Args:
        param: Dictionary containing 'widths' and 'depths' parameters
        input_size: Dimension of input layer (default: 784 for MNIST 28x28 images)
        output_size: Dimension of output layer (default: 10 for 10 classes)

    Returns:
        FullyConnectedNetwork instance
    """
    width = param['widths']
    depth = param['depths']

    model = FullyConnectedNetwork(input_size, width, depth, output_size)

    return model

def build_dataloader(param):
    """
    Build a data loader for learning the Gaussian function f(x) = exp(-x²/2).

    Args:
        param: Dictionary containing:
            - x_min: Minimum x value
            - x_max: Maximum x value
            - dx: Spacing between x points
            - n_samples: Number of training samples

    Returns:
        Training data loader
        Test data loader
    """
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader

    # Generate training data: random samples in [x_min, x_max]
    x_train = np.random.uniform(param['x_min'], param['x_max'], size=(param['n_samples'], 1))
    y_train = np.exp(-x_train**2 / 2)

    # Generate test data: uniform grid for evaluation
    x_test = np.arange(param['x_min'], param['x_max'] + param['dx'], param['dx']).reshape(-1, 1)
    y_test = np.exp(-x_test**2 / 2)

    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)),
        batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test)),
        batch_size=32, shuffle=False
    )

    return train_loader, test_loader


def save_training_history(train_losses, test_losses, result, max_experiments=4):
    """
    Save complete training history to CSV for the first few experiments.

    Args:
        train_losses: List of training losses per epoch
        test_losses: List of test losses per epoch
        result: Dictionary containing experiment metadata
        max_experiments: Maximum number of experiments to save history for (default: 4)
    """
    import csv

    # Only save for first max_experiments (inclusive)
    if result['experiment_index'] > max_experiments:
        return

    # Create history directory
    history_dir = os.path.join('results', result['name'], 'history')
    os.makedirs(history_dir, exist_ok=True)

    # Create CSV file
    history_file = os.path.join(history_dir, f"exp_{result['experiment_index']}_history.csv")

    with open(history_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'test_loss'])

        for epoch, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses), start=1):
            writer.writerow([epoch, train_loss, test_loss])

    print(f"Training history saved to: {history_file}")


def main():
    """Main entry point for the runner script."""
    if len(sys.argv) != 2:
        print("Usage: python runner.py <path_to_yaml_file>")
        sys.exit(1)

    yaml_path = sys.argv[1]

    # Load parameter configuration
    param = load_param(yaml_path)

    print(f"Loaded parameters from: {yaml_path}")
    print(f"Parameters: {param}")
    print()

    # Rename param to result before training begins
    result = param.copy()

    # Add metadata about the target function
    result['target_function'] = 'exp(-x^2/2)'
    result['target_function_description'] = 'Gaussian function with mean=0, simplified form'

    # Build model and dataloaders
    model = build_network(result, input_size=1, output_size=1)
    train_loader, test_loader = build_dataloader(result)

    print(f"Model architecture:")
    print(model)
    print()

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Training setup
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = param['num_epochs']

    # Track losses
    train_losses = []
    test_losses = []

    # Training loop
    print("Starting training...")
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

    end_time = time.time()
    training_time = end_time - start_time

    print("\nTraining complete!")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Add training statistics to result
    # Select evenly spaced checkpoints (first, last, and 8 evenly spaced in between)
    num_checkpoints = 10
    checkpoint_indices = [int(i * (num_epochs - 1) / (num_checkpoints - 1)) for i in range(num_checkpoints)]

    result['train_losses'] = [float(train_losses[i]) for i in checkpoint_indices]
    result['test_losses'] = [float(test_losses[i]) for i in checkpoint_indices]
    result['checkpoint_epochs'] = [i + 1 for i in checkpoint_indices]
    result['final_train_loss'] = float(train_losses[-1])
    result['final_test_loss'] = float(test_losses[-1])
    result['num_epochs'] = num_epochs
    result['learning_rate'] = 0.001
    result['optimizer'] = 'Adam'
    result['loss_function'] = 'MSE'

    # Performance metrics
    result['training_time_seconds'] = float(training_time)
    result['training_time_minutes'] = float(training_time / 60)
    result['seconds_per_epoch'] = float(training_time / num_epochs)

    # Best performance metrics
    best_test_loss = float(min(test_losses))
    best_test_epoch = int(test_losses.index(min(test_losses)) + 1)
    result['best_test_loss'] = best_test_loss
    result['best_test_epoch'] = best_test_epoch

    # Convergence metrics
    result['generalization_gap'] = float(result['final_test_loss'] - result['final_train_loss'])
    result['overfit_ratio'] = float(result['final_test_loss'] / result['final_train_loss'])

    # Model complexity
    result['total_parameters'] = int(total_params)
    result['trainable_parameters'] = int(trainable_params)

    # Dataset sizes
    result['train_dataset_size'] = int(param['n_samples'])
    result['test_dataset_size'] = int(len(test_loader.dataset))

    # System information
    result['device'] = str(next(model.parameters()).device)

    print(f"\nFinal train loss: {result['final_train_loss']:.6f}")
    print(f"Final test loss: {result['final_test_loss']:.6f}")
    print(f"Best test loss: {result['best_test_loss']:.6f} (epoch {result['best_test_epoch']})")
    print(f"Generalization gap: {result['generalization_gap']:.6f}")

    # Save complete training history for first few experiments
    save_training_history(train_losses, test_losses, result, max_experiments=4)

    # Save results to ./results/{name}/runs/exp_{experiment_index}.yaml
    runs_dir = os.path.join('results', result['name'], 'runs')
    os.makedirs(runs_dir, exist_ok=True)

    results_file = os.path.join(runs_dir, f"exp_{result['experiment_index']}.yaml")
    with open(results_file, 'w') as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
