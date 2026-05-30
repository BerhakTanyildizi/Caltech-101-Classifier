"""
Utility functions for Oxford-IIIT Pet Classification project
"""

import torch
import os
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List
import json


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_directories(paths: List[Path]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths: List of Path objects to create
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_model(model: torch.nn.Module, filepath: Path) -> None:
    """
    Save model state dictionary.
    
    Args:
        model: PyTorch model
        filepath: Path to save the model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model: torch.nn.Module, filepath: Path, device: str) -> torch.nn.Module:
    """
    Load model state dictionary.
    
    Args:
        model: PyTorch model architecture
        filepath: Path to saved model
        device: Device to load model to
        
    Returns:
        Model with loaded weights
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    print(f"Model loaded from {filepath}")
    return model


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: Path,
    **kwargs
) -> None:
    """
    Save complete training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        filepath: Path to save checkpoint
        **kwargs: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: Path, model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer = None, device: str = 'cpu'):
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load to
        
    Returns:
        Tuple of (model, optimizer, epoch)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    
    return model, optimizer, epoch


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else -1,
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_capability'] = torch.cuda.get_device_capability(0)
        info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info


def print_device_info() -> None:
    """Print device information."""
    info = get_device_info()
    
    print("\n" + "="*50)
    print("Device Information")
    print("="*50)
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Device Count: {info['device_count']}")
        print(f"Current Device: {info['current_device']}")
        print(f"Device Name: {info['device_name']}")
        print(f"Device Capability: {info['device_capability']}")
        print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
    else:
        print("Using CPU for computation")
    
    print(f"PyTorch Version: {torch.__version__}")
    print("="*50 + "\n")


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_info(model: torch.nn.Module) -> None:
    """Print model parameter information."""
    total, trainable = count_parameters(model)
    
    print("\n" + "="*50)
    print("Model Information")
    print("="*50)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Non-trainable Parameters: {total - trainable:,}")
    print(f"Trainable Ratio: {100 * trainable / total:.2f}%")
    print("="*50 + "\n")


def save_metrics(metrics: dict, filepath: Path) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save metrics
    """
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath: Path) -> dict:
    """
    Load metrics from JSON file.
    
    Args:
        filepath: Path to metrics file
        
    Returns:
        Dictionary of metrics
    """
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        """
        Initialize EarlyStopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> None:
        """
        Check if training should stop.
        
        Args:
            val_loss: Validation loss
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("EarlyStopping triggered!")
