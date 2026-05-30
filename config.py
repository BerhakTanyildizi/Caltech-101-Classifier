"""
Configuration settings for Oxford-IIIT Pet Classification project
"""

import torch
from pathlib import Path

# Device Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRINT_DEVICE_INFO = True

# Data Configuration
DATA_PATH = Path('oxford-iiit-pet')
DATASET_NAME = 'OxfordIIITPet'
NUM_CLASSES = 37
AUTO_DOWNLOAD = True

# Model Configuration
MODEL_NAME = 'mobilenet_v3_small'
PRETRAINED = True
FREEZE_FEATURES = True
INPUT_SIZE = 224

# Training Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 5

# Data Augmentation Configuration
TRAIN_IMAGE_SIZE = 224
TEST_IMAGE_SIZE = 224
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# Augmentation Parameters
RANDOM_CROP_SIZE = 224
HORIZONTAL_FLIP_PROB = 0.5
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2

# Output Configuration
SAVE_MODEL_PATH = Path('models')
CHECKPOINT_PATH = Path('checkpoints')
RESULTS_PATH = Path('results')
LOG_PATH = Path('logs')

# Note: Call create_directories() from utils.py when needed
# Directories: models/, checkpoints/, results/, logs/

# Random Seeds for Reproducibility
TORCH_SEED = 42
NUMPY_SEED = 42
PYTHON_SEED = 42

# Logging Configuration
VERBOSE = True
LOG_INTERVAL = 10

# Performance
PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
