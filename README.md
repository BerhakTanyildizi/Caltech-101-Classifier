# Oxford-IIIT Pet Classifier

Pet image classification using MobileNetV3 and transfer learning on PyTorch.

## Overview

Transfer learning approach to classify pet images from the Oxford-IIIT Pet dataset using a pre-trained MobileNetV3 model. Efficient classification with feature freezing and fine-tuning.

## 📌 Important Note on Notebook Rendering

If GitHub displays an **"An error occurred"** message while trying to view `Caltech-101.ipynb`, please **download the file** and open it locally (via VS Code, Jupyter Notebook, or Google Colab). 

The repository's render engine sometimes fails due to GitHub's internal timeout limitations, but the file itself is completely intact and opens perfectly fine when downloaded or cloned.

## Key Features

- **Transfer Learning**: Pre-trained MobileNetV3 weights from ImageNet
- **Data Augmentation**: RandomCrop, HorizontalFlip, ColorJitter for robust training
- **GPU Support**: Automatic GPU detection and CUDA support
- **Efficient Architecture**: MobileNetV3-Small for fast inference
- **Complete Pipeline**: Training and testing with accuracy metrics

## Dataset

**Oxford-IIIT Pet Dataset**:
- 37 pet categories (dogs and cats)
- 7,394 images total
- Auto-downloaded via torchvision

## Requirements

- Python 3.8+
- PyTorch
- TorchVision
- TorchMetrics
- TorchInfo
- Matplotlib
- NumPy
- Pillow

Install dependencies:
```bash
pip install -r requirements.txt
```

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/BerhakTanyildizi/oxford-pet-classifier.git
cd oxford-pet-classifier
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run Training

```bash
python main.py
```

The script will:
- Download Oxford-IIIT Pet dataset
- Create and setup MobileNetV3 model
- Train for 10 epochs
- Print accuracy metrics each epoch

### Configuration

Edit `config.py` to customize:
- `BATCH_SIZE` - Default: 64
- `LEARNING_RATE` - Default: 0.001
- `EPOCHS` - Default: 10
- `DEVICE` - Auto-detected (cuda/cpu)

## Project Structure

```
oxford-pet-classifier/
├── main.py                 # Training script
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── Caltech-101.ipynb      # Jupyter notebook
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── LICENSE                # MIT License
└── .gitignore             # Git ignore rules
```

## Model Architecture

- **Base Model**: MobileNetV3-Small (pre-trained)
- **Parameters**: ~1.5M total, ~629K trainable
- **Input Size**: 224x224 pixels
- **Output Classes**: 37 pet categories

## Training Details

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 64
- **Data Augmentation**: RandomCrop, HorizontalFlip, ColorJitter
- **Feature Extraction**: Frozen (except classifier layer)

## Performance

Expected results after training:
- **Training Accuracy**: ~85-90%
- **Test Accuracy**: ~75-85%
- **Training Time**: ~30 minutes (GPU) / 3+ hours (CPU)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA Out of Memory | Reduce `BATCH_SIZE` in config.py |
| Slow Training | Verify GPU is being used |
| Import Errors | Reinstall requirements: `pip install -r requirements.txt` |

## References

- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02175)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [PyTorch Docs](https://pytorch.org/)
