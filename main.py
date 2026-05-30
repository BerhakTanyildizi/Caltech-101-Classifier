"""
Oxford-IIIT Pet Image Classification using MobileNetV3
A transfer learning approach for pet image classification.
"""

import torch
import torchvision
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from torchinfo import summary
from torchvision import transforms


# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
DATA_PATH = Path('oxford-iiit-pet')


def explore_data(data_path):
    """Explore the dataset structure."""
    print(f"\n--- Dataset Structure ---")
    for dirpath, dirnames, filenames in os.walk(data_path):
        print(f'Directories: {len(dirnames)}, Images: {len(filenames)} in {dirpath}')


def setup_transforms():
    """Setup data augmentation and normalization transforms."""
    # Train transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
    ])

    # Test transforms without augmentation
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, test_transforms


def load_datasets(train_transforms, test_transforms):
    """Load Oxford-IIIT Pet dataset."""
    print("\n--- Loading Datasets ---")
    train_data = torchvision.datasets.OxfordIIITPet(
        root='',
        split='trainval',
        download=True,
        transform=train_transforms
    )
    
    test_data = torchvision.datasets.OxfordIIITPet(
        root='',
        split='test',
        download=True,
        transform=test_transforms
    )
    
    return train_data, test_data


def create_dataloaders(train_data, test_data, batch_size=BATCH_SIZE):
    """Create DataLoaders for training and testing."""
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_dataloader, test_dataloader


def setup_model(num_classes):
    """Setup MobileNetV3 model with transfer learning."""
    print("\n--- Setting up Model ---")
    
    # Load pre-trained weights
    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
    model = torchvision.models.mobilenet_v3_small(weights=weights).to(DEVICE)
    
    # Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Modify classifier for our dataset
    model.classifier[3] = nn.Linear(
        in_features=1024,
        out_features=num_classes,
        bias=True
    ).to(DEVICE)
    
    print(f"Model moved to {DEVICE}")
    
    return model


def plot_images(dataloader, data_num=2, class_names=None):
    """Plot sample images from dataloader."""
    images, labels = next(iter(dataloader))
    
    img = images[data_num].permute(1, 2, 0)
    label = labels[data_num]
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    if class_names:
        plt.title(f'Class: {class_names[label]}')
    else:
        plt.title(f'Label: {label}')
    plt.show()


def train_step(model, dataloader, loss_fn, optimizer, device):
    """Single training step."""
    model.train()
    
    train_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred_logits = model(X)
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        y_pred_labels = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
        correct_predictions += (y_pred_labels == y).sum().item()
        total_samples += y.size(0)
    
    train_loss /= len(dataloader)
    train_acc = correct_predictions / total_samples
    
    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device):
    """Single testing step."""
    model.eval()
    
    test_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            y_pred_logits = model(X)
            loss = loss_fn(y_pred_logits, y).item()
            test_loss += loss
            
            # Calculate accuracy
            y_pred_labels = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
            correct_predictions += (y_pred_labels == y).sum().item()
            total_samples += y.size(0)
        
        test_loss /= len(dataloader)
        test_acc = correct_predictions / total_samples
    
    return test_loss, test_acc


def train_and_test(model, train_dataloader, test_dataloader, loss_fn, optimizer, 
                   device, epochs=EPOCHS):
    """Training loop with validation."""
    print(f"\n--- Training for {epochs} epochs ---")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        print(
            f"Epoch: {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )


def main():
    """Main execution function."""
    print("=" * 50)
    print("Oxford-IIIT Pet Classification with MobileNetV3")
    print("=" * 50)
    
    # Explore data
    explore_data(DATA_PATH)
    
    # Setup transforms
    train_transforms, test_transforms = setup_transforms()
    
    # Load datasets
    train_data, test_data = load_datasets(train_transforms, test_transforms)
    class_names = train_data.classes
    num_classes = len(class_names)
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Sample classes: {class_names[:5]}")
    
    # Create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(
        train_data, test_data, BATCH_SIZE
    )
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Setup model
    model = setup_model(num_classes)
    
    # Setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Print model summary
    print("\n--- Model Summary ---")
    summary(
        model=model,
        input_size=[32, 3, 256, 256],
        col_names=['input_size', 'num_params', 'trainable']
    )
    
    # Train and test
    train_and_test(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS
    )
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
