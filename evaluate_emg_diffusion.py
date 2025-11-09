"""
Evaluate trained EMG-Diffusion model
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys

# Add src to path
sys.path.insert(0, 'src')

from models.emg_diffusion import EMGDiffusionModel
from data.data_loader import EMGDataLoader, create_data_split


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass (inference mode)
            logits = model(batch_x, training=False)

            # Compute loss
            loss = criterion(logits, batch_y)

            # Track metrics
            total_loss += loss.item()

            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading EMG Dataset...")
    train_loader_data = EMGDataLoader('.', dataset_type='training')
    test_loader_data = EMGDataLoader('.', dataset_type='testing')

    X_train, y_train, _ = train_loader_data.load_dataset()
    X_test, y_test, _ = test_loader_data.load_dataset()

    # Exclude Pinch class
    print("Excluding Pinch class (label 5)...")
    train_mask = y_train != 5
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    test_mask = y_test != 5
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    # Train/Val split
    X_train, X_val, y_train, y_val = create_data_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    num_classes = len(np.unique(y_train))

    print(f"\nDataset sizes:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Number of classes: {num_classes}")

    # Create datasets
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    # Create dataloaders
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Create model
    print("\nCreating model...")
    model = EMGDiffusionModel(
        in_channels=X_train.shape[1],
        num_classes=num_classes,
        d_model=256,
        nhead=8,
        num_layers=6,
        feature_dim=128,
        num_timesteps=100,
        hidden_dim=256
    ).to(device)

    # Load trained weights
    model_path = 'results/trial52_emg_diffusion/best_model.pth'
    print(f"\nLoading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    print("\n" + "="*60)
    print("Evaluating model...")
    print("="*60)

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nValidation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.2f}%")

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")

    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == '__main__':
    main()
