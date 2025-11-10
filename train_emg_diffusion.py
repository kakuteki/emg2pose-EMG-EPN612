"""
Training script for EMG-Diffusion model

Two-stage architecture:
    EMG → Transformer Feature Extractor → Diffusion Classifier → Gesture Class

Usage:
    python train_emg_diffusion.py --exclude_pinch --epochs 100 --batch_size 64
"""

import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, 'src')

from models.emg_diffusion import EMGDiffusionModel, count_parameters
from data.data_loader import EMGDataLoader, create_data_split


def parse_args():
    parser = argparse.ArgumentParser(description='Train EMG-Diffusion model')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='.',
                        help='Path to data directory')
    parser.add_argument('--exclude_pinch', action='store_true',
                        help='Exclude Pinch class (class 5)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')

    # Model parameters
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of Transformer layers')
    parser.add_argument('--feature_dim', type=int, default=128,
                        help='Feature dimension')
    parser.add_argument('--num_timesteps', type=int, default=100,
                        help='Number of diffusion timesteps')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Diffusion hidden dimension')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--denoise_weight', type=float, default=0.1,
                        help='Weight for denoising loss')

    # Output parameters
    parser.add_argument('--save_dir', type=str, default='results/trial52_emg_diffusion',
                        help='Directory to save results')

    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device, denoise_weight):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_denoise_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in tqdm(dataloader, desc="Training"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, loss_dict = model(batch_x, training=True)

        # Compute losses
        cls_loss = criterion(logits, batch_y)
        denoise_loss = loss_dict['denoise_loss']

        # Combined loss
        loss = cls_loss + denoise_weight * denoise_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_denoise_loss += denoise_loss.item()

        _, predicted = logits.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_denoise_loss = total_denoise_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, avg_cls_loss, avg_denoise_loss, accuracy


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


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Total Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Classification Loss
    axes[0, 1].plot(history['train_cls_loss'], label='Train Cls Loss', marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Denoising Loss
    axes[1, 0].plot(history['train_denoise_loss'], label='Train Denoise Loss', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Denoising Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Accuracy
    axes[1, 1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1, 1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1, 1].plot(history['test_acc'], label='Test Acc', marker='^')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("EMG-Diffusion Model Training")
    print("="*60)

    # Load data
    print("\nLoading EMG Dataset...")
    train_loader_data = EMGDataLoader(args.data_path, dataset_type='training')
    test_loader_data = EMGDataLoader(args.data_path, dataset_type='testing')

    X_train, y_train, _ = train_loader_data.load_dataset()
    X_test, y_test, _ = test_loader_data.load_dataset()

    # Exclude Pinch class if specified
    if args.exclude_pinch:
        print("Excluding Pinch class (label 5)...")
        train_mask = y_train != 5
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        test_mask = y_test != 5
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

    print(f"After loading - Train: {X_train.shape}, Test: {X_test.shape}")

    # Train/Val split
    X_train, X_val, y_train, y_val = create_data_split(
        X_train, y_train, test_size=0.2, random_state=args.random_state
    )

    num_classes = len(np.unique(y_train))

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Val set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Number of classes: {num_classes}")
    print(f"Input shape: {X_train.shape[1:]} (channels × time)")

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # Create model
    print("\nCreating model...")
    model = EMGDiffusionModel(
        in_channels=X_train.shape[1],
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        feature_dim=args.feature_dim,
        num_timesteps=args.num_timesteps,
        hidden_dim=args.hidden_dim
    ).to(device)

    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'train_cls_loss': [],
        'train_denoise_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_cls_loss, train_denoise_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.denoise_weight
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_acc)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['test_loss'].append(test_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_denoise_loss'].append(train_denoise_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Denoise: {train_denoise_loss:.4f})")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"[SAVED] Best model saved! (Val Acc: {best_val_acc:.2f}%, Test Acc: {best_test_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Save results
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")

    results = {
        'model_type': 'emg_diffusion',
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'total_params': total_params,
        'trainable_params': total_params,
        'args': vars(args),
        'history': history
    }

    # Save results JSON
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot training history
    print("\nGenerating plots...")
    plot_training_history(history, os.path.join(args.save_dir, 'training_history.png'))

    print(f"\nAll results saved to: {args.save_dir}")
    print("\nFiles saved:")
    print(f"  - best_model.pth")
    print(f"  - results.json")
    print(f"  - training_history.png")


if __name__ == '__main__':
    main()
