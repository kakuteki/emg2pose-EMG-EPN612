"""
Deep Learning training with moderate undersampling for EMG gesture recognition

Trial 3: Moderate Undersampling Strategy
- Remove Pinch class (1 sample only) → 5-class classification
- Undersample No Gesture from 70% to ~52% (4,412 → 2,000 samples)
- Keep ALL minority class samples (real data only)
- No synthetic data (avoid SMOTE overfitting)
- Target distribution: ~52% No Gesture : 48% gestures
"""
import sys
from pathlib import Path
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_loader import EMGDataLoader, create_data_split
from data.pytorch_dataset import create_dataloaders
from features.feature_extractor import EMGPreprocessor
from models.cnn_lstm import get_model


def remove_pinch_class(X, y, pinch_label=5):
    """
    Remove Pinch class (label 5) which has only 1 sample

    Args:
        X: Feature array (n_samples, n_channels, seq_length)
        y: Label array (n_samples,)
        pinch_label: Label for Pinch class (default: 5)

    Returns:
        X_filtered, y_filtered: Arrays without Pinch class
    """
    mask = y != pinch_label
    X_filtered = X[mask]
    y_filtered = y[mask]

    print(f"\n[Pinch Removal]")
    print(f"  Original samples: {len(y)}")
    print(f"  Pinch samples removed: {(~mask).sum()}")
    print(f"  Remaining samples: {len(y_filtered)}")

    return X_filtered, y_filtered


def moderate_undersample_no_gesture(X, y, no_gesture_label=0, target_samples=2000, random_state=42):
    """
    Moderately undersample No Gesture class to improve class balance

    Strategy:
    - No Gesture: 4,412 samples (70.4%) → 2,000 samples (~52%)
    - All other classes: Keep ALL samples (1,856 samples → 48%)
    - Total: 6,268 → 3,856 samples
    - New distribution: 52:48 (much better than 70:30)

    Args:
        X: Feature array (n_samples, n_channels, seq_length)
        y: Label array (n_samples,)
        no_gesture_label: Label for No Gesture class (default: 0)
        target_samples: Target number of No Gesture samples (default: 2000)
        random_state: Random seed for reproducibility

    Returns:
        X_resampled, y_resampled: Undersampled arrays
    """
    np.random.seed(random_state)

    # Get original distribution
    print(f"\n[Original Distribution (after Pinch removal)]")
    unique_labels, counts = np.unique(y, return_counts=True)
    total = len(y)
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count:5d} samples ({count/total*100:5.2f}%)")

    # Separate No Gesture and other classes
    no_gesture_mask = (y == no_gesture_label)
    other_mask = ~no_gesture_mask

    X_no_gesture = X[no_gesture_mask]
    y_no_gesture = y[no_gesture_mask]
    X_other = X[other_mask]
    y_other = y[other_mask]

    print(f"\n[Undersampling No Gesture]")
    print(f"  Original No Gesture: {len(y_no_gesture)} samples")
    print(f"  Target No Gesture: {target_samples} samples")
    print(f"  Reduction: {len(y_no_gesture) - target_samples} samples ({(1 - target_samples/len(y_no_gesture))*100:.1f}%)")

    # Randomly select target_samples from No Gesture
    if len(y_no_gesture) > target_samples:
        indices = np.random.choice(len(y_no_gesture), target_samples, replace=False)
        X_no_gesture_sampled = X_no_gesture[indices]
        y_no_gesture_sampled = y_no_gesture[indices]
    else:
        X_no_gesture_sampled = X_no_gesture
        y_no_gesture_sampled = y_no_gesture

    # Combine undersampled No Gesture with all other classes
    X_resampled = np.concatenate([X_no_gesture_sampled, X_other], axis=0)
    y_resampled = np.concatenate([y_no_gesture_sampled, y_other], axis=0)

    # Shuffle the combined dataset
    shuffle_idx = np.random.permutation(len(y_resampled))
    X_resampled = X_resampled[shuffle_idx]
    y_resampled = y_resampled[shuffle_idx]

    # Print new distribution
    print(f"\n[New Distribution (after undersampling)]")
    unique_labels, counts = np.unique(y_resampled, return_counts=True)
    total_new = len(y_resampled)
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count:5d} samples ({count/total_new*100:5.2f}%)")

    print(f"\n[Dataset Size Change]")
    print(f"  Original: {total} samples")
    print(f"  New: {total_new} samples")
    print(f"  Reduction: {total - total_new} samples ({(1 - total_new/total)*100:.1f}%)")

    return X_resampled, y_resampled


class Trainer:
    """Deep Learning model trainer for 5-class classification"""

    def __init__(self, model, device, train_loader, val_loader, test_loader,
                 criterion, optimizer, scheduler=None, num_epochs=50,
                 save_dir='results/deep_learning_undersampling'):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.save_dir / 'tensorboard')

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0

        # 5 classes (No Gesture, Fist, Wave In, Wave Out, Open)
        self.class_names = ['No Gesture', 'Fist', 'Wave In', 'Wave Out', 'Open']

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': running_loss / (pbar.n + 1),
                            'acc': 100. * correct / total})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({'loss': running_loss / (pbar.n + 1),
                                'acc': 100. * correct / total})

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("TRAINING START")
        print("="*80)

        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # Print summary
            print(f"\nEpoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")

        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("="*80)

        # Save final model
        torch.save(self.model.state_dict(), self.save_dir / 'final_model.pth')

        # Plot training curves
        self.plot_training_curves()

    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curve
        ax1.plot(self.train_losses, label='Train Loss', marker='o', markersize=3)
        ax1.plot(self.val_losses, label='Val Loss', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curve
        ax2.plot(self.train_accs, label='Train Acc', marker='o', markersize=3)
        ax2.plot(self.val_accs, label='Val Acc', marker='s', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Training curves saved to {self.save_dir / 'training_curves.png'}")
        plt.close()

    def test(self):
        """Evaluate on test set"""
        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)

        # Load best model
        self.model.load_state_dict(torch.load(self.save_dir / 'best_model.pth'))
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Testing'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        test_acc = accuracy_score(all_labels, all_preds)
        print(f"\nTest Accuracy: {test_acc*100:.2f}%")

        # Classification report
        print("\n" + "-"*80)
        print("Classification Report:")
        print("-"*80)
        report = classification_report(all_labels, all_preds,
                                       target_names=self.class_names,
                                       digits=4)
        print(report)

        # Save report
        with open(self.save_dir / 'classification_report.txt', 'w') as f:
            f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)

        return test_acc, all_preds, all_labels

    def plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Percentage'})

        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {self.save_dir / 'confusion_matrix.png'}")
        plt.close()


def main(args):
    """Main training function"""

    print("\n" + "="*80)
    print("EMG GESTURE RECOGNITION - TRIAL 3: MODERATE UNDERSAMPLING")
    print("="*80)
    print("\nTrial 3 Strategy:")
    print("  1. Remove Pinch class (1 sample only)")
    print("  2. Undersample No Gesture: 70.4% → ~52% (4,412 → 2,000 samples)")
    print("  3. Keep ALL minority class samples (real data only)")
    print("  4. No synthetic data (avoid SMOTE overfitting)")
    print("  5. Target: 52:48 class distribution")
    print("="*80)

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] Using: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\n[Data Loading] Loading from: {args.data_path}")
    data_loader = EMGDataLoader(args.data_path)
    data_loader.load_all_users(max_users=args.max_users)

    # Preprocess
    preprocessor = EMGPreprocessor()
    X, y = preprocessor.prepare_deep_learning_data(
        data_loader.features,
        data_loader.labels
    )

    print(f"\n[Original Data]")
    print(f"  Total samples: {len(y)}")
    print(f"  Feature shape: {X.shape}")

    # Remove Pinch class
    X, y = remove_pinch_class(X, y, pinch_label=5)

    # Apply moderate undersampling
    X_resampled, y_resampled = moderate_undersample_no_gesture(
        X, y,
        no_gesture_label=0,
        target_samples=args.no_gesture_samples,
        random_state=args.seed
    )

    # Create train/val/test splits
    print(f"\n[Data Split] Creating train/val/test splits")
    train_idx, val_idx, test_idx = create_data_split(
        len(y_resampled),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed
    )

    X_train = X_resampled[train_idx]
    y_train = y_resampled[train_idx]
    X_val = X_resampled[val_idx]
    y_val = y_resampled[val_idx]
    X_test = X_resampled[test_idx]
    y_test = y_resampled[test_idx]

    print(f"  Train: {len(y_train)} samples")
    print(f"  Val: {len(y_val)} samples")
    print(f"  Test: {len(y_test)} samples")

    # Calculate class weights for training
    unique_labels, counts = np.unique(y_train, return_counts=True)
    class_weights = 1.0 / (counts + 1)  # Add 1 to avoid division by zero
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)

    print(f"\n[Class Weights]")
    for label, weight in zip(unique_labels, class_weights.cpu().numpy()):
        print(f"  Class {label}: {weight:.4f}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model (5 classes)
    print(f"\n[Model] Creating {args.model_type} model")
    model = get_model(
        model_type=args.model_type,
        input_channels=X.shape[1],
        sequence_length=X.shape[2],
        num_classes=5,  # 5 classes (without Pinch)
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Create save directory with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"{args.model_type}_undersampling_{timestamp}"

    # Train
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        save_dir=save_dir
    )

    trainer.train()

    # Test
    test_acc, predictions, labels = trainer.test()

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print(f"Results saved to: {save_dir}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deep learning models with moderate undersampling')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='.',
                       help='Path to EMG dataset')
    parser.add_argument('--max_users', type=int, default=None,
                       help='Maximum number of users to load (None = all)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')

    # Undersampling parameters
    parser.add_argument('--no_gesture_samples', type=int, default=2000,
                       help='Target number of No Gesture samples after undersampling')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                       choices=['cnn_lstm', 'attention_lstm', 'attention_resnet18', 'transformer'],
                       help='Model architecture')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden size for LSTM/Transformer')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM/Transformer layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (L2 regularization)')

    # Other parameters
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='results/deep_learning',
                       help='Directory to save results')

    args = parser.parse_args()

    main(args)
