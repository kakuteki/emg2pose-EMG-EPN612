"""
Deep Learning models training script for EMG gesture recognition
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


class Trainer:
    """Deep Learning model trainer"""

    def __init__(self, model, device, train_loader, val_loader, test_loader,
                 criterion, optimizer, scheduler=None, num_epochs=50,
                 save_dir='results/deep_learning'):
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

        self.class_names = ['No Gesture', 'Fist', 'Wave In',
                           'Wave Out', 'Open', 'Pinch']

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
        """Complete training loop"""
        print(f"\n{'='*80}")
        print("Starting Training")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {self.num_epochs}")
        print(f"{'='*80}\n")

        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"  [*] New best model saved! (Val Acc: {val_acc:.2f}%)")

        print(f"\n{'='*80}")
        print("Training Completed!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*80}\n")

        # Plot training curves
        self.plot_training_curves()

    def evaluate(self, dataloader, dataset_name='Test'):
        """Evaluate on test set"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc=f'Evaluating on {dataset_name} set'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\n{'='*80}")
        print(f"{dataset_name} Set Evaluation")
        print(f"{'='*80}")
        print(f"Accuracy: {accuracy*100:.2f}%\n")

        # Get unique labels actually present in the data
        unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
        present_class_names = [self.class_names[i] for i in unique_labels if i < len(self.class_names)]

        print("Classification Report:")
        print(classification_report(all_labels, all_preds,
                                   labels=unique_labels,
                                   target_names=present_class_names,
                                   zero_division=0))

        # Plot confusion matrix
        self.plot_confusion_matrix(cm, unique_labels, dataset_name)

        return accuracy, cm, all_preds, all_labels

    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', linewidth=2)
        ax1.plot(self.val_losses, label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Acc', linewidth=2)
        ax2.plot(self.val_accs, label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {self.save_dir / 'training_curves.png'}")
        plt.close()

    def plot_confusion_matrix(self, cm, labels, dataset_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))

        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Get class names for present labels
        label_names = [self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
                      for i in labels]

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=label_names,
                   yticklabels=label_names,
                   cbar_kws={'label': 'Percentage'})

        plt.title(f'Confusion Matrix - {dataset_name} Set', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(self.save_dir / f'confusion_matrix_{dataset_name.lower()}.png',
                   dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {self.save_dir / f'confusion_matrix_{dataset_name.lower()}.png'}")
        plt.close()

    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }
        torch.save(checkpoint, self.save_dir / filename)


def main(args):
    """Main training function"""
    print("="*80)
    print("EMG Gesture Recognition - Deep Learning Training")
    print("="*80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # ===============================
    # 1. Load data
    # ===============================
    print("\n[Step 1/5] Loading dataset...")
    data_path = args.data_path

    train_loader_data = EMGDataLoader(data_path, dataset_type='training')
    X_train_raw, y_train, _ = train_loader_data.load_dataset(max_users=args.max_users)

    test_loader_data = EMGDataLoader(data_path, dataset_type='testing')
    X_test_raw, y_test, _ = test_loader_data.load_dataset(max_users=args.max_users)

    print(f"Training set: {X_train_raw.shape}")
    print(f"Test set: {X_test_raw.shape}")

    # ===============================
    # 2. Preprocess
    # ===============================
    print("\n[Step 2/5] Preprocessing...")
    preprocessor = EMGPreprocessor(sampling_rate=200)

    X_train_preprocessed = preprocessor.preprocess(X_train_raw, apply_bandpass=True,
                                                   apply_notch=True, normalize=True)
    X_test_preprocessed = preprocessor.preprocess(X_test_raw, apply_bandpass=True,
                                                  apply_notch=True, normalize=True)

    # Train/Val split
    X_train, X_val, y_train_split, y_val = create_data_split(
        X_train_preprocessed, y_train, test_size=args.val_split, random_state=args.random_state
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test_preprocessed.shape}")

    # ===============================
    # 3. Create DataLoaders
    # ===============================
    print("\n[Step 3/5] Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train_split, X_val, y_val, X_test_preprocessed, y_test,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ===============================
    # 4. Create model
    # ===============================
    print("\n[Step 4/5] Creating model...")
    model = get_model(
        model_type=args.model_type,
        input_channels=8,
        num_classes=6,
        dropout=args.dropout
    ).to(device)

    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train_split)
    class_weights = 1.0 / (class_counts + 1)  # +1 to avoid division by zero
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # ===============================
    # 5. Train
    # ===============================
    print("\n[Step 5/5] Training...")
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
        save_dir=f'results/{args.model_type}'
    )

    trainer.train()

    # ===============================
    # 6. Evaluate
    # ===============================
    print("\n[Step 6/6] Final Evaluation...")

    # Load best model
    checkpoint = torch.load(trainer.save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on validation and test sets
    val_acc, _, _, _ = trainer.evaluate(val_loader, 'Validation')
    test_acc, _, _, _ = trainer.evaluate(test_loader, 'Test')

    print(f"\n{'='*80}")
    print("Final Results")
    print(f"{'='*80}")
    print(f"Best Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train deep learning models for EMG gesture recognition')

    # Data
    parser.add_argument('--data_path', type=str, default='.',
                       help='Path to dataset')
    parser.add_argument('--max_users', type=int, default=None,
                       help='Max users to load (None = all)')

    # Model
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                       choices=['cnn_lstm', 'cnn', 'attention_lstm', 'attention_resnet18', 'transformer'],
                       help='Model type')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')

    # Other
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (use 0 for Windows)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state')

    args = parser.parse_args()

    main(args)
