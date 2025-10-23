"""
Two-Stage Training Strategy for EMG gesture recognition
Stage 1: Pre-training on balanced data (with SMOTE)
Stage 2: Fine-tuning on all data
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
from imblearn.over_sampling import SMOTE

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_loader import EMGDataLoader, create_data_split
from data.pytorch_dataset import create_dataloaders
from features.feature_extractor import EMGPreprocessor
from models.cnn_lstm import get_model


class TwoStageTrainer:
    """Two-Stage Training: Stage 1 (balanced data) -> Stage 2 (all data)"""

    def __init__(self, model, device, criterion, num_classes=6,
                 save_dir='results/two_stage_training'):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.save_dir / 'tensorboard')

        self.class_names = ['No Gesture', 'Fist', 'Wave In',
                           'Wave Out', 'Open', 'Pinch']

        # Stage 1 metrics
        self.stage1_train_losses = []
        self.stage1_val_losses = []
        self.stage1_train_accs = []
        self.stage1_val_accs = []
        self.stage1_best_val_acc = 0.0

        # Stage 2 metrics
        self.stage2_train_losses = []
        self.stage2_val_losses = []
        self.stage2_train_accs = []
        self.stage2_val_accs = []
        self.stage2_best_val_acc = 0.0

    def train_epoch(self, epoch, train_loader, optimizer, stage='Stage 1', total_epochs=30):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'{stage} - Epoch {epoch+1}/{total_epochs} [Train]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': running_loss / (pbar.n + 1),
                            'acc': 100. * correct / total})

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch, val_loader, stage='Stage 1', total_epochs=30):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'{stage} - Epoch {epoch+1}/{total_epochs} [Val]')
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

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def stage1_train(self, train_loader, val_loader, optimizer, scheduler, num_epochs):
        """Stage 1: Pre-training on balanced data (SMOTE applied)"""
        print(f"\n{'='*80}")
        print("STAGE 1: Pre-training on Balanced Data (SMOTE)")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*80}\n")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                epoch, train_loader, optimizer, 'Stage 1', num_epochs
            )
            self.stage1_train_losses.append(train_loss)
            self.stage1_train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(
                epoch, val_loader, 'Stage 1', num_epochs
            )
            self.stage1_val_losses.append(val_loss)
            self.stage1_val_accs.append(val_acc)

            # Scheduler step
            if scheduler:
                scheduler.step(val_loss)

            # Log to tensorboard
            self.writer.add_scalar('Stage1/Loss/train', train_loss, epoch)
            self.writer.add_scalar('Stage1/Loss/val', val_loss, epoch)
            self.writer.add_scalar('Stage1/Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Stage1/Accuracy/val', val_acc, epoch)

            # Print epoch summary
            print(f"\nStage 1 - Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > self.stage1_best_val_acc:
                self.stage1_best_val_acc = val_acc
                self.save_checkpoint(epoch, 'stage1_best_model.pth', stage=1)
                print(f"  [*] New best Stage 1 model saved! (Val Acc: {val_acc:.2f}%)")

        print(f"\n{'='*80}")
        print("Stage 1 Training Completed!")
        print(f"Best Stage 1 Validation Accuracy: {self.stage1_best_val_acc:.2f}%")
        print(f"{'='*80}\n")

    def stage2_train(self, train_loader, val_loader, optimizer, scheduler, num_epochs):
        """Stage 2: Fine-tuning on all data"""
        print(f"\n{'='*80}")
        print("STAGE 2: Fine-tuning on All Data")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*80}\n")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                epoch, train_loader, optimizer, 'Stage 2', num_epochs
            )
            self.stage2_train_losses.append(train_loss)
            self.stage2_train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(
                epoch, val_loader, 'Stage 2', num_epochs
            )
            self.stage2_val_losses.append(val_loss)
            self.stage2_val_accs.append(val_acc)

            # Scheduler step
            if scheduler:
                scheduler.step(val_loss)

            # Log to tensorboard
            self.writer.add_scalar('Stage2/Loss/train', train_loss, epoch)
            self.writer.add_scalar('Stage2/Loss/val', val_loss, epoch)
            self.writer.add_scalar('Stage2/Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Stage2/Accuracy/val', val_acc, epoch)

            # Print epoch summary
            print(f"\nStage 2 - Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > self.stage2_best_val_acc:
                self.stage2_best_val_acc = val_acc
                self.save_checkpoint(epoch, 'stage2_best_model.pth', stage=2)
                print(f"  [*] New best Stage 2 model saved! (Val Acc: {val_acc:.2f}%)")

        print(f"\n{'='*80}")
        print("Stage 2 Training Completed!")
        print(f"Best Stage 2 Validation Accuracy: {self.stage2_best_val_acc:.2f}%")
        print(f"{'='*80}\n")

        # Plot combined training curves
        self.plot_combined_training_curves()

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

    def plot_combined_training_curves(self):
        """Plot training and validation curves for both stages"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Combine stage 1 and stage 2 data
        all_train_losses = self.stage1_train_losses + self.stage2_train_losses
        all_val_losses = self.stage1_val_losses + self.stage2_val_losses
        all_train_accs = self.stage1_train_accs + self.stage2_train_accs
        all_val_accs = self.stage1_val_accs + self.stage2_val_accs

        stage1_epochs = len(self.stage1_train_losses)
        total_epochs = len(all_train_losses)

        # Loss curves
        ax1.plot(all_train_losses, label='Train Loss', linewidth=2)
        ax1.plot(all_val_losses, label='Val Loss', linewidth=2)
        ax1.axvline(x=stage1_epochs-1, color='r', linestyle='--', label='Stage 1 -> Stage 2')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Two-Stage Training: Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2.plot(all_train_accs, label='Train Acc', linewidth=2)
        ax2.plot(all_val_accs, label='Val Acc', linewidth=2)
        ax2.axvline(x=stage1_epochs-1, color='r', linestyle='--', label='Stage 1 -> Stage 2')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Two-Stage Training: Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'two_stage_training_curves.png', dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {self.save_dir / 'two_stage_training_curves.png'}")
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

        plt.title(f'Confusion Matrix - {dataset_name} Set (Two-Stage)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(self.save_dir / f'confusion_matrix_{dataset_name.lower()}.png',
                   dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {self.save_dir / f'confusion_matrix_{dataset_name.lower()}.png'}")
        plt.close()

    def save_checkpoint(self, epoch, filename, stage=1):
        """Save model checkpoint"""
        if stage == 1:
            checkpoint = {
                'epoch': epoch,
                'stage': 1,
                'model_state_dict': self.model.state_dict(),
                'best_val_acc': self.stage1_best_val_acc,
                'train_losses': self.stage1_train_losses,
                'val_losses': self.stage1_val_losses,
                'train_accs': self.stage1_train_accs,
                'val_accs': self.stage1_val_accs,
            }
        else:
            checkpoint = {
                'epoch': epoch,
                'stage': 2,
                'model_state_dict': self.model.state_dict(),
                'best_val_acc': self.stage2_best_val_acc,
                'stage1_best_val_acc': self.stage1_best_val_acc,
                'train_losses': self.stage2_train_losses,
                'val_losses': self.stage2_val_losses,
                'train_accs': self.stage2_train_accs,
                'val_accs': self.stage2_val_accs,
            }
        torch.save(checkpoint, self.save_dir / filename)


def apply_smote(X, y, random_state=42):
    """Apply SMOTE to balance the dataset"""
    print("\n[SMOTE] Applying SMOTE to balance dataset...")

    # Original distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Original distribution: {dict(zip(unique, counts))}")

    # Reshape for SMOTE: (n_samples, n_features)
    n_samples, n_channels, n_timesteps = X.shape
    X_reshaped = X.reshape(n_samples, -1)

    # Apply SMOTE
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)

    # Reshape back: (n_samples, n_channels, n_timesteps)
    X_resampled = X_resampled.reshape(-1, n_channels, n_timesteps)

    # New distribution
    unique, counts = np.unique(y_resampled, return_counts=True)
    print(f"Resampled distribution: {dict(zip(unique, counts))}")
    print(f"New dataset size: {X_resampled.shape}")

    return X_resampled, y_resampled


def main(args):
    """Main training function"""
    print("="*80)
    print("EMG Gesture Recognition - Two-Stage Training Strategy")
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
    print("\n[Step 1/7] Loading dataset...")
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
    print("\n[Step 2/7] Preprocessing...")
    preprocessor = EMGPreprocessor(sampling_rate=200)

    X_train_preprocessed = preprocessor.preprocess(X_train_raw, apply_bandpass=True,
                                                   apply_notch=True, normalize=True)
    X_test_preprocessed = preprocessor.preprocess(X_test_raw, apply_bandpass=True,
                                                  apply_notch=True, normalize=True)

    # Train/Val split for original data (Stage 2)
    X_train_orig, X_val_orig, y_train_orig, y_val_orig = create_data_split(
        X_train_preprocessed, y_train, test_size=args.val_split, random_state=args.random_state
    )

    print(f"Original Train: {X_train_orig.shape}, Val: {X_val_orig.shape}")

    # ===============================
    # 3. Stage 1: Apply SMOTE
    # ===============================
    print("\n[Step 3/7] Preparing Stage 1 data (with SMOTE)...")
    X_train_balanced, y_train_balanced = apply_smote(
        X_train_orig, y_train_orig, random_state=args.random_state
    )

    # Create dataloaders for Stage 1 (balanced data)
    print("\n[Step 4/7] Creating Stage 1 DataLoaders...")
    stage1_train_loader, stage1_val_loader, _ = create_dataloaders(
        X_train_balanced, y_train_balanced,
        X_val_orig, y_val_orig,
        X_test_preprocessed, y_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create dataloaders for Stage 2 (all data)
    print("\n[Step 5/7] Creating Stage 2 DataLoaders...")
    stage2_train_loader, stage2_val_loader, test_loader = create_dataloaders(
        X_train_orig, y_train_orig,
        X_val_orig, y_val_orig,
        X_test_preprocessed, y_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # ===============================
    # 6. Create model
    # ===============================
    print("\n[Step 6/7] Creating model...")
    model = get_model(
        model_type=args.model_type,
        input_channels=8,
        num_classes=6,
        dropout=args.dropout
    ).to(device)

    # Class weights for balanced data (Stage 1)
    class_counts_balanced = np.bincount(y_train_balanced)
    class_weights_balanced = 1.0 / (class_counts_balanced + 1)
    class_weights_balanced = class_weights_balanced / class_weights_balanced.sum() * len(class_weights_balanced)
    class_weights_balanced = torch.FloatTensor(class_weights_balanced).to(device)

    # Class weights for original data (Stage 2)
    class_counts_orig = np.bincount(y_train_orig)
    class_weights_orig = 1.0 / (class_counts_orig + 1)
    class_weights_orig = class_weights_orig / class_weights_orig.sum() * len(class_weights_orig)
    class_weights_orig = torch.FloatTensor(class_weights_orig).to(device)

    criterion_stage1 = nn.CrossEntropyLoss(weight=class_weights_balanced)
    criterion_stage2 = nn.CrossEntropyLoss(weight=class_weights_orig)

    # ===============================
    # 7. Two-Stage Training
    # ===============================
    print("\n[Step 7/7] Starting Two-Stage Training...")

    trainer = TwoStageTrainer(
        model=model,
        device=device,
        criterion=criterion_stage1,  # Will be updated for Stage 2
        num_classes=6,
        save_dir=f'results/two_stage_{args.model_type}'
    )

    # ========== STAGE 1 ==========
    optimizer_stage1 = optim.Adam(model.parameters(), lr=args.stage1_lr, weight_decay=args.weight_decay)
    scheduler_stage1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage1, mode='min', patience=5, factor=0.5)

    trainer.stage1_train(
        train_loader=stage1_train_loader,
        val_loader=stage1_val_loader,
        optimizer=optimizer_stage1,
        scheduler=scheduler_stage1,
        num_epochs=args.stage1_epochs
    )

    # Load best Stage 1 model
    print("\n[Loading] Loading best Stage 1 model for Stage 2...")
    checkpoint = torch.load(trainer.save_dir / 'stage1_best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded Stage 1 model with Val Acc: {checkpoint['best_val_acc']:.2f}%")

    # ========== STAGE 2 ==========
    # Update criterion for Stage 2
    trainer.criterion = criterion_stage2

    optimizer_stage2 = optim.Adam(model.parameters(), lr=args.stage2_lr, weight_decay=args.weight_decay)
    scheduler_stage2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage2, mode='min', patience=5, factor=0.5)

    trainer.stage2_train(
        train_loader=stage2_train_loader,
        val_loader=stage2_val_loader,
        optimizer=optimizer_stage2,
        scheduler=scheduler_stage2,
        num_epochs=args.stage2_epochs
    )

    # ===============================
    # 8. Final Evaluation
    # ===============================
    print("\n[Step 8/8] Final Evaluation...")

    # Load best Stage 2 model
    checkpoint = torch.load(trainer.save_dir / 'stage2_best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on validation and test sets
    val_acc, _, _, _ = trainer.evaluate(stage2_val_loader, 'Validation')
    test_acc, _, _, _ = trainer.evaluate(test_loader, 'Test')

    print(f"\n{'='*80}")
    print("Two-Stage Training - Final Results")
    print(f"{'='*80}")
    print(f"Stage 1 Best Validation Accuracy: {trainer.stage1_best_val_acc:.2f}%")
    print(f"Stage 2 Best Validation Accuracy: {trainer.stage2_best_val_acc:.2f}%")
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Two-Stage Training for EMG gesture recognition')

    # Data
    parser.add_argument('--data_path', type=str, default='.',
                       help='Path to dataset')
    parser.add_argument('--max_users', type=int, default=None,
                       help='Max users to load (None = all)')

    # Model
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                       choices=['cnn_lstm', 'cnn', 'attention_lstm', 'attention_resnet18',
                               'transformer', 'wavenet', 'waveformer', 'waveformer_complete'],
                       help='Model type')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')

    # Stage 1 Training (Balanced Data)
    parser.add_argument('--stage1_epochs', type=int, default=30,
                       help='Number of epochs for Stage 1')
    parser.add_argument('--stage1_lr', type=float, default=1e-3,
                       help='Learning rate for Stage 1 (high for fast learning)')

    # Stage 2 Training (All Data)
    parser.add_argument('--stage2_epochs', type=int, default=20,
                       help='Number of epochs for Stage 2')
    parser.add_argument('--stage2_lr', type=float, default=1e-4,
                       help='Learning rate for Stage 2 (low for fine-tuning)')

    # General Training
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
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
