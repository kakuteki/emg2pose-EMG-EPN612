"""
Trial 13: Integrated Training Script with All Optimization Techniques

This script combines all optimization techniques from Trials 1-12.
Implementation only - no training execution.
"""
import sys
from pathlib import Path
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import tsaug

sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_loader import EMGDataLoader, create_data_split
from features.feature_extractor import EMGPreprocessor
from models.cnn_lstm import get_model


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class WarmupCosineAnnealingLR:
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr


class DataAugmenter:
    """Time-series data augmentation using tsaug"""
    def __init__(self, enable=True):
        self.enable = enable
        if enable:
            # Create list of augmenters to apply sequentially with probabilities
            self.augmenters = [
                tsaug.AddNoise(scale=0.01) @ 0.5,
                tsaug.TimeWarp(n_speed_change=3, max_speed_ratio=2.0) @ 0.3,
                tsaug.Drift(max_drift=0.1, n_drift_points=3) @ 0.3,
                tsaug.Quantize(n_levels=10) @ 0.2
            ]

    def augment(self, X):
        if not self.enable:
            return X
        X_transposed = X.transpose(0, 2, 1)
        # Apply each augmenter sequentially with their probabilities
        X_aug = X_transposed
        for aug in self.augmenters:
            X_aug = aug.augment(X_aug)
        # Fix negative strides issue by making a copy
        return X_aug.transpose(0, 2, 1).copy()


def apply_resampling(X, y):
    """Apply SMOTE over-sampling and random under-sampling"""
    N, C, L = X.shape
    X_flat = X.reshape(N, C * L)

    over = SMOTE(sampling_strategy='auto', random_state=42)
    under = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    pipeline = Pipeline([('over', over), ('under', under)])

    X_resampled, y_resampled = pipeline.fit_resample(X_flat, y)
    # Fix negative strides issue by making a contiguous copy
    X_resampled = X_resampled.reshape(-1, C, L).copy()
    return X_resampled, y_resampled


class IntegratedTrainer:
    """Integrated trainer with all optimization techniques"""
    def __init__(self, model, device, train_loader, val_loader, test_loader,
                 criterion, optimizer, scheduler, num_epochs=50,
                 gradient_clip=1.0, save_dir='results/integrated',
                 exclude_pinch=False):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.gradient_clip = gradient_clip
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.save_dir / 'tensorboard')
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        self.best_val_acc = 0.0

        if exclude_pinch:
            self.class_names = ['No Gesture', 'Fist', 'Wave In', 'Wave Out', 'Open']
        else:
            self.class_names = ['No Gesture', 'Fist', 'Wave In', 'Wave Out', 'Open', 'Pinch']

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

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})

        return running_loss / len(self.train_loader), 100. * correct / total

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
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})

        return running_loss / len(self.val_loader), 100. * correct / total

    def train(self):
        """Complete training loop"""
        print(f"\n{'='*80}\nStarting Integrated Training\n{'='*80}")
        print(f"Device: {self.device}\nModel: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {self.num_epochs}\nGradient Clip: {self.gradient_clip}\n{'='*80}\n")

        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            if self.scheduler:
                lr = self.scheduler.step()
                self.learning_rates.append(lr)

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            if self.scheduler:
                self.writer.add_scalar('LearningRate', lr, epoch)

            print(f"\nEpoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if self.scheduler:
                print(f"  Learning Rate: {lr:.6f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"  [*] New best model saved! (Val Acc: {val_acc:.2f}%)")

        print(f"\n{'='*80}\nTraining Completed!\nBest Val Acc: {self.best_val_acc:.2f}%\n{'='*80}\n")
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
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\n{'='*80}\n{dataset_name} Set Evaluation\n{'='*80}")
        print(f"Accuracy: {accuracy*100:.2f}%\n")

        unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
        present_class_names = [self.class_names[i] for i in unique_labels if i < len(self.class_names)]
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, labels=unique_labels,
                                   target_names=present_class_names, zero_division=0))

        self.plot_confusion_matrix(cm, unique_labels, dataset_name)
        return accuracy, cm, all_preds, all_labels

    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.train_accs, label='Train Acc', linewidth=2)
        axes[1].plot(self.val_accs, label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        if self.learning_rates:
            axes[2].plot(self.learning_rates, linewidth=2, color='red')
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Learning Rate', fontsize=12)
            axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {self.save_dir / 'training_curves.png'}")
        plt.close()

    def plot_confusion_matrix(self, cm, labels, dataset_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        label_names = [self.class_names[i] if i < len(self.class_names) else f"Class_{i}" for i in labels]

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=label_names, yticklabels=label_names, cbar_kws={'label': 'Percentage'})

        plt.title(f'Confusion Matrix - {dataset_name} Set', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'confusion_matrix_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }
        torch.save(checkpoint, self.save_dir / filename)


def main(args):
    """Main training function"""
    print("="*80 + "\nEMG Gesture Recognition - Integrated Training (Trial 13)\n" + "="*80)
    print(f"\nConfiguration:\n  Model: {args.model_type}\n  Focal Loss: {args.use_focal_loss}")
    print(f"  Resampling: {args.use_resampling}\n  Augmentation: {args.use_augmentation}")
    print(f"  Two-Stage: {args.two_stage}\n  Exclude Pinch: {args.exclude_pinch}\n" + "="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    print("\n[Step 1/6] Loading dataset...")
    train_loader_data = EMGDataLoader(args.data_path, dataset_type='training')
    X_train_raw, y_train, _ = train_loader_data.load_dataset(max_users=args.max_users)
    test_loader_data = EMGDataLoader(args.data_path, dataset_type='testing')
    X_test_raw, y_test, _ = test_loader_data.load_dataset(max_users=args.max_users)

    # Exclude Pinch class if requested
    if args.exclude_pinch:
        print("\nExcluding Pinch class (label 5)...")
        train_mask = y_train != 5
        test_mask = y_test != 5
        X_train_raw = X_train_raw[train_mask]
        y_train = y_train[train_mask]
        X_test_raw = X_test_raw[test_mask]
        y_test = y_test[test_mask]
        print(f"After exclusion - Train: {X_train_raw.shape[0]} samples, Test: {X_test_raw.shape[0]} samples")

    # Preprocess
    print("\n[Step 2/6] Preprocessing...")
    preprocessor = EMGPreprocessor(sampling_rate=200)
    X_train_preprocessed = preprocessor.preprocess(X_train_raw, apply_bandpass=True, apply_notch=True, normalize=True)
    X_test_preprocessed = preprocessor.preprocess(X_test_raw, apply_bandpass=True, apply_notch=True, normalize=True)
    X_train, X_val, y_train_split, y_val = create_data_split(X_train_preprocessed, y_train, test_size=args.val_split, random_state=args.random_state)

    # Create model
    print("\n[Step 3/6] Creating model...")
    num_classes = 5 if args.exclude_pinch else 6
    model = get_model(model_type=args.model_type, input_channels=8, num_classes=num_classes, dropout=args.dropout).to(device)
    print(f"Model: {args.model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Apply Resampling (SMOTE + Under-sampling)
    if args.use_resampling:
        print("\n[Step 4/6] Applying SMOTE + Under-sampling...")
        X_train, y_train_split = apply_resampling(X_train, y_train_split)
        print(f"Resampled train data shape: {X_train.shape}")

    # Data Augmentation
    augmenter = DataAugmenter(enable=args.use_augmentation)
    if args.use_augmentation:
        print("\n[Step 5/6] Data augmentation enabled")

    # Create data loaders
    print("\n[Step 6/6] Creating data loaders...")
    # Fix negative strides issue by ensuring contiguous arrays
    train_dataset = TensorDataset(torch.FloatTensor(X_train.copy()), torch.LongTensor(y_train_split))
    val_dataset = TensorDataset(torch.FloatTensor(X_val.copy()), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_preprocessed.copy()), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Setup criterion (Focal Loss or CrossEntropy)
    if args.use_focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma)
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropy Loss")

    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup learning rate scheduler
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs,
                                       max_epochs=args.epochs, eta_min=1e-6)

    # Setup trainer
    save_dir = args.save_dir if hasattr(args, 'save_dir') else f'results/{args.model_type}'
    trainer = IntegratedTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        gradient_clip=args.gradient_clip,
        save_dir=save_dir,
        exclude_pinch=args.exclude_pinch
    )

    # Train the model
    trainer.train()

    # Load best model and evaluate on test set
    best_checkpoint = torch.load(trainer.save_dir / 'best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {best_checkpoint['epoch']+1}")

    # Evaluate on validation and test sets
    print("\n" + "="*80)
    val_acc, val_cm, _, _ = trainer.evaluate(val_loader, 'Validation')
    test_acc, test_cm, _, _ = trainer.evaluate(test_loader, 'Test')
    print("="*80)

    # Save final results
    results_summary = {
        'model_type': args.model_type,
        'best_val_acc': float(trainer.best_val_acc),
        'test_acc': float(test_acc * 100),
        'num_classes': num_classes,
        'exclude_pinch': args.exclude_pinch,
        'use_focal_loss': args.use_focal_loss,
        'use_resampling': args.use_resampling,
        'use_augmentation': args.use_augmentation,
        'two_stage': args.two_stage
    }

    import json
    with open(trainer.save_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults summary saved to: {trainer.save_dir / 'results_summary.json'}")
    print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integrated training for EMG gesture recognition')

    parser.add_argument('--data_path', type=str, default='.', help='Dataset path')
    parser.add_argument('--max_users', type=int, default=None, help='Max users')
    parser.add_argument('--exclude_pinch', action='store_true', help='Exclude Pinch class')
    parser.add_argument('--model_type', type=str, default='waveformer_complete',
                       choices=['cnn_lstm', 'cnn', 'attention_lstm', 'attention_resnet18',
                               'transformer', 'wavenet', 'waveformer', 'waveformer_complete'],
                       help='Model type')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
    parser.add_argument('--use_focal_loss', type=bool, default=True, help='Use Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal gamma')
    parser.add_argument('--use_resampling', type=bool, default=True, help='SMOTE + Under-sampling')
    parser.add_argument('--use_augmentation', type=bool, default=True, help='Data augmentation')
    parser.add_argument('--two_stage', type=bool, default=True, help='Two-stage training')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--val_split', type=float, default=0.2, help='Val split')
    parser.add_argument('--num_workers', type=int, default=0, help='Workers')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--save_dir', type=str, default='results/integrated', help='Save directory')

    args = parser.parse_args()
    main(args)
