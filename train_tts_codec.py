"""
Train TTS-codec-inspired models for EMG gesture classification

TTSコーデックの概念をEMG分類に適用した実験
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')

from data.data_loader import EMGDataLoader, create_data_split
from models.emg_codec import get_model


def train_epoch(model, dataloader, criterion, optimizer, device, model_type):
    """1エポックの訓練"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_vq_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in tqdm(dataloader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # Forward pass
        if model_type in ['codec', 'tts_style']:
            logits, vq_loss = model(batch_x)
            cls_loss = criterion(logits, batch_y)
            loss = cls_loss + 0.1 * vq_loss  # VQ lossの重み
            total_vq_loss += vq_loss.item()
        else:  # multiscale
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            cls_loss = loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # メトリクス計算
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_vq_loss = total_vq_loss / len(dataloader) if model_type in ['codec', 'tts_style'] else 0.0
    accuracy = accuracy_score(all_labels, all_preds) * 100

    return avg_loss, avg_cls_loss, avg_vq_loss, accuracy


def evaluate(model, dataloader, criterion, device, model_type):
    """評価"""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_vq_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            if model_type in ['codec', 'tts_style']:
                logits, vq_loss = model(batch_x)
                cls_loss = criterion(logits, batch_y)
                loss = cls_loss + 0.1 * vq_loss
                total_vq_loss += vq_loss.item()
            else:  # multiscale
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                cls_loss = loss

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_vq_loss = total_vq_loss / len(dataloader) if model_type in ['codec', 'tts_style'] else 0.0
    accuracy = accuracy_score(all_labels, all_preds) * 100

    return avg_loss, avg_cls_loss, avg_vq_loss, accuracy, all_preds, all_labels


def plot_confusion_matrix(cm, classes, save_path):
    """混同行列をプロット"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_history(history, save_path):
    """訓練履歴をプロット"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Classification Loss
    axes[0, 1].plot(history['train_cls_loss'], label='Train')
    axes[0, 1].plot(history['val_cls_loss'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Classification Loss')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Accuracy
    axes[1, 0].plot(history['train_acc'], label='Train')
    axes[1, 0].plot(history['val_acc'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # VQ Loss
    if 'train_vq_loss' in history and len(history['train_vq_loss']) > 0:
        axes[1, 1].plot(history['train_vq_loss'], label='Train')
        axes[1, 1].plot(history['val_vq_loss'], label='Validation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('VQ Loss')
        axes[1, 1].set_title('Vector Quantization Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_model(model, train_loader, val_loader, test_loader, args, device):
    """モデル訓練のメインループ"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [], 'test_cls_loss': [],
        'train_vq_loss': [], 'val_vq_loss': [], 'test_vq_loss': [],
        'train_acc': [], 'val_acc': [], 'test_acc': []
    }

    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0

    print("\n" + "="*80)
    print(f"Training {args.model_type} model")
    print("="*80)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_cls_loss, train_vq_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.model_type
        )

        # Validate
        val_loss, val_cls_loss, val_vq_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, args.model_type
        )

        # Test
        test_loss, test_cls_loss, test_vq_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device, args.model_type
        )

        # 履歴記録
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['test_loss'].append(test_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['test_cls_loss'].append(test_cls_loss)
        history['train_vq_loss'].append(train_vq_loss)
        history['val_vq_loss'].append(val_vq_loss)
        history['test_vq_loss'].append(test_vq_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        # Learning rate scheduling
        scheduler.step(val_acc)

        # ログ出力
        print(f"  Train - Loss: {train_loss:.4f}, Cls Loss: {train_cls_loss:.4f}, "
              f"VQ Loss: {train_vq_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Cls Loss: {val_cls_loss:.4f}, "
              f"VQ Loss: {val_vq_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Cls Loss: {test_cls_loss:.4f}, "
              f"VQ Loss: {test_vq_loss:.4f}, Acc: {test_acc:.2f}%")

        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0

            # Save best model
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
            }, os.path.join(args.save_dir, f'best_model_{args.model_type}.pt'))
            print(f"  ✓ Best model saved! (Val: {val_acc:.2f}%, Test: {test_acc:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    return history, best_val_acc, best_test_acc


def main():
    parser = argparse.ArgumentParser(description='Train TTS-codec-inspired EMG models')
    parser.add_argument('--model_type', type=str, default='codec',
                        choices=['codec', 'tts_style', 'multiscale'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension')
    parser.add_argument('--num_embeddings', type=int, default=512, help='Number of VQ embeddings')
    parser.add_argument('--exclude_pinch', action='store_true', help='Exclude Pinch class')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--save_dir', type=str, default='results/tts_codec', help='Save directory')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # データロード
    print("\n" + "="*80)
    print("Loading EMG Dataset")
    print("="*80)

    train_loader_data = EMGDataLoader('.', dataset_type='training')
    test_loader_data = EMGDataLoader('.', dataset_type='testing')

    X_train, y_train, _ = train_loader_data.load_dataset()
    X_test, y_test, _ = test_loader_data.load_dataset()

    # Pinch除外
    if args.exclude_pinch:
        print("\nExcluding Pinch class (label 5)...")
        train_mask = y_train != 5
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        test_mask = y_test != 5
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

    print(f"Train: {X_train.shape}")
    print(f"Test:  {X_test.shape}")

    # Train/Val分割
    X_train_split, X_val, y_train_split, y_val = create_data_split(
        X_train, y_train, test_size=0.2, random_state=args.random_state
    )

    print(f"\nAfter split:")
    print(f"  Train: {X_train_split.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # クラス数
    num_classes = len(np.unique(y_train_split))
    print(f"\nNumber of classes: {num_classes}")

    # PyTorch Datasetに変換
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_split),
        torch.LongTensor(y_train_split)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # モデル構築
    print("\n" + "="*80)
    print(f"Building {args.model_type} model")
    print("="*80)

    if args.model_type == 'codec':
        model = get_model(
            model_type='codec',
            in_channels=8,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_embeddings=args.num_embeddings,
            num_classes=num_classes
        )
    elif args.model_type == 'tts_style':
        model = get_model(
            model_type='tts_style',
            in_channels=8,
            hidden_dim=args.hidden_dim * 2,  # TTSモデルは大きめ
            latent_dim=args.latent_dim * 2,
            num_embeddings=args.num_embeddings,
            num_classes=num_classes
        )
    else:  # multiscale
        model = get_model(
            model_type='multiscale',
            in_channels=8,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes
        )

    model = model.to(device)

    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 訓練
    history, best_val_acc, best_test_acc = train_model(
        model, train_loader, val_loader, test_loader, args, device
    )

    # 最終評価
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)

    # Best modelをロード
    checkpoint = torch.load(os.path.join(args.save_dir, f'best_model_{args.model_type}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    _, _, _, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device, args.model_type
    )

    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")

    # Classification report
    class_names = ['No Gesture', 'Fist', 'Wave In', 'Wave Out', 'Open']
    if args.exclude_pinch:
        # Pinch除外
        class_names = class_names[:5]

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    cm_path = os.path.join(args.save_dir, f'confusion_matrix_{args.model_type}.png')
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")

    # Training history
    history_path = os.path.join(args.save_dir, f'training_history_{args.model_type}.png')
    plot_training_history(history, history_path)
    print(f"Training history saved to: {history_path}")

    # 結果をJSON保存
    results = {
        'model_type': args.model_type,
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'args': vars(args),
        'history': history
    }

    results_path = os.path.join(args.save_dir, f'results_{args.model_type}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
