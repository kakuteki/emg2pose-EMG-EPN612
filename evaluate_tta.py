"""
Test-Time Augmentation (TTA) Evaluation Script
Applies data augmentation at test time and averages predictions
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset

from models.waveformer import WaveFormerComplete
from utils.data_preprocessing import load_emg_data, EMGPreprocessor


def apply_tta_augmentations(X, num_augs=5):
    """
    Apply test-time augmentations

    Args:
        X: Input data (N, channels, length)
        num_augs: Number of augmentation versions to create

    Returns:
        List of augmented versions of X
    """
    augmented_versions = [X]  # Original version

    for i in range(num_augs - 1):
        X_aug = X.copy()

        # Augmentation 1: Add Gaussian noise
        if i % 3 == 0:
            noise = np.random.normal(0, 0.01, X_aug.shape)
            X_aug = X_aug + noise

        # Augmentation 2: Scale amplitude
        elif i % 3 == 1:
            scale = np.random.uniform(0.95, 1.05)
            X_aug = X_aug * scale

        # Augmentation 3: Time shift
        elif i % 3 == 2:
            shift = np.random.randint(-5, 5)
            if shift != 0:
                X_aug = np.roll(X_aug, shift, axis=2)

        augmented_versions.append(X_aug)

    return augmented_versions


def evaluate_with_tta(model_path, data_path, num_augs=5, exclude_pinch=True):
    """
    Evaluate model with Test-Time Augmentation

    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to EMG dataset
        num_augs: Number of augmentations to apply
        exclude_pinch: Whether to exclude Pinch class
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Number of TTA augmentations: {num_augs}")

    # Load test data
    print("\nLoading test data...")
    X_train, y_train, X_test, y_test, label_mapping = load_emg_data(
        data_path=data_path,
        test_users=list(range(307, 613))
    )

    print(f"Test data shape: {X_test.shape}")

    # Preprocess data
    preprocessor = EMGPreprocessor()
    X_test_preprocessed = preprocessor.fit_transform(X_test)

    # Handle Pinch exclusion
    if exclude_pinch:
        print("\nExcluding Pinch class...")
        mask = y_test != 5
        X_test_preprocessed = X_test_preprocessed[mask]
        y_test = y_test[mask]
        # Remap labels
        y_test[y_test > 5] -= 1
        num_classes = 5
    else:
        num_classes = 6

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = WaveFormerComplete(num_classes=num_classes, dropout=0.3).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate without TTA (baseline)
    print("\nEvaluating without TTA (baseline)...")
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_preprocessed.copy()),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds_baseline = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            all_preds_baseline.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds_baseline = np.array(all_preds_baseline)
    all_labels = np.array(all_labels)

    baseline_acc = accuracy_score(all_labels, all_preds_baseline) * 100
    print(f"Baseline accuracy (no TTA): {baseline_acc:.2f}%")

    # Evaluate with TTA
    print(f"\nEvaluating with TTA ({num_augs} augmentations)...")

    # Apply TTA augmentations
    print("Generating augmented versions...")
    X_augmented_list = apply_tta_augmentations(X_test_preprocessed, num_augs=num_augs)

    # Collect predictions from all augmented versions
    all_probs_tta = []

    for aug_idx, X_aug in enumerate(X_augmented_list):
        print(f"Processing augmentation {aug_idx+1}/{num_augs}...")

        aug_dataset = TensorDataset(
            torch.FloatTensor(X_aug.copy()),
            torch.LongTensor(y_test)
        )
        aug_loader = DataLoader(aug_dataset, batch_size=64, shuffle=False)

        aug_probs = []

        with torch.no_grad():
            for batch_X, batch_y in aug_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                aug_probs.append(probs.cpu().numpy())

        aug_probs = np.vstack(aug_probs)
        all_probs_tta.append(aug_probs)

    # Average predictions across all augmentations
    all_probs_tta = np.array(all_probs_tta)  # Shape: (num_augs, num_samples, num_classes)
    avg_probs = np.mean(all_probs_tta, axis=0)
    tta_preds = np.argmax(avg_probs, axis=1)

    # Calculate TTA accuracy
    tta_acc = accuracy_score(all_labels, tta_preds) * 100

    print(f"\n{'='*80}")
    print(f"Baseline Accuracy (no TTA): {baseline_acc:.2f}%")
    print(f"TTA Accuracy ({num_augs} augs): {tta_acc:.2f}%")
    print(f"Improvement: {tta_acc - baseline_acc:+.2f}%")
    print(f"{'='*80}")

    # Confusion matrix for TTA
    cm = confusion_matrix(all_labels, tta_preds)
    print("\nConfusion Matrix (TTA):")
    print(cm)

    # Classification report
    print("\nClassification Report (TTA):")
    target_names = ['No Gesture', 'Wave In', 'Wave Out', 'Fist', 'Open'] if exclude_pinch else \
                   ['No Gesture', 'Wave In', 'Wave Out', 'Fist', 'Open', 'Pinch']
    print(classification_report(all_labels, tta_preds, target_names=target_names))

    # Save results
    results = {
        'model_path': str(model_path),
        'num_augs': num_augs,
        'baseline_accuracy': float(baseline_acc),
        'tta_accuracy': float(tta_acc),
        'improvement': float(tta_acc - baseline_acc),
        'num_classes': num_classes,
        'exclude_pinch': exclude_pinch,
        'confusion_matrix': cm.tolist()
    }

    output_file = 'tta_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return tta_acc, baseline_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model with Test-Time Augmentation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint (best_model.pth)')
    parser.add_argument('--data_path', type=str, default='.',
                       help='Path to EMG dataset')
    parser.add_argument('--num_augs', type=int, default=5,
                       help='Number of augmentations to apply at test time')
    parser.add_argument('--exclude_pinch', action='store_true',
                       help='Exclude Pinch class')

    args = parser.parse_args()

    evaluate_with_tta(
        model_path=args.model_path,
        data_path=args.data_path,
        num_augs=args.num_augs,
        exclude_pinch=args.exclude_pinch
    )
