"""
Quick evaluation script for Trial 13
Loads the trained model and evaluates on test set
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import the necessary classes
from data.data_loader import EMGDataLoader
from features.feature_extractor import EMGPreprocessor
from models.cnn_lstm import get_model

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    return accuracy, cm, report, all_preds, all_labels


def main():
    print("="*80)
    print("Trial 13 - Test Set Evaluation")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load test data
    print("\n[1/4] Loading test dataset...")
    test_loader_data = EMGDataLoader('.', dataset_type='testing')
    X_test_raw, y_test, _ = test_loader_data.load_dataset(max_users=None)

    # Exclude Pinch class (same as training - label 5)
    print("\nExcluding Pinch class (label 5)...")
    test_mask = y_test != 5
    X_test_raw = X_test_raw[test_mask]
    y_test = y_test[test_mask]

    # Define class names (5 classes after excluding Pinch)
    class_names = ['No Gesture', 'Fist', 'Wave In', 'Wave Out', 'Open']

    print(f"Test samples: {len(X_test_raw)}")
    print(f"Classes: {class_names}")

    # Preprocess
    print("\n[2/4] Preprocessing...")
    preprocessor = EMGPreprocessor(sampling_rate=200)
    X_test_preprocessed = preprocessor.preprocess(X_test_raw, apply_bandpass=True, apply_notch=True, normalize=True)

    # Create test loader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_preprocessed.copy()),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Load model
    print("\n[3/4] Loading trained model...")
    num_classes = len(class_names)
    model = get_model(
        model_type='waveformer_complete',
        input_channels=8,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)

    # Load best checkpoint
    checkpoint_path = Path('results/integrated/best_model.pth')
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    print(f"Best validation accuracy: {checkpoint.get('val_acc', 'N/A')}")

    # Evaluate
    print("\n[4/4] Evaluating on test set...")
    print("-"*80)
    test_acc, test_cm, test_report, _, _ = evaluate_model(model, test_loader, device, class_names)

    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(test_cm)
    print(f"\nClassification Report:")
    print(test_report)

    # Save results
    results = {
        'test_accuracy': float(test_acc * 100),
        'best_val_accuracy': float(checkpoint.get('val_acc', 0)),
        'confusion_matrix': test_cm.tolist(),
        'class_names': class_names,
        'model_type': 'waveformer_complete',
        'epoch': int(checkpoint['epoch'] + 1)
    }

    results_path = Path('results/integrated/test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("="*80)

if __name__ == "__main__":
    main()
