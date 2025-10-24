"""
Ensemble Evaluation Script
Combines predictions from multiple trained models for improved accuracy
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset

from src.models.waveformer import WaveFormerComplete
from src.data.data_preprocessing import load_emg_data, EMGPreprocessor


def load_model(model_path, num_classes, device):
    """Load a trained model from checkpoint"""
    model = WaveFormerComplete(num_classes=num_classes, dropout=0.3).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def ensemble_predict(models, data_loader, device, method='average'):
    """
    Make ensemble predictions

    Args:
        models: List of trained models
        data_loader: DataLoader for test data
        device: torch device
        method: 'average' for probability averaging, 'voting' for majority voting

    Returns:
        predictions: numpy array of predicted labels
        true_labels: numpy array of true labels
    """
    all_probs = []
    true_labels = []

    # Get predictions from each model
    for model in models:
        model.eval()
        model_probs = []

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                model_probs.append(probs.cpu().numpy())

                if len(true_labels) == 0 or len(true_labels) < len(batch_y):
                    true_labels.extend(batch_y.numpy())

        model_probs = np.vstack(model_probs)
        all_probs.append(model_probs)

    # Combine predictions
    all_probs = np.array(all_probs)  # Shape: (num_models, num_samples, num_classes)

    if method == 'average':
        # Average probabilities across models
        avg_probs = np.mean(all_probs, axis=0)
        predictions = np.argmax(avg_probs, axis=1)
    elif method == 'voting':
        # Majority voting
        model_predictions = np.argmax(all_probs, axis=2)  # Shape: (num_models, num_samples)
        predictions = []
        for i in range(model_predictions.shape[1]):
            votes = model_predictions[:, i]
            prediction = np.bincount(votes).argmax()
            predictions.append(prediction)
        predictions = np.array(predictions)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    true_labels = np.array(true_labels)

    return predictions, true_labels


def evaluate_ensemble(model_dirs, data_path, method='average', exclude_pinch=True):
    """
    Evaluate ensemble of models

    Args:
        model_dirs: List of directories containing trained models
        data_path: Path to EMG dataset
        method: Ensemble method ('average' or 'voting')
        exclude_pinch: Whether to exclude Pinch class
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Ensemble method: {method}")
    print(f"Number of models: {len(model_dirs)}")

    # Load test data
    print("\nLoading test data...")
    X_train, y_train, X_test, y_test, label_mapping = load_emg_data(
        data_path=data_path,
        test_users=list(range(307, 613))
    )

    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

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

    # Create data loader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_preprocessed.copy()),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load all models
    print("\nLoading models...")
    models = []
    for i, model_dir in enumerate(model_dirs):
        model_path = Path(model_dir) / 'best_model.pth'
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}, skipping...")
            continue

        print(f"Loading model {i+1}/{len(model_dirs)}: {model_dir}")
        model = load_model(model_path, num_classes, device)
        models.append(model)

    if len(models) == 0:
        print("Error: No models loaded!")
        return

    print(f"\nSuccessfully loaded {len(models)} models")

    # Make ensemble predictions
    print("\nMaking ensemble predictions...")
    predictions, true_labels = ensemble_predict(models, test_loader, device, method=method)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions) * 100
    print(f"\n{'='*80}")
    print(f"Ensemble Test Accuracy ({method}): {accuracy:.2f}%")
    print(f"{'='*80}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    target_names = ['No Gesture', 'Wave In', 'Wave Out', 'Fist', 'Open'] if exclude_pinch else \
                   ['No Gesture', 'Wave In', 'Wave Out', 'Fist', 'Open', 'Pinch']
    print(classification_report(true_labels, predictions, target_names=target_names))

    # Individual model accuracies
    print("\nIndividual Model Accuracies:")
    for i, model_dir in enumerate(model_dirs):
        if i < len(models):
            model = models[i]
            model_preds, _ = ensemble_predict([model], test_loader, device, method='average')
            model_acc = accuracy_score(true_labels, model_preds) * 100
            print(f"  Model {i+1} ({Path(model_dir).name}): {model_acc:.2f}%")

    # Save results
    results = {
        'ensemble_method': method,
        'num_models': len(models),
        'model_dirs': [str(d) for d in model_dirs],
        'test_accuracy': float(accuracy),
        'num_classes': num_classes,
        'exclude_pinch': exclude_pinch,
        'confusion_matrix': cm.tolist()
    }

    output_file = f'ensemble_results_{method}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ensemble of EMG gesture models')
    parser.add_argument('--model_dirs', type=str, nargs='+', required=True,
                       help='List of model directories')
    parser.add_argument('--data_path', type=str, default='.',
                       help='Path to EMG dataset')
    parser.add_argument('--method', type=str, default='average',
                       choices=['average', 'voting'],
                       help='Ensemble method')
    parser.add_argument('--exclude_pinch', action='store_true',
                       help='Exclude Pinch class')

    args = parser.parse_args()

    evaluate_ensemble(
        model_dirs=args.model_dirs,
        data_path=args.data_path,
        method=args.method,
        exclude_pinch=args.exclude_pinch
    )
