"""
Simplified Ensemble Evaluation Script
"""
import sys
from pathlib import Path
import torch
import numpy as np
import argparse
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_loader import EMGDataLoader, create_data_split
from features.feature_extractor import EMGPreprocessor
from models.waveformer import WaveFormerComplete


def load_model(model_path, num_classes, device):
    """Load a trained model from checkpoint"""
    model = WaveFormerComplete(num_classes=num_classes, dropout=0.3).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def ensemble_predict(models, data_loader, device, method='average'):
    """Make ensemble predictions"""
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

                if len(true_labels) < len(batch_y) * len(data_loader):
                    true_labels.extend(batch_y.numpy())

        model_probs = np.vstack(model_probs)
        all_probs.append(model_probs)

    # Combine predictions
    all_probs = np.array(all_probs)  # Shape: (num_models, num_samples, num_classes)

    if method == 'average':
        avg_probs = np.mean(all_probs, axis=0)
        predictions = np.argmax(avg_probs, axis=1)
    elif method == 'voting':
        model_predictions = np.argmax(all_probs, axis=2)
        predictions = []
        for i in range(model_predictions.shape[1]):
            votes = model_predictions[:, i]
            prediction = np.bincount(votes).argmax()
            predictions.append(prediction)
        predictions = np.array(predictions)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    true_labels = np.array(true_labels[:len(predictions)])

    return predictions, true_labels


def main():
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Ensemble method: {args.method}")
    print(f"Number of model directories: {len(args.model_dirs)}")

    # Load data
    print("\nLoading data...")
    data_loader_obj = EMGDataLoader(data_path=args.data_path)
    X, y = data_loader_obj.load_all_data()

    # Create train/test split
    X_train, y_train, X_test, y_test = create_data_split(X, y, test_users=list(range(307, 613)))

    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # Preprocess data
    print("Preprocessing data...")
    preprocessor = EMGPreprocessor()
    X_test_preprocessed = preprocessor.fit_transform(X_test)

    # Handle Pinch exclusion
    if args.exclude_pinch:
        print("Excluding Pinch class...")
        mask = y_test != 5
        X_test_preprocessed = X_test_preprocessed[mask]
        y_test = y_test[mask]
        y_test[y_test > 5] -= 1
        num_classes = 5
    else:
        num_classes = 6

    print(f"Final test data shape: {X_test_preprocessed.shape}")
    print(f"Number of classes: {num_classes}")

    # Create data loader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_preprocessed),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load all models
    print("\nLoading models...")
    models = []
    for i, model_dir in enumerate(args.model_dirs):
        model_path = Path(model_dir) / 'best_model.pth'
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}, skipping...")
            continue

        print(f"  [{i+1}/{len(args.model_dirs)}] {model_dir}")
        model = load_model(model_path, num_classes, device)
        models.append(model)

    if len(models) == 0:
        print("Error: No models loaded!")
        return

    print(f"\nSuccessfully loaded {len(models)} models")

    # Make ensemble predictions
    print(f"\nMaking ensemble predictions using {args.method} method...")
    predictions, true_labels = ensemble_predict(models, test_loader, device, method=args.method)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions) * 100
    print(f"\n{'='*80}")
    print(f"Ensemble Test Accuracy ({args.method}): {accuracy:.2f}%")
    print(f"{'='*80}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    target_names = ['No Gesture', 'Wave In', 'Wave Out', 'Fist', 'Open'] if args.exclude_pinch else \
                   ['No Gesture', 'Wave In', 'Wave Out', 'Fist', 'Open', 'Pinch']
    print(classification_report(true_labels, predictions, target_names=target_names))

    # Individual model accuracies
    print("\nIndividual Model Accuracies:")
    for i, model_dir in enumerate(args.model_dirs):
        if i < len(models):
            model = models[i]
            model_preds, _ = ensemble_predict([model], test_loader, device, method='average')
            model_acc = accuracy_score(true_labels, model_preds) * 100
            print(f"  Model {i+1} ({Path(model_dir).name}): {model_acc:.2f}%")

    # Save results
    results = {
        'ensemble_method': args.method,
        'num_models': len(models),
        'model_dirs': [str(d) for d in args.model_dirs],
        'test_accuracy': float(accuracy),
        'num_classes': num_classes,
        'exclude_pinch': args.exclude_pinch,
        'confusion_matrix': cm.tolist()
    }

    output_file = f'ensemble_results_{args.method}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
