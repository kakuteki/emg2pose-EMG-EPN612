"""
Extract learned features from trained EMG-Diffusion model

This script loads a trained model and extracts the transformer features
for all samples in the dataset.
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, 'src')

from models.emg_diffusion import EMGDiffusionModel
from data.data_loader import EMGDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Extract features from EMG-Diffusion model')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='.',
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for extraction')
    parser.add_argument('--exclude_pinch', action='store_true',
                        help='Exclude Pinch class (class 5)')
    parser.add_argument('--output_dir', type=str, default='extracted_features',
                        help='Output directory for features')

    return parser.parse_args()


def extract_features(model, dataloader, device):
    """Extract features from all samples"""
    model.eval()
    all_features = []
    all_labels = []

    print("Extracting features...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Processing"):
            batch_x = batch_x.to(device)

            # Extract features using the transformer feature extractor
            features = model.extract_features(batch_x)

            all_features.append(features.cpu().numpy())
            all_labels.append(batch_y.numpy())

    # Concatenate all batches
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


def main():
    args = parse_args()

    print("="*80)
    print("EMG-Diffusion Feature Extraction")
    print("="*80)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    print("\nLoading dataset...")
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

    num_classes = len(np.unique(y_train))
    print(f"\nTrain samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Number of classes: {num_classes}")

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Load model
    print("\nLoading trained model...")
    model = EMGDiffusionModel(
        in_channels=X_train.shape[1],
        num_classes=num_classes,
        d_model=256,
        nhead=8,
        num_layers=6,
        feature_dim=128,
        num_timesteps=100,
        hidden_dim=256
    ).to(device)

    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("Model loaded successfully!")

    # Extract features for training set
    print("\n" + "="*60)
    print("Extracting training features...")
    print("="*60)
    train_features, train_labels = extract_features(model, train_loader, device)

    # Extract features for test set
    print("\n" + "="*60)
    print("Extracting test features...")
    print("="*60)
    test_features, test_labels = extract_features(model, test_loader, device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save features
    print("\n" + "="*60)
    print("Saving features...")
    print("="*60)

    np.save(os.path.join(args.output_dir, 'train_features.npy'), train_features)
    np.save(os.path.join(args.output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(args.output_dir, 'test_features.npy'), test_features)
    np.save(os.path.join(args.output_dir, 'test_labels.npy'), test_labels)

    print(f"\nFeatures saved to: {args.output_dir}")
    print(f"  - train_features.npy: {train_features.shape}")
    print(f"  - train_labels.npy: {train_labels.shape}")
    print(f"  - test_features.npy: {test_features.shape}")
    print(f"  - test_labels.npy: {test_labels.shape}")

    # Print feature statistics
    print("\n" + "="*60)
    print("Feature Statistics")
    print("="*60)
    print(f"\nTraining features:")
    print(f"  Shape: {train_features.shape}")
    print(f"  Mean: {train_features.mean():.4f}")
    print(f"  Std: {train_features.std():.4f}")
    print(f"  Min: {train_features.min():.4f}")
    print(f"  Max: {train_features.max():.4f}")

    print(f"\nTest features:")
    print(f"  Shape: {test_features.shape}")
    print(f"  Mean: {test_features.mean():.4f}")
    print(f"  Std: {test_features.std():.4f}")
    print(f"  Min: {test_features.min():.4f}")
    print(f"  Max: {test_features.max():.4f}")

    print("\n" + "="*80)
    print("Feature extraction completed!")
    print("="*80)


if __name__ == '__main__':
    main()
