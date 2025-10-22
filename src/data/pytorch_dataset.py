"""
PyTorch Dataset and DataLoader for EMG data
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class EMGDataset(Dataset):
    """PyTorch Dataset for EMG signals"""

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        """
        Args:
            X: EMG signals shape (num_samples, num_channels, sequence_length)
            y: Labels shape (num_samples,)
            transform: Optional transform to apply
        """
        # Make copies to avoid negative stride issues with PyTorch
        self.X = torch.FloatTensor(X.copy())
        self.y = torch.LongTensor(y.copy())
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def create_dataloaders(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       batch_size: int = 64,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = EMGDataset(X_train, y_train)
    val_dataset = EMGDataset(X_val, y_val)
    test_dataset = EMGDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory for Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory for Windows compatibility
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory for Windows compatibility
    )

    return train_loader, val_loader, test_loader


class EMGAugmentation:
    """Data augmentation for EMG signals"""

    @staticmethod
    def add_noise(x, noise_level=0.01):
        """Add Gaussian noise"""
        noise = torch.randn_like(x) * noise_level
        return x + noise

    @staticmethod
    def time_shift(x, shift_range=10):
        """Shift signal in time"""
        shift = np.random.randint(-shift_range, shift_range)
        return torch.roll(x, shift, dims=-1)

    @staticmethod
    def scale(x, scale_range=(0.9, 1.1)):
        """Scale signal amplitude"""
        scale_factor = np.random.uniform(*scale_range)
        return x * scale_factor

    @staticmethod
    def random_augment(x):
        """Apply random augmentation"""
        if np.random.rand() > 0.5:
            x = EMGAugmentation.add_noise(x)
        if np.random.rand() > 0.5:
            x = EMGAugmentation.time_shift(x)
        if np.random.rand() > 0.5:
            x = EMGAugmentation.scale(x)
        return x
