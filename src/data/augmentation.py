"""
Time Series Data Augmentation for EMG Signals

This module provides various augmentation techniques for time series EMG data
to improve model generalization and handle class imbalance.
"""

import numpy as np
from tsaug import AddNoise, Quantize, Drift, TimeWarp, Convolve


def apply_jitter(X, sigma=0.03):
    """
    Add random noise (jitter) to the time series.

    Args:
        X: Input data of shape (n_samples, n_features, n_timesteps)
        sigma: Standard deviation of the noise

    Returns:
        Augmented data with the same shape as input
    """
    augmenter = AddNoise(scale=sigma)
    X_aug = np.zeros_like(X)

    for i in range(X.shape[0]):
        # Apply augmentation to each sample
        # tsaug expects (n_timesteps, n_features)
        sample = X[i].T  # Transpose to (n_timesteps, n_features)
        aug_sample = augmenter.augment(sample)
        X_aug[i] = aug_sample.T  # Transpose back to (n_features, n_timesteps)

    return X_aug


def apply_scaling(X, sigma=0.1):
    """
    Apply random scaling to the amplitude of the time series.

    Args:
        X: Input data of shape (n_samples, n_features, n_timesteps)
        sigma: Standard deviation of the scaling factor

    Returns:
        Augmented data with the same shape as input
    """
    X_aug = np.zeros_like(X)

    for i in range(X.shape[0]):
        # Generate random scaling factor for each sample
        scaling_factor = np.random.normal(1.0, sigma, size=(X.shape[1], 1))
        X_aug[i] = X[i] * scaling_factor

    return X_aug


def apply_time_warp(X, n_speed_change=3, max_speed_ratio=2):
    """
    Apply time warping to the time series by changing speed at random locations.

    Args:
        X: Input data of shape (n_samples, n_features, n_timesteps)
        n_speed_change: Number of speed changes
        max_speed_ratio: Maximum ratio of speed change

    Returns:
        Augmented data with the same shape as input
    """
    augmenter = TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)
    X_aug = np.zeros_like(X)

    for i in range(X.shape[0]):
        # Apply augmentation to each sample
        # tsaug expects (n_timesteps, n_features)
        sample = X[i].T  # Transpose to (n_timesteps, n_features)
        aug_sample = augmenter.augment(sample)
        X_aug[i] = aug_sample.T  # Transpose back to (n_features, n_timesteps)

    return X_aug


def apply_magnitude_warp(X, sigma=0.2):
    """
    Apply magnitude warping by smoothly distorting the magnitude of the signal.

    Args:
        X: Input data of shape (n_samples, n_features, n_timesteps)
        sigma: Standard deviation of the warping curve

    Returns:
        Augmented data with the same shape as input
    """
    X_aug = np.zeros_like(X)

    for i in range(X.shape[0]):
        # Generate smooth random curve for magnitude warping
        n_timesteps = X.shape[2]
        # Generate random knots
        knots = np.random.normal(1.0, sigma, size=(X.shape[1], 5))

        # Interpolate to create smooth warping curve
        warp_curve = np.zeros((X.shape[1], n_timesteps))
        for j in range(X.shape[1]):
            knot_positions = np.linspace(0, n_timesteps - 1, 5)
            warp_curve[j] = np.interp(
                np.arange(n_timesteps),
                knot_positions,
                knots[j]
            )

        X_aug[i] = X[i] * warp_curve

    return X_aug


def augment_emg_data(X, y, augmentation_factor=3,
                     jitter_sigma=0.03, scaling_sigma=0.1,
                     time_warp_n_speed_change=3, time_warp_max_speed_ratio=2,
                     magnitude_warp_sigma=0.2):
    """
    Apply multiple augmentation techniques to EMG data.

    This function combines different augmentation methods to generate
    augmented samples from the original data.

    Args:
        X: Input data of shape (n_samples, n_features, n_timesteps)
        y: Labels of shape (n_samples,)
        augmentation_factor: Number of augmented samples per original sample
        jitter_sigma: Standard deviation for jitter noise
        scaling_sigma: Standard deviation for scaling
        time_warp_n_speed_change: Number of speed changes for time warping
        time_warp_max_speed_ratio: Maximum speed ratio for time warping
        magnitude_warp_sigma: Standard deviation for magnitude warping

    Returns:
        X_augmented: Augmented data including original samples
        y_augmented: Labels for augmented data
    """
    augmentation_methods = [
        lambda x: apply_jitter(x, sigma=jitter_sigma),
        lambda x: apply_scaling(x, sigma=scaling_sigma),
        lambda x: apply_time_warp(x, n_speed_change=time_warp_n_speed_change,
                                   max_speed_ratio=time_warp_max_speed_ratio),
        lambda x: apply_magnitude_warp(x, sigma=magnitude_warp_sigma)
    ]

    X_augmented_list = [X]  # Start with original data
    y_augmented_list = [y]

    # Generate augmented samples
    for i in range(augmentation_factor):
        # Randomly select an augmentation method
        method = np.random.choice(augmentation_methods)
        X_aug = method(X)

        X_augmented_list.append(X_aug)
        y_augmented_list.append(y)

    # Concatenate all augmented data
    X_augmented = np.concatenate(X_augmented_list, axis=0)
    y_augmented = np.concatenate(y_augmented_list, axis=0)

    return X_augmented, y_augmented


def augment_by_class(X, y, augmentation_factor=3, target_classes=None,
                     jitter_sigma=0.03, scaling_sigma=0.1,
                     time_warp_n_speed_change=3, time_warp_max_speed_ratio=2,
                     magnitude_warp_sigma=0.2):
    """
    Apply augmentation selectively to specific classes (useful for handling class imbalance).

    Args:
        X: Input data of shape (n_samples, n_features, n_timesteps)
        y: Labels of shape (n_samples,)
        augmentation_factor: Number of augmented samples per original sample
        target_classes: List of classes to augment. If None, augment all classes
        jitter_sigma: Standard deviation for jitter noise
        scaling_sigma: Standard deviation for scaling
        time_warp_n_speed_change: Number of speed changes for time warping
        time_warp_max_speed_ratio: Maximum speed ratio for time warping
        magnitude_warp_sigma: Standard deviation for magnitude warping

    Returns:
        X_augmented: Augmented data including original samples
        y_augmented: Labels for augmented data
    """
    if target_classes is None:
        # Augment all classes
        return augment_emg_data(
            X, y, augmentation_factor, jitter_sigma, scaling_sigma,
            time_warp_n_speed_change, time_warp_max_speed_ratio, magnitude_warp_sigma
        )

    X_augmented_list = [X]  # Start with all original data
    y_augmented_list = [y]

    # Augment only target classes
    for class_label in target_classes:
        class_mask = (y == class_label)
        X_class = X[class_mask]
        y_class = y[class_mask]

        if len(X_class) > 0:
            X_class_aug, y_class_aug = augment_emg_data(
                X_class, y_class, augmentation_factor,
                jitter_sigma, scaling_sigma,
                time_warp_n_speed_change, time_warp_max_speed_ratio,
                magnitude_warp_sigma
            )
            # Only add the newly augmented samples (exclude original)
            X_augmented_list.append(X_class_aug[len(X_class):])
            y_augmented_list.append(y_class_aug[len(y_class):])

    # Concatenate all data
    X_augmented = np.concatenate(X_augmented_list, axis=0)
    y_augmented = np.concatenate(y_augmented_list, axis=0)

    return X_augmented, y_augmented
