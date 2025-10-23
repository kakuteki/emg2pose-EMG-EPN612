"""Time Series Data Augmentation for EMG Signals"""
import numpy as np
from tsaug import AddNoise, TimeWarp

def apply_jitter(X, sigma=0.03):
    augmenter = AddNoise(scale=sigma)
    X_aug = np.zeros_like(X)
    for i in range(X.shape[0]):
        sample = X[i].T
        aug_sample = augmenter.augment(sample)
        X_aug[i] = aug_sample.T
    return X_aug

def apply_scaling(X, sigma=0.1):
    X_aug = np.zeros_like(X)
    for i in range(X.shape[0]):
        scaling_factor = np.random.normal(1.0, sigma, size=(X.shape[1], 1))
        X_aug[i] = X[i] * scaling_factor
    return X_aug

def apply_time_warp(X, n_speed_change=3, max_speed_ratio=2):
    augmenter = TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)
    X_aug = np.zeros_like(X)
    for i in range(X.shape[0]):
        sample = X[i].T
        aug_sample = augmenter.augment(sample)
        X_aug[i] = aug_sample.T
    return X_aug

def apply_magnitude_warp(X, sigma=0.2):
    X_aug = np.zeros_like(X)
    for i in range(X.shape[0]):
        n_timesteps = X.shape[2]
        knots = np.random.normal(1.0, sigma, size=(X.shape[1], 5))
        warp_curve = np.zeros((X.shape[1], n_timesteps))
        for j in range(X.shape[1]):
            knot_positions = np.linspace(0, n_timesteps - 1, 5)
            warp_curve[j] = np.interp(np.arange(n_timesteps), knot_positions, knots[j])
        X_aug[i] = X[i] * warp_curve
    return X_aug

def augment_emg_data(X, y, augmentation_factor=3, jitter_sigma=0.03, scaling_sigma=0.1,
                     time_warp_n_speed_change=3, time_warp_max_speed_ratio=2, magnitude_warp_sigma=0.2):
    methods = [
        lambda x: apply_jitter(x, sigma=jitter_sigma),
        lambda x: apply_scaling(x, sigma=scaling_sigma),
        lambda x: apply_time_warp(x, n_speed_change=time_warp_n_speed_change, max_speed_ratio=time_warp_max_speed_ratio),
        lambda x: apply_magnitude_warp(x, sigma=magnitude_warp_sigma)
    ]
    X_list, y_list = [X], [y]
    for i in range(augmentation_factor):
        method = np.random.choice(methods)
        X_list.append(method(X))
        y_list.append(y)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
