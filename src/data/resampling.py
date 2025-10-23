"""
Data resampling utilities for handling class imbalance.

This module provides functions to apply SMOTE (Synthetic Minority Over-sampling Technique)
and combined resampling strategies (SMOTE + Under-sampling) to balance class distribution.
"""

import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN


def apply_smote_resampling(X_train, y_train, target_strategy='auto', random_state=42):
    """
    Apply SMOTE to balance minority classes.

    Args:
        X_train (np.ndarray): Training features of shape (n_samples, n_features)
        y_train (np.ndarray): Training labels of shape (n_samples,)
        target_strategy (str or dict): Sampling strategy for SMOTE
            - 'auto': Balance all minority classes to match majority class
            - 'minority': Resample all classes but the majority class
            - dict: Custom sampling strategy, e.g., {0: 2000, 1: 2000, 2: 2000}
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_resampled, y_resampled) - Resampled features and labels
    """
    print("\n" + "="*70)
    print("Applying SMOTE Resampling")
    print("="*70)

    # Print original class distribution
    original_distribution = Counter(y_train)
    print("\nOriginal class distribution:")
    for class_label, count in sorted(original_distribution.items()):
        print(f"  Class {class_label}: {count} samples")
    print(f"  Total: {len(y_train)} samples")

    # Apply SMOTE
    smote = SMOTE(sampling_strategy=target_strategy, random_state=random_state, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Print new class distribution
    new_distribution = Counter(y_resampled)
    print("\nResampled class distribution (after SMOTE):")
    for class_label, count in sorted(new_distribution.items()):
        change = count - original_distribution.get(class_label, 0)
        print(f"  Class {class_label}: {count} samples (+{change})")
    print(f"  Total: {len(y_resampled)} samples")

    print("="*70)

    return X_resampled, y_resampled


def apply_combined_resampling(X_train, y_train,
                              smote_strategy='auto',
                              undersample_strategy='auto',
                              random_state=42):
    """
    Apply combined resampling: SMOTE for minority classes + Under-sampling for majority class.

    This two-step approach:
    1. First applies SMOTE to increase minority class samples
    2. Then applies random under-sampling to reduce majority class samples

    Args:
        X_train (np.ndarray): Training features of shape (n_samples, n_features)
        y_train (np.ndarray): Training labels of shape (n_samples,)
        smote_strategy (str or dict): SMOTE sampling strategy
            - 'auto': Balance all minority classes to match majority class
            - dict: Custom strategy, e.g., {0: 2000, 1: 2000, 2: 2000}
        undersample_strategy (str or dict): Under-sampling strategy
            - 'auto': Balance all classes to the minority class count
            - dict: Custom strategy, e.g., {3: 2500}
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_resampled, y_resampled) - Resampled features and labels
    """
    print("\n" + "="*70)
    print("Applying Combined Resampling (SMOTE + Under-sampling)")
    print("="*70)

    # Print original class distribution
    original_distribution = Counter(y_train)
    print("\nOriginal class distribution:")
    for class_label, count in sorted(original_distribution.items()):
        print(f"  Class {class_label}: {count} samples")
    print(f"  Total: {len(y_train)} samples")

    # Step 1: Apply SMOTE to minority classes
    print("\nStep 1: Applying SMOTE to minority classes...")
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=random_state, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    smote_distribution = Counter(y_smote)
    print("After SMOTE:")
    for class_label, count in sorted(smote_distribution.items()):
        change = count - original_distribution.get(class_label, 0)
        print(f"  Class {class_label}: {count} samples (+{change})")
    print(f"  Total: {len(y_smote)} samples")

    # Step 2: Apply under-sampling to majority class
    print("\nStep 2: Applying under-sampling to majority class...")
    undersampler = RandomUnderSampler(sampling_strategy=undersample_strategy,
                                      random_state=random_state)
    X_resampled, y_resampled = undersampler.fit_resample(X_smote, y_smote)

    # Print final class distribution
    final_distribution = Counter(y_resampled)
    print("\nFinal resampled class distribution:")
    for class_label, count in sorted(final_distribution.items()):
        original_count = original_distribution.get(class_label, 0)
        change = count - original_count
        sign = '+' if change >= 0 else ''
        print(f"  Class {class_label}: {count} samples ({sign}{change})")
    print(f"  Total: {len(y_resampled)} samples")

    print("="*70)

    return X_resampled, y_resampled


def get_target_sampling_strategy(y_train, minority_target=2000, majority_target=2500):
    """
    Generate a custom sampling strategy based on current class distribution.

    This function identifies minority and majority classes and creates a sampling
    strategy dictionary for both SMOTE and under-sampling.

    Args:
        y_train (np.ndarray): Training labels
        minority_target (int): Target number of samples for minority classes
        majority_target (int): Target number of samples for majority class

    Returns:
        tuple: (smote_strategy, undersample_strategy) - Strategy dictionaries
    """
    class_distribution = Counter(y_train)

    # Find the majority class (class with most samples)
    majority_class = max(class_distribution, key=class_distribution.get)
    majority_count = class_distribution[majority_class]

    # Create SMOTE strategy: increase minority classes
    smote_strategy = {}
    for class_label, count in class_distribution.items():
        if class_label != majority_class:
            # Only oversample if current count is less than target
            if count < minority_target:
                smote_strategy[class_label] = minority_target

    # Create under-sampling strategy: reduce majority class
    undersample_strategy = {}
    if majority_count > majority_target:
        undersample_strategy[majority_class] = majority_target

    print(f"\nGenerated sampling strategy:")
    print(f"  Majority class: {majority_class} ({majority_count} samples)")
    print(f"  SMOTE strategy: {smote_strategy}")
    print(f"  Under-sampling strategy: {undersample_strategy}")

    return smote_strategy, undersample_strategy


def apply_smote_tomek(X_train, y_train, sampling_strategy='auto', random_state=42):
    """
    Apply SMOTE followed by Tomek links cleaning.

    This method combines SMOTE with Tomek links removal to clean overlapping samples
    at the class boundaries.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        sampling_strategy (str or dict): SMOTE sampling strategy
        random_state (int): Random seed

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("\n" + "="*70)
    print("Applying SMOTE + Tomek Links")
    print("="*70)

    original_distribution = Counter(y_train)
    print("\nOriginal class distribution:")
    for class_label, count in sorted(original_distribution.items()):
        print(f"  Class {class_label}: {count} samples")

    smote_tomek = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

    final_distribution = Counter(y_resampled)
    print("\nResampled class distribution:")
    for class_label, count in sorted(final_distribution.items()):
        change = count - original_distribution.get(class_label, 0)
        print(f"  Class {class_label}: {count} samples (+{change})")

    print("="*70)

    return X_resampled, y_resampled


def apply_smote_enn(X_train, y_train, sampling_strategy='auto', random_state=42):
    """
    Apply SMOTE followed by Edited Nearest Neighbours cleaning.

    This method combines SMOTE with ENN to remove samples whose class label differs
    from at least two of its three nearest neighbors.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        sampling_strategy (str or dict): SMOTE sampling strategy
        random_state (int): Random seed

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("\n" + "="*70)
    print("Applying SMOTE + ENN")
    print("="*70)

    original_distribution = Counter(y_train)
    print("\nOriginal class distribution:")
    for class_label, count in sorted(original_distribution.items()):
        print(f"  Class {class_label}: {count} samples")

    smote_enn = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

    final_distribution = Counter(y_resampled)
    print("\nResampled class distribution:")
    for class_label, count in sorted(final_distribution.items()):
        change = count - original_distribution.get(class_label, 0)
        print(f"  Class {class_label}: {count} samples (+{change})")

    print("="*70)

    return X_resampled, y_resampled
