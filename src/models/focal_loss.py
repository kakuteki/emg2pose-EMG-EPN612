"""
Focal Loss implementation for handling class imbalance

Focal Loss: https://arxiv.org/abs/1708.02002
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where:
- p_t: predicted probability for the correct class
- alpha_t: class-specific weighting factor (inverse class frequency)
- gamma: focusing parameter (default: 2.0)

The focusing parameter gamma reduces the relative loss for well-classified examples,
putting more focus on hard, misclassified examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with class imbalance

    Args:
        alpha (torch.Tensor, optional): Class weights (inverse frequency).
                                       Shape: (num_classes,). Default: None
        gamma (float): Focusing parameter. Higher gamma puts more focus on hard examples.
                      gamma=0 is equivalent to CrossEntropyLoss. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output:
                        'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        >>> # Calculate class weights from training data
        >>> class_counts = torch.bincount(y_train)
        >>> alpha = 1.0 / (class_counts + 1e-6)
        >>> alpha = alpha / alpha.sum() * len(alpha)
        >>>
        >>> # Create Focal Loss with class weights
        >>> criterion = FocalLoss(alpha=alpha, gamma=2.0)
        >>>
        >>> # Use in training loop
        >>> outputs = model(inputs)
        >>> loss = criterion(outputs, labels)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from model. Shape: (batch_size, num_classes)
            targets (torch.Tensor): Ground truth class indices. Shape: (batch_size,)

        Returns:
            torch.Tensor: Focal loss value
        """
        # Get probabilities using softmax
        p = F.softmax(inputs, dim=1)

        # Get cross entropy loss (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get p_t: probability of the correct class
        # Create one-hot encoding of targets
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # p_t = probability assigned to the correct class
        p_t = torch.sum(p * targets_one_hot, dim=1)

        # Focal loss modulation factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Apply modulating factor to CE loss
        focal_loss = modulating_factor * ce_loss

        # Apply alpha (class weights) if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # Get alpha for each sample based on target class
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that automatically adjusts alpha based on class frequencies

    This variant automatically computes class weights from the batch,
    making it easier to use without manual weight calculation.

    Args:
        gamma (float): Focusing parameter. Default: 2.0
        reduction (str): Reduction method. Default: 'mean'
        smooth_alpha (float): Smoothing factor for alpha calculation. Default: 0.1
    """

    def __init__(self, gamma=2.0, reduction='mean', smooth_alpha=0.1):
        super(AdaptiveFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.smooth_alpha = smooth_alpha

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from model. Shape: (batch_size, num_classes)
            targets (torch.Tensor): Ground truth class indices. Shape: (batch_size,)

        Returns:
            torch.Tensor: Adaptive focal loss value
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)

        # Cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get p_t
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        p_t = torch.sum(p * targets_one_hot, dim=1)

        # Modulating factor
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Calculate adaptive alpha from batch
        # Count samples per class in current batch
        batch_class_counts = torch.bincount(targets, minlength=inputs.size(1)).float()

        # Add smoothing to avoid division by zero
        batch_class_counts = batch_class_counts + self.smooth_alpha

        # Inverse frequency weighting
        alpha = 1.0 / batch_class_counts
        alpha = alpha / alpha.sum() * inputs.size(1)

        # Get alpha for each sample
        alpha_t = alpha[targets].to(inputs.device)

        # Apply both modulating factor and alpha
        focal_loss = alpha_t * modulating_factor * ce_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Helper function to create Focal Loss with automatic class weight calculation
def create_focal_loss(class_counts, gamma=2.0, reduction='mean', device='cpu'):
    """
    Create Focal Loss with automatic alpha (class weights) calculation

    Args:
        class_counts (list or numpy.ndarray or torch.Tensor):
                     Number of samples per class in training set
        gamma (float): Focusing parameter. Default: 2.0
        reduction (str): Reduction method. Default: 'mean'
        device (str or torch.device): Device to put alpha on. Default: 'cpu'

    Returns:
        FocalLoss: Configured Focal Loss instance

    Example:
        >>> import numpy as np
        >>> # From training labels
        >>> class_counts = np.bincount(y_train)
        >>> print(f"Class distribution: {class_counts}")
        >>>
        >>> # Create Focal Loss
        >>> criterion = create_focal_loss(class_counts, gamma=2.0, device='cuda')
        >>>
        >>> # Use in training
        >>> for inputs, labels in train_loader:
        >>>     outputs = model(inputs)
        >>>     loss = criterion(outputs, labels)
    """
    # Convert to tensor if needed
    if not isinstance(class_counts, torch.Tensor):
        class_counts = torch.tensor(class_counts, dtype=torch.float32)

    # Calculate alpha (inverse frequency)
    alpha = 1.0 / (class_counts + 1e-6)  # Add small epsilon to avoid division by zero

    # Normalize alpha so they sum to num_classes
    alpha = alpha / alpha.sum() * len(alpha)

    # Move to device
    alpha = alpha.to(device)

    print(f"\nFocal Loss Configuration:")
    print(f"  Gamma: {gamma}")
    print(f"  Alpha (class weights): {alpha.cpu().numpy()}")
    print(f"  Reduction: {reduction}")

    return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
