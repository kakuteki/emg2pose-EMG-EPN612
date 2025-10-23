"""
Focal Loss implementation for imbalanced classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks

    Reference:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal loss for dense object detection. ICCV.

    Args:
        alpha: Class weights (tensor of shape [num_classes])
        gamma: Focusing parameter for modulating loss (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss

        Args:
            alpha: Class weights as tensor or None
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute focal loss

        Args:
            inputs: Predictions (logits) of shape [batch_size, num_classes]
            targets: Ground truth labels of shape [batch_size]

        Returns:
            Focal loss value
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # Get probabilities
        p = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = (1 - p) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
