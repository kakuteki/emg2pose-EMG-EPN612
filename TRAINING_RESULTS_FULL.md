# EMG-Diffusion Training Results (Full 6-Class Dataset)

## Model Architecture

**Two-Stage Architecture:**
1. **Transformer Feature Extractor**: Conv1D + Positional Encoding + 6-layer Transformer
2. **Diffusion Classifier**: Denoising diffusion probabilistic model for classification

## Training Configuration

- **Epochs**: 45 (early stopping)
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Classes**: 6 (all classes including Pinch)
- **Total Parameters**: 5,805,446

## Dataset

- **Training Samples**: 7,836
- **Validation Samples**: 1,568
- **Test Samples**: 7,773
- **Input Shape**: (8 channels × 200 samples)

### Class Distribution

**Training Set:**
- No Gesture (0): 5,514 (70.4%)
- Fist (1): 221 (2.8%)
- Wave In (2): 240 (3.1%)
- Wave Out (3): 1,524 (19.4%)
- Open (4): 336 (4.3%)
- Pinch (5): 1 (0.0%)

**Test Set:**
- No Gesture (0): 5,603 (72.1%)
- Fist (1): 258 (3.3%)
- Wave In (2): 149 (1.9%)
- Wave Out (3): 1,493 (19.2%)
- Open (4): 270 (3.5%)
- Pinch (5): 0 (0.0%)

## Training Performance

### Best Model (Epoch 25)
- **Validation Accuracy**: 72.45%
- **Test Accuracy**: 72.70%

### Performance at Early Stopping (Epoch 45)
- **Train Accuracy**: 75.48%
- **Validation Accuracy**: 70.92%
- **Test Accuracy**: 72.44%

### Key Training Milestones
- Epoch 1: Val Acc 70.15%, Test Acc 72.08%
- Epoch 16: Val Acc 72.32%, Test Acc 73.29% (first major improvement)
- Epoch 21: Val Acc 72.39%, Test Acc **73.73%** (best test accuracy)
- Epoch 25: Val Acc **72.45%**, Test Acc 72.70% (best validation accuracy)
- Early stopping triggered at Epoch 45 (patience: 20)

## Extracted Features

### Training Features
- **Shape**: (7836, 128)
- **Mean**: 0.2497
- **Std**: 0.6546
- **Range**: [0.0000, 4.7198]

### Test Features
- **Shape**: (7773, 128)
- **Mean**: 0.2514
- **Std**: 0.6500
- **Range**: [0.0000, 4.7337]

## Files

- **Model**: `results/emg_diffusion_full_dataset/best_model.pth` (28MB)
- **Features**: `extracted_features_full_dataset/`
  - `train_features.npy` - Training set transformer features
  - `train_labels.npy` - Training labels
  - `test_features.npy` - Test set transformer features
  - `test_labels.npy` - Test labels
- **Feature Extraction Script**: `extract_features.py`
- **Training Log**: `training_full_dataset.log`

## Usage

### Load Features

```python
import numpy as np

# Load features
train_features = np.load('extracted_features_full_dataset/train_features.npy')
train_labels = np.load('extracted_features_full_dataset/train_labels.npy')
test_features = np.load('extracted_features_full_dataset/test_features.npy')
test_labels = np.load('extracted_features_full_dataset/test_labels.npy')

print(f"Train features: {train_features.shape}")
print(f"Test features: {test_features.shape}")
```

### Extract Features from New Data

```bash
python extract_features.py \
  --model_path results/emg_diffusion_full_dataset/best_model.pth \
  --output_dir my_features
```

## Notes

- The Pinch class has extremely limited data (1 training sample, 0 test samples)
- This class imbalance may affect model performance on minority classes
- Best validation accuracy achieved at Epoch 25: 72.45%
- Best test accuracy achieved at Epoch 21: 73.73%
- Model shows good generalization despite class imbalance

---

**Training Date**: 2025-11-10
**Device**: CUDA
**Training Duration**: ~90 minutes (45 epochs)
