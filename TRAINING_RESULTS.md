# EMG-Diffusion Training Results

## Model Architecture

**Two-Stage Architecture:**
1. **Transformer Feature Extractor**: Conv1D + Positional Encoding + 6-layer Transformer
2. **Diffusion Classifier**: Denoising diffusion probabilistic model for classification

## Training Configuration

- **Epochs**: 50
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Classes**: 5 (Pinch excluded)
- **Total Parameters**: 5,805,317

## Dataset

- **Training Samples**: 7,835
- **Validation Samples**: 1,567  
- **Test Samples**: 7,773
- **Input Shape**: (8 channels × 200 samples)

## Extracted Features

### Training Features
- **Shape**: (7835, 128)
- **Mean**: 0.3897
- **Std**: 0.6452
- **Range**: [0.0000, 2.7430]

### Test Features
- **Shape**: (7773, 128)
- **Mean**: 0.3897
- **Std**: 0.6452
- **Range**: [0.0000, 2.7430]

## Files

- **Model**: `results/emg_diffusion_retrain/best_model.pth` (28MB)
- **Features**: `extracted_features_retrain/`
  - `train_features.npy` - Training set transformer features
  - `train_labels.npy` - Training labels
  - `test_features.npy` - Test set transformer features
  - `test_labels.npy` - Test labels
- **Feature Extraction Script**: `extract_features.py`

## Usage

### Load Features

```python
import numpy as np

# Load features
train_features = np.load('extracted_features_retrain/train_features.npy')
train_labels = np.load('extracted_features_retrain/train_labels.npy')
test_features = np.load('extracted_features_retrain/test_features.npy')
test_labels = np.load('extracted_features_retrain/test_labels.npy')

print(f"Train features: {train_features.shape}")
print(f"Test features: {test_features.shape}")
```

### Extract Features from New Data

```bash
python extract_features.py \
  --model_path results/emg_diffusion_retrain/best_model.pth \
  --exclude_pinch \
  --output_dir my_features
```

---

**Training Date**: 2025-11-10
**Device**: CUDA
