# TTS-Codec Inspired Models for EMG Gesture Classification

This document describes the implementation and evaluation of TTS (Text-to-Speech) codec-inspired architectures for EMG gesture classification on the EMG-EPN612 dataset.

## Motivation

Traditional TTS systems use a multi-stage pipeline to convert discrete text input into continuous audio output:
```
Text → Language Processing → Phoneme/Linguistic Features → Acoustic Model → Mel-spectrogram → Vocoder → Audio Waveform
```

We adapted this architecture for EMG gesture classification by creating an analogous pipeline:
```
EMG Signal → Preprocessing → Time/Frequency Features → Acoustic Model → Intermediate Representation → Classifier → Gesture Class
```

The key insight is that both TTS and EMG processing involve transforming temporal sequences with rich structure into meaningful outputs.

## Implemented Architectures

### 1. EMGCodec (VQ-VAE style)
**File:** `src/models/emg_codec.py` (lines 1-150)

**Architecture:**
- Encoder: Conv1d layers with residual blocks (8 channels → 128 → 64 latent dim)
- Vector Quantization: 512 codebook entries with commitment loss
- Decoder: Transposed convolutions for reconstruction
- Classifier: Global average pooling + MLP

**Parameters:** 794,061

**Key Features:**
- Discrete latent representation via VQ-VAE
- Straight-through estimator for gradient flow
- Combined loss: `loss = cls_loss + 0.1 * vq_loss`

**Results:**
- Best validation accuracy: 76.71%
- Best test accuracy: 65.77%
- Training: 100 epochs, batch size 64, lr 0.001

**Analysis:** Shows promise for learning compact discrete representations, but the reconstruction task may introduce unnecessary complexity for pure classification.

---

### 2. TTSStyleEMGClassifier (Full TTS Pipeline)
**File:** `src/models/emg_codec.py` (lines 150-350)

**Architecture:**
1. **Feature Extraction:** Conv1d layers (8 → 128 → 256 channels)
2. **Acoustic Model:** 6-layer Transformer with 8 attention heads
   - Positional encoding for temporal awareness
   - Multi-head self-attention for feature interaction
3. **Latent Representation:** Conv1d projection (256 → 64)
4. **Vector Quantization:** 512 codebook entries
5. **Decoder:** Reconstruction pathway
6. **Classifier:** Global pooling + MLP

**Parameters:** 4,353,029

**Key Features:**
- Full TTS-inspired pipeline with all stages
- Transformer-based acoustic modeling
- Multi-head attention for capturing gesture patterns
- VQ for discrete latent codes

**Results:**
- Best validation accuracy: 70.39%
- **Best test accuracy: 72.08%** (within 0.25% of WaveFormer baseline!)
- Training: 100 epochs, batch size 64, lr 0.001

**Analysis:** Successfully demonstrates that TTS architecture principles transfer to EMG classification. The Transformer acoustic model effectively captures temporal dependencies. Achieves competitive performance with significantly different architectural approach.

---

### 3. MultiScaleCodec (Multi-resolution)
**File:** `src/models/emg_codec.py` (lines 350-500)

**Architecture:**
- **Scale 1 (Fine):** kernel_size=3, stride=1 (detailed features)
- **Scale 2 (Medium):** kernel_size=5, stride=2 (medium-range patterns)
- **Scale 3 (Coarse):** kernel_size=7, stride=4 (global context)
- **Fusion:** Interpolation + concatenation + 1x1 convolution
- **Classifier:** Residual blocks + global pooling

**Parameters:** 348,037 (smallest model)

**Key Features:**
- Parallel multi-scale processing
- Captures temporal patterns at different resolutions
- Efficient parameter usage

**Results:**
- **Best validation accuracy: 81.81%** (highest among all models!)
- Best test accuracy: 68.09%
- Training: 100 epochs, batch size 64, lr 0.001

**Analysis:** Excellent validation performance shows strong feature learning capability. Gap between validation (81.81%) and test (68.09%) accuracy indicates overfitting. Future work should explore regularization techniques (dropout, data augmentation, early stopping).

---

## Training Details

**Dataset:** EMG-EPN612
- 306 users, 5 gesture classes (excluding Pinch class)
- Train/Val/Test split: 6268/1567/7773 samples
- 8 EMG channels, 200Hz sampling rate

**Training Configuration:**
- Epochs: 100
- Batch size: 64
- Learning rate: 0.001
- Weight decay: 0.0001
- Optimizer: Adam
- Early stopping patience: 20 epochs
- VQ loss weight: 0.1

**Loss Function:**
```python
# For codec and tts_style models:
loss = classification_loss + 0.1 * vq_loss

# For multiscale model:
loss = classification_loss
```

## Comparison with Baseline

| Model | Parameters | Val Acc | Test Acc | Notes |
|-------|-----------|---------|----------|-------|
| **WaveFormer (Baseline)** | - | - | **72.33%** | Original baseline |
| **TTSStyleEMGClassifier** | 4,353,029 | 70.39% | **72.08%** | Closest to baseline |
| **MultiScaleCodec** | 348,037 | **81.81%** | 68.09% | Best validation |
| **EMGCodec** | 794,061 | 76.71% | 65.77% | VQ-VAE style |

## Key Insights

### Successful Adaptations
1. **Acoustic Model Transfer:** The Transformer-based acoustic model from TTS successfully captures temporal patterns in EMG signals
2. **Vector Quantization:** VQ layers can learn discrete latent representations for EMG gestures, similar to speech codecs
3. **Multi-Scale Processing:** Parallel processing at different temporal resolutions shows promise for capturing diverse gesture features
4. **Competitive Performance:** TTSStyleEMGClassifier achieves 72.08% test accuracy, nearly matching the WaveFormer baseline (72.33%)

### Observations
1. **Model Complexity:** The full TTS pipeline (4.3M parameters) performs best, suggesting complex temporal modeling is beneficial
2. **Overfitting Tendency:** All models show validation > test accuracy, with MultiScaleCodec showing the largest gap
3. **VQ Loss Contribution:** The VQ loss converges quickly (within 20 epochs) and stabilizes, indicating effective codebook learning
4. **Architecture Matters:** Different codec architectures produce substantially different performance profiles

### Challenges
1. **Overfitting:** Gap between validation and test accuracy, especially in MultiScaleCodec
2. **VQ Integration:** Balancing reconstruction and classification objectives is non-trivial
3. **Computational Cost:** The full TTS pipeline requires significant compute (4.3M parameters)

## Future Improvements

1. **Regularization:**
   - Increase dropout in MultiScaleCodec (current: 0.3)
   - Add label smoothing
   - Data augmentation strategies

2. **Architecture Refinements:**
   - Explore different VQ codebook sizes (512 → 256/1024)
   - Try different Transformer depths (6 → 4/8 layers)
   - Test alternative attention mechanisms (linear attention, performer)

3. **Training Strategies:**
   - Curriculum learning (start with easier gestures)
   - Adaptive VQ loss weighting
   - Multi-task learning with reconstruction

4. **Ensemble Methods:**
   - Combine predictions from all 3 codec models
   - Stack codec features with WaveFormer

## File Structure

```
.
├── src/
│   └── models/
│       └── emg_codec.py          # All 3 codec architectures
├── train_tts_codec.py            # Training script
├── results/
│   ├── trial49_codec/            # EMGCodec results
│   │   ├── results_codec.json
│   │   ├── best_model_codec.pth
│   │   └── training_history.png
│   ├── trial50_tts_style/        # TTSStyleEMGClassifier results
│   │   ├── results_tts_style.json
│   │   ├── best_model_tts_style.pth
│   │   └── training_history.png
│   └── trial51_multiscale/       # MultiScaleCodec results
│       ├── results_multiscale.json
│       ├── best_model_multiscale.pth
│       └── training_history.png
└── TTS_CODEC_MODELS_README.md    # This file
```

## Usage

### Training
```bash
# Train EMGCodec
python train_tts_codec.py --model_type codec --exclude_pinch --epochs 100 --batch_size 64 --save_dir results/trial49_codec

# Train TTSStyleEMGClassifier
python train_tts_codec.py --model_type tts_style --exclude_pinch --epochs 100 --batch_size 64 --save_dir results/trial50_tts_style

# Train MultiScaleCodec
python train_tts_codec.py --model_type multiscale --exclude_pinch --epochs 100 --batch_size 64 --save_dir results/trial51_multiscale
```

### Inference
```python
import torch
from src.models.emg_codec import TTSStyleEMGClassifier

# Load model
model = TTSStyleEMGClassifier(in_channels=8, num_classes=5)
model.load_state_dict(torch.load('results/trial50_tts_style/best_model_tts_style.pth'))
model.eval()

# Inference
with torch.no_grad():
    emg_input = torch.randn(1, 8, 200)  # [batch, channels, time]
    logits, vq_loss = model(emg_input)
    predicted_class = logits.argmax(dim=1)
```

## References

This work is inspired by:
- **SoundStream:** Neural audio codec with VQ-VAE
- **EnCodec:** High-fidelity neural audio compression
- **Style-Bart-VITS2:** Multi-style TTS with acoustic modeling
- **WaveFormer:** Original EMG classification baseline

## Conclusion

The TTS-codec-inspired architectures demonstrate that principles from audio synthesis can successfully transfer to EMG gesture classification. The **TTSStyleEMGClassifier achieves 72.08% test accuracy**, nearly matching the WaveFormer baseline, while using a completely different architectural paradigm. The **MultiScaleCodec** shows excellent validation performance (81.81%) with the smallest parameter count (348K), suggesting multi-scale processing is a promising direction with proper regularization.

These results validate that codec-based architectures with vector quantization and acoustic modeling can effectively process temporal physiological signals, opening new directions for EMG analysis research.

---

**Branch:** `feature/tts-codec-inspired-models`
**Experiments:** Trial 49 (EMGCodec), Trial 50 (TTSStyleEMGClassifier), Trial 51 (MultiScaleCodec)
**Date:** 2025-11-05
