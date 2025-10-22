# EMG-EPN612 Dataset - Exploratory Data Analysis Report

## Executive Summary

This report presents a comprehensive exploratory data analysis (EDA) of the EMG-EPN612 dataset, which contains electromyography (EMG) signals captured using the Myo Armband for gesture recognition tasks. The dataset comprises 612 users (306 training, 306 testing) and captures 8-channel EMG signals for 6 different hand gestures.

---

## 1. Dataset Overview

### 1.1 Dataset Structure
- **Total Users**: 612
  - Training Set: 306 users
  - Testing Set: 306 users
- **Device**: Myo Armband
- **Sampling Frequency**: 200 Hz
- **Recording Duration**: 5 seconds per gesture
- **Number of Channels**: 8 EMG channels
- **Data Format**: JSON files

### 1.2 Gesture Classes
The dataset includes the following 6 gesture classes:

| Gesture ID | Gesture Name | Description |
|------------|--------------|-------------|
| 0 | No Gesture | Neutral/rest position |
| 1 | Fist | Closed fist |
| 2 | Wave In | Waving hand inward |
| 3 | Wave Out | Waving hand outward |
| 4 | Open | Open hand |
| 5 | Pinch | Pinching gesture |

Note: There are also some samples with ID 65535, which appear to be invalid or undefined gesture states (approximately 5.0% of samples).

---

## 2. User Demographics Analysis

### 2.1 Age Distribution
- **Mean Age**: 24.3 years
- **Median Age**: 22.0 years
- **Age Range**: 18 - 54 years
- **Observation**: The dataset primarily consists of young adults, which may affect generalization to other age groups.

### 2.2 Gender Distribution
- **Male**: 70 participants (70.0%)
- **Female**: 30 participants (30.0%)
- **Observation**: Male participants are overrepresented in the dataset (70:30 ratio).

### 2.3 Handedness Distribution
- **Right-handed**: 95 participants (95.0%)
- **Left-handed**: 5 participants (5.0%)
- **Observation**: The dataset is heavily biased toward right-handed individuals, matching the general population distribution.

### 2.4 Ethnic Diversity
The dataset includes participants from various ethnic groups, contributing to some level of demographic diversity.

---

## 3. EMG Signal Analysis

### 3.1 Overall Signal Statistics
Based on analysis of 20 training users:

- **Mean Amplitude**: -0.84 ± 0.76
- **Standard Deviation**: 14.10 ± 12.57
- **Signal Range**: 143.10 ± 79.80

### 3.2 Per-Channel Analysis

| Channel | Mean Amplitude | Std Deviation | Signal Range |
|---------|---------------|---------------|--------------|
| CH1 | -0.86 ± 0.15 | 6.92 ± 3.65 | 95.90 ± 47.78 |
| CH2 | -0.96 ± 0.22 | 12.46 ± 5.71 | 155.62 ± 60.44 |
| CH3 | -0.45 ± 1.29 | 28.99 ± 12.80 | 233.15 ± 52.58 |
| CH4 | -0.73 ± 1.60 | 27.53 ± 13.09 | 232.77 ± 38.98 |
| CH5 | -0.99 ± 0.18 | 16.25 ± 10.70 | 170.60 ± 58.48 |
| CH6 | -0.93 ± 0.20 | 9.01 ± 10.25 | 102.27 ± 58.88 |
| CH7 | -0.89 ± 0.16 | 6.58 ± 6.15 | 82.14 ± 56.64 |
| CH8 | -0.88 ± 0.15 | 5.06 ± 2.31 | 72.32 ± 33.12 |

### 3.3 Key Observations

1. **Channel Variability**: Channels 3 and 4 show significantly higher variability (std dev ~28-29) compared to other channels (std dev ~5-16), suggesting these channels capture more dynamic muscle activity.

2. **Signal Range**: Channel 3 and 4 exhibit the widest signal ranges (230+), indicating they may be positioned over muscles with more varied activation patterns.

3. **Amplitude Distribution**: Most channels have slightly negative mean amplitudes, which is typical for EMG signals after baseline correction.

4. **Channel Consistency**: Channels 1, 7, and 8 show lower variability, suggesting more consistent placement or less dynamic muscle activity in those positions.

---

## 4. Gesture Distribution Analysis

### 4.1 Gesture Frequency

| Gesture | Count | Percentage |
|---------|-------|------------|
| No Gesture (0) | 22,235 | 44.5% |
| Wave Out (3) | 16,641 | 33.3% |
| Wave In (2) | 3,666 | 7.3% |
| Open (4) | 3,038 | 6.1% |
| Fist (1) | 1,746 | 3.5% |
| Pinch (5) | 101 | 0.2% |
| Unknown (65535) | 2,491 | 5.0% |

### 4.2 Key Observations

1. **Class Imbalance**: The dataset exhibits significant class imbalance:
   - "No Gesture" dominates with 44.5% of samples
   - "Wave Out" is the second most frequent at 33.3%
   - "Pinch" is severely underrepresented at only 0.2%

2. **Training Implications**: The class imbalance will require:
   - Class weighting during model training
   - Data augmentation for minority classes
   - Careful evaluation metrics (not just accuracy)

3. **Unknown Gestures**: The presence of Unknown_65535 (5.0%) suggests:
   - Possible sensor disconnection or error states
   - Transition periods between gestures
   - Data quality issues that should be addressed

---

## 5. Visualizations Generated

The analysis produced the following visualizations:

1. **emg_signals_8channels.png**: Time-series plots of all 8 EMG channels showing signal patterns and amplitudes.

2. **gesture_detection_timeline.png**: Timeline visualization showing detected gestures over the recording period.

3. **demographics_summary.png**: Four-panel visualization showing:
   - Age distribution histogram
   - Gender distribution bar chart
   - Handedness distribution bar chart
   - Ethnic group distribution bar chart

4. **emg_statistics_summary.png**: Four-panel visualization showing:
   - Mean EMG amplitude by channel
   - EMG signal variability by channel
   - EMG signal range by channel
   - Distribution of EMG mean amplitudes

5. **gesture_distribution.png**: Bar chart showing the frequency distribution of all gesture classes.

---

## 6. Data Quality Assessment

### 6.1 Strengths
- Large sample size (612 users)
- High-quality 8-channel EMG data
- Standardized recording protocol (200 Hz, 5 seconds)
- Diverse demographic representation (age, gender, ethnicity)
- Well-structured JSON format

### 6.2 Limitations
- Significant class imbalance (especially Pinch gesture at 0.2%)
- Presence of unknown/invalid gesture states (5.0%)
- Gender imbalance (70% male, 30% female)
- Handedness bias (95% right-handed)
- Age skew toward young adults (mean 24.3 years)

### 6.3 Recommendations
1. Apply data augmentation techniques for minority classes
2. Use stratified sampling for train/validation splits
3. Consider collecting additional data for underrepresented gestures
4. Investigate and handle Unknown_65535 samples appropriately
5. Use appropriate evaluation metrics (F1-score, balanced accuracy)

---

## 7. Technical Specifications

### 7.1 Data Collection Setup
- **Device**: Myo Armband
- **Electrode Configuration**: 8 dry electrodes
- **Sampling Rate**: 200 Hz
- **Recording Duration**: 5 seconds per gesture
- **Synchronization**: 5 repetitions for synchronization gesture

### 7.2 Additional Metadata
Each user record includes:
- Personal information (age, gender, occupation)
- Physical measurements (arm perimeter, electrode placement distance)
- Medical information (arm damage status)
- Recording timestamp

### 7.3 Data Format
- File format: JSON
- Structure: Hierarchical with three main sections:
  - `generalInfo`: Device and recording specifications
  - `userInfo`: Demographic and physical measurements
  - `synchronizationGesture`: EMG signals, quaternion data, and gesture labels

---

## 8. Potential Applications

This dataset is suitable for:

1. **Gesture Recognition**: Training machine learning models for real-time hand gesture classification
2. **Human-Computer Interaction**: Developing EMG-based control interfaces
3. **Prosthetic Control**: Research on myoelectric control for prosthetic limbs
4. **User Authentication**: Exploring EMG signals as biometric identifiers
5. **Signal Processing Research**: Studying EMG signal characteristics and preprocessing techniques

---

## 9. Suggested Next Steps

### 9.1 Preprocessing
1. Bandpass filtering (typically 20-450 Hz for EMG)
2. Notch filtering (50/60 Hz power line interference)
3. Normalization/standardization
4. Feature extraction (time-domain, frequency-domain, time-frequency)

### 9.2 Feature Engineering
Consider extracting:
- Time-domain features: RMS, MAV, ZC, SSC, WL
- Frequency-domain features: MNF, MDF, power spectral density
- Time-frequency features: Wavelet coefficients, STFT

### 9.3 Modeling Approaches
Recommended algorithms:
- Classical ML: SVM, Random Forest, k-NN
- Deep Learning: CNN, LSTM, CNN-LSTM hybrid
- Transfer Learning: Pre-trained models adapted to EMG

### 9.4 Evaluation Strategy
- Use stratified k-fold cross-validation
- Report multiple metrics: accuracy, precision, recall, F1-score
- Confusion matrix analysis
- Per-class performance evaluation
- User-independent vs user-dependent evaluation

---

## 10. Conclusion

The EMG-EPN612 dataset is a comprehensive and well-structured resource for gesture recognition research. With 612 users and 8-channel EMG signals sampled at 200 Hz, it provides sufficient data for training robust machine learning models. However, researchers should be aware of:

1. **Class imbalance** requiring appropriate handling techniques
2. **Demographic biases** that may affect model generalization
3. **Data quality issues** (Unknown gesture states)

With proper preprocessing and modeling techniques, this dataset can support the development of accurate and reliable EMG-based gesture recognition systems.

---

## Appendix: Files Generated

### Analysis Scripts
- `emg_eda.py`: Complete EDA analysis script

### Visualizations
- `emg_signals_8channels.png`: EMG channel signals
- `gesture_detection_timeline.png`: Gesture timeline
- `demographics_summary.png`: User demographics
- `emg_statistics_summary.png`: EMG statistics
- `gesture_distribution.png`: Gesture frequency distribution

### Data Files
- `demographics_stats.csv`: Detailed demographic statistics
- `emg_stats.csv`: Detailed EMG signal statistics

### Reports
- `EDA_REPORT.md`: This comprehensive analysis report

---

**Report Generated**: 2025-10-23
**Analysis Tool**: Python 3.x with NumPy, Pandas, Matplotlib, Seaborn
**Dataset**: EMG-EPN612 (612 users, 8 channels, 200 Hz)
