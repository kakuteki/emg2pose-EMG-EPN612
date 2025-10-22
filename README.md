# EMG-EPN612 Dataset Analysis

This repository contains exploratory data analysis (EDA) tools and results for the EMG-EPN612 dataset.

## Overview

The EMG-EPN612 dataset contains electromyography (EMG) signals captured using the Myo Armband for gesture recognition tasks. This repository provides analysis scripts and comprehensive EDA results.

### Dataset Specifications
- **Total Users**: 612 (306 training, 306 testing)
- **Device**: Myo Armband
- **Sampling Frequency**: 200 Hz
- **Recording Duration**: 5 seconds per gesture
- **EMG Channels**: 8 channels
- **Gesture Classes**: 6 (fist, waveIn, waveOut, open, pinch, noGesture)

## Repository Structure

```
.
├── emg_eda.py                      # Main EDA analysis script
├── eda_results/                    # Analysis results directory
│   ├── EDA_REPORT.md              # Comprehensive analysis report
│   ├── emg_signals_8channels.png  # 8-channel EMG signal visualization
│   ├── gesture_detection_timeline.png  # Gesture timeline
│   ├── demographics_summary.png   # User demographics visualization
│   ├── emg_statistics_summary.png # EMG statistics visualization
│   ├── gesture_distribution.png   # Gesture frequency distribution
│   ├── demographics_stats.csv     # Detailed demographic statistics
│   └── emg_stats.csv             # Detailed EMG statistics
├── .gitignore                     # Git ignore file (excludes dataset)
└── README.md                      # This file
```

## Dataset Download

The actual dataset files (trainingJSON/ and testingJSON/) are not included in this repository due to their size.

You can download the EMG-EPN612 dataset from:
- [Original Dataset Source]
- Place the downloaded `trainingJSON/` and `testingJSON/` folders in the root directory

## Requirements

```bash
pip install numpy pandas matplotlib seaborn
```

Python 3.7+ required.

## Usage

1. Download and place the dataset in the root directory
2. Run the EDA script:

```bash
python emg_eda.py
```

The script will:
- Analyze dataset structure and demographics
- Compute EMG signal statistics
- Generate visualizations
- Create a comprehensive report

All results will be saved in the `eda_results/` directory.

## Key Findings

### User Demographics
- **Age**: Mean 24.3 years (range: 18-54)
- **Gender**: 70% male, 30% female
- **Handedness**: 95% right-handed, 5% left-handed

### EMG Signal Characteristics
- **Overall Mean Amplitude**: -0.84 ± 0.76
- **Standard Deviation**: 14.10 ± 12.57
- **Signal Range**: 143.10 ± 79.80
- Channels 3 & 4 show highest variability (std ~28-29)

### Gesture Distribution
- **No Gesture**: 44.5%
- **Wave Out**: 33.3%
- **Wave In**: 7.3%
- **Open**: 6.1%
- **Fist**: 3.5%
- **Pinch**: 0.2%

⚠️ **Note**: Significant class imbalance exists, requiring appropriate handling techniques.

## Analysis Features

The EDA script provides:

1. **Dataset Overview**
   - User count and distribution
   - Recording specifications

2. **Demographic Analysis**
   - Age, gender, handedness distributions
   - Ethnic diversity

3. **EMG Signal Analysis**
   - Per-channel statistics
   - Signal amplitude and variability
   - Gesture frequency distribution

4. **Visualizations**
   - 8-channel EMG time-series plots
   - Gesture detection timeline
   - Statistical summary plots
   - Demographic distribution charts

5. **Statistical Reports**
   - CSV files with detailed statistics
   - Comprehensive markdown report

## Recommendations for ML Development

1. **Preprocessing**
   - Bandpass filtering (20-450 Hz)
   - Notch filtering (50/60 Hz)
   - Normalization/standardization

2. **Feature Engineering**
   - Time-domain: RMS, MAV, ZC, SSC, WL
   - Frequency-domain: MNF, MDF, power spectral density
   - Time-frequency: Wavelet coefficients

3. **Model Training**
   - Address class imbalance (class weights, oversampling)
   - Use stratified k-fold cross-validation
   - Consider: SVM, Random Forest, CNN, LSTM

4. **Evaluation**
   - Use F1-score and balanced accuracy
   - Analyze confusion matrix
   - Report per-class performance

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

Please refer to the original EMG-EPN612 dataset license terms.

## Citation

If you use this analysis in your research, please cite the original dataset:

```
[Add appropriate citation for EMG-EPN612 dataset]
```

## Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: 2025-10-23
