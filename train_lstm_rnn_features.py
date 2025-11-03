"""
LSTMとRNNモデルを使った高度な特徴量の学習

Trial 47: LSTM with Advanced Features (464 or 200 dimensions)
Trial 48: RNN with Advanced Features (464 or 200 dimensions)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, 'src')

from data.data_loader import EMGDataLoader, create_data_split
from features.feature_extractor import EMGPreprocessor


# ====================================================================================
# Advanced Feature Extractor (from train_advanced_features.py)
# ====================================================================================

import pywt
from scipy import signal, stats


class AdvancedEMGFeatureExtractor:
    """高度なEMG特徴量抽出器"""

    def __init__(self, sampling_rate=200):
        self.sampling_rate = sampling_rate

    def extract_wavelet_features(self, signal_data):
        """ウェーブレット特徴量"""
        try:
            coeffs = pywt.wavedec(signal_data, 'db4', level=3)
            features = []
            for coeff in coeffs:
                if len(coeff) > 0:
                    features.extend([
                        np.mean(np.abs(coeff)),
                        np.std(coeff),
                        np.max(np.abs(coeff)),
                        stats.skew(coeff) if len(coeff) > 1 else 0.0,
                        stats.kurtosis(coeff) if len(coeff) > 1 else 0.0
                    ])
            return np.array(features[:20])  # 固定長20
        except:
            return np.zeros(20)

    def extract_entropy_features(self, signal_data):
        """エントロピー特徴量"""
        features = []

        # Approximate entropy
        features.append(self._approximate_entropy(signal_data, m=2, r=0.2*np.std(signal_data)))

        # Sample entropy
        features.append(self._sample_entropy(signal_data, m=2, r=0.2*np.std(signal_data)))

        # Spectral entropy
        try:
            freqs, psd = signal.periodogram(signal_data, self.sampling_rate)
            psd_norm = psd / (np.sum(psd) + 1e-10)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            features.append(spectral_entropy)
        except:
            features.append(0.0)

        return np.array(features)

    def _approximate_entropy(self, U, m, r):
        """Approximate Entropy計算"""
        try:
            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

            def _phi(m):
                x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(len(U) - m + 1)]
                C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (len(U) - m + 1.0) for x_i in x]
                return (len(U) - m + 1.0)**(-1) * sum(np.log(C))

            return abs(_phi(m + 1) - _phi(m))
        except:
            return 0.0

    def _sample_entropy(self, U, m, r):
        """Sample Entropy計算"""
        try:
            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

            def _phi(m):
                x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(len(U) - m + 1)]
                B = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1) / (len(U) - m) for x_i in x]
                return sum(B)

            A = _phi(m + 1)
            B = _phi(m)
            return -np.log(A / (B + 1e-10))
        except:
            return 0.0

    def extract_ar_features(self, signal_data, order=4):
        """AR係数"""
        try:
            r = np.correlate(signal_data, signal_data, mode='full')
            r = r[len(r)//2:]

            R = np.zeros((order, order))
            for i in range(order):
                for j in range(order):
                    R[i, j] = r[abs(i-j)]

            ar_coeffs = np.linalg.solve(R, r[1:order+1])
            return ar_coeffs
        except:
            return np.zeros(order)

    def extract_histogram_features(self, signal_data, bins=10):
        """ヒストグラム特徴量"""
        try:
            hist, _ = np.histogram(signal_data, bins=bins, density=True)
            return hist
        except:
            return np.zeros(bins)

    def extract_higher_order_statistics(self, signal_data):
        """高次統計量"""
        features = []

        features.extend([
            np.mean(signal_data),
            np.median(signal_data),
            np.std(signal_data),
            np.var(signal_data),
            stats.skew(signal_data),
            stats.kurtosis(signal_data),
            np.ptp(signal_data)
        ])

        features.extend([
            np.percentile(signal_data, 25),
            np.percentile(signal_data, 75),
            np.percentile(signal_data, 90)
        ])

        features.append(np.percentile(signal_data, 75) - np.percentile(signal_data, 25))

        return np.array(features)

    def extract_time_frequency_features(self, signal_data):
        """時間-周波数特徴量"""
        try:
            f, t, Zxx = signal.stft(signal_data, fs=self.sampling_rate, nperseg=32)
            spectrogram = np.abs(Zxx)

            features = []

            # スペクトログラム統計量
            features.extend([
                np.mean(spectrogram),
                np.std(spectrogram),
                np.max(spectrogram),
                np.min(spectrogram)
            ])

            # 時間軸統計量
            temporal_mean = np.mean(spectrogram, axis=1)
            features.extend([
                np.mean(temporal_mean),
                np.std(temporal_mean),
                np.max(temporal_mean)
            ])

            # 周波数軸統計量
            freq_mean = np.mean(spectrogram, axis=0)
            features.extend([
                np.mean(freq_mean),
                np.std(freq_mean),
                np.max(freq_mean)
            ])

            return np.array(features)
        except:
            return np.zeros(10)

    def extract_advanced_features(self, emg_signal):
        """全特徴量を抽出: 58 features per channel"""
        all_features = []

        all_features.extend(self.extract_wavelet_features(emg_signal))
        all_features.extend(self.extract_entropy_features(emg_signal))
        all_features.extend(self.extract_ar_features(emg_signal))
        all_features.extend(self.extract_histogram_features(emg_signal))
        all_features.extend(self.extract_higher_order_statistics(emg_signal))
        all_features.extend(self.extract_time_frequency_features(emg_signal))

        return np.array(all_features)

    def extract_features_batch(self, emg_signals):
        """
        バッチで特徴量抽出

        Args:
            emg_signals: shape (num_samples, num_channels, sequence_length)

        Returns:
            features: shape (num_samples, num_channels * 58)
        """
        num_samples, num_channels, seq_len = emg_signals.shape
        features_list = []

        for i in tqdm(range(num_samples), desc="Extracting advanced features"):
            sample_features = []
            for ch in range(num_channels):
                channel_signal = emg_signals[i, ch, :]
                channel_features = self.extract_advanced_features(channel_signal)
                sample_features.extend(channel_features)

            features_list.append(sample_features)

        return np.array(features_list)


# ====================================================================================
# LSTM and RNN Models
# ====================================================================================

class LSTMFeatureClassifier(nn.Module):
    """LSTM model for feature-based classification"""

    def __init__(self, input_dim=464, hidden_dim=256, num_layers=2, num_classes=5, dropout=0.3):
        super(LSTMFeatureClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 特徴量をチャンネル単位に再構成 (8 channels × 58 features)
        self.num_channels = 8
        self.features_per_channel = input_dim // self.num_channels

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.features_per_channel,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            out: (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # 特徴量を (batch, seq_len, features) に reshape
        # input_dim → (num_channels, features_per_channel)
        x = x.view(batch_size, self.num_channels, self.features_per_channel)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 最終の隠れ状態を使用
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)

        # Classification
        out = self.fc(last_hidden)

        return out


class RNNFeatureClassifier(nn.Module):
    """Simple RNN model for feature-based classification"""

    def __init__(self, input_dim=464, hidden_dim=256, num_layers=2, num_classes=5, dropout=0.3):
        super(RNNFeatureClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 特徴量をチャンネル単位に再構成
        self.num_channels = 8
        self.features_per_channel = input_dim // self.num_channels

        # RNN layers
        self.rnn = nn.RNN(
            input_size=self.features_per_channel,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity='tanh'
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            out: (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # 特徴量を (batch, seq_len, features) に reshape
        x = x.view(batch_size, self.num_channels, self.features_per_channel)

        # RNN forward
        rnn_out, h_n = self.rnn(x)

        # 最終の隠れ状態を使用
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)

        # Classification
        out = self.fc(last_hidden)

        return out


# ====================================================================================
# PyTorch Dataset
# ====================================================================================

class FeatureDataset(Dataset):
    """Feature-based dataset"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ====================================================================================
# Training and Evaluation Functions
# ====================================================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda'):
    """モデルの訓練"""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        scheduler.step()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

    return model, best_val_acc


def evaluate_model(model, test_loader, device='cuda'):
    """モデルの評価"""

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds) * 100

    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=['No Gesture', 'Fist', 'Wave In', 'Wave Out', 'Open']
    ))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    return test_acc, all_preds, all_labels


# ====================================================================================
# Main Function
# ====================================================================================

def main():
    print("=" * 80)
    print("LSTM/RNN with Advanced Features Training")
    print("=" * 80)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # データロード
    print("\nLoading data...")
    train_loader_data = EMGDataLoader('.', dataset_type='training')
    test_loader_data = EMGDataLoader('.', dataset_type='testing')

    X_train, y_train, _ = train_loader_data.load_dataset()
    X_test, y_test, _ = test_loader_data.load_dataset()

    # Pinchクラスを除外
    print("\nExcluding Pinch class (label 5)...")
    train_mask = y_train != 5
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    test_mask = y_test != 5
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")

    # Train/Val分割
    X_train_split, X_val, y_train_split, y_val = create_data_split(
        X_train, y_train, test_size=0.2, random_state=789
    )

    print(f"\nAfter split:")
    print(f"  Train: {X_train_split.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # 前処理
    print("\nPreprocessing data...")
    preprocessor = EMGPreprocessor(sampling_rate=200)
    X_train_preprocessed = preprocessor.preprocess(X_train_split)
    X_val_preprocessed = preprocessor.preprocess(X_val)
    X_test_preprocessed = preprocessor.preprocess(X_test)

    # 特徴量抽出
    print("\n" + "=" * 80)
    print("Advanced Feature Extraction")
    print("=" * 80)

    feature_extractor = AdvancedEMGFeatureExtractor(sampling_rate=200)

    X_train_features = feature_extractor.extract_features_batch(X_train_preprocessed)
    X_val_features = feature_extractor.extract_features_batch(X_val_preprocessed)
    X_test_features = feature_extractor.extract_features_batch(X_test_preprocessed)

    print(f"\nExtracted feature shapes:")
    print(f"  Train: {X_train_features.shape}")
    print(f"  Val: {X_val_features.shape}")
    print(f"  Test: {X_test_features.shape}")

    # NaN/Inf処理
    print("\nHandling NaN/Inf values...")
    X_train_features = np.nan_to_num(X_train_features, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_features = np.nan_to_num(X_val_features, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_features = np.nan_to_num(X_test_features, nan=0.0, posinf=0.0, neginf=0.0)

    # DataLoaderを作成
    batch_size = 128

    train_dataset = FeatureDataset(X_train_features, y_train_split)
    val_dataset = FeatureDataset(X_val_features, y_val)
    test_dataset = FeatureDataset(X_test_features, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {}

    # Trial 47: LSTM
    print("\n" + "=" * 80)
    print("Trial 47: LSTM with Advanced Features")
    print("=" * 80)

    input_dim = X_train_features.shape[1]
    print(f"\nInput dimension: {input_dim}")

    lstm_model = LSTMFeatureClassifier(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=2,
        num_classes=5,
        dropout=0.3
    )

    print(f"\nModel architecture:")
    print(lstm_model)
    print(f"\nTotal parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    lstm_model, lstm_val_acc = train_model(
        lstm_model, train_loader, val_loader,
        epochs=100, lr=0.001, device=device
    )

    lstm_test_acc, _, _ = evaluate_model(lstm_model, test_loader, device=device)

    results['trial47_lstm'] = {
        'method': 'LSTM',
        'input_dim': input_dim,
        'hidden_dim': 256,
        'num_layers': 2,
        'val_acc': lstm_val_acc,
        'test_acc': lstm_test_acc
    }

    # モデル保存
    os.makedirs('results/trial47_lstm', exist_ok=True)
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'val_acc': lstm_val_acc,
        'test_acc': lstm_test_acc
    }, 'results/trial47_lstm/best_model.pth')

    # Trial 48: RNN
    print("\n" + "=" * 80)
    print("Trial 48: RNN with Advanced Features")
    print("=" * 80)

    rnn_model = RNNFeatureClassifier(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=2,
        num_classes=5,
        dropout=0.3
    )

    print(f"\nModel architecture:")
    print(rnn_model)
    print(f"\nTotal parameters: {sum(p.numel() for p in rnn_model.parameters()):,}")

    rnn_model, rnn_val_acc = train_model(
        rnn_model, train_loader, val_loader,
        epochs=100, lr=0.001, device=device
    )

    rnn_test_acc, _, _ = evaluate_model(rnn_model, test_loader, device=device)

    results['trial48_rnn'] = {
        'method': 'RNN',
        'input_dim': input_dim,
        'hidden_dim': 256,
        'num_layers': 2,
        'val_acc': rnn_val_acc,
        'test_acc': rnn_test_acc
    }

    # モデル保存
    os.makedirs('results/trial48_rnn', exist_ok=True)
    torch.save({
        'model_state_dict': rnn_model.state_dict(),
        'val_acc': rnn_val_acc,
        'test_acc': rnn_test_acc
    }, 'results/trial48_rnn/best_model.pth')

    # 結果サマリー
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)

    for trial_name, trial_results in results.items():
        print(f"\n{trial_name}:")
        for key, value in trial_results.items():
            print(f"  {key}: {value}")

    # JSON形式で保存
    os.makedirs('results', exist_ok=True)
    with open('results/lstm_rnn_features_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n結果を保存: results/lstm_rnn_features_results.json")

    # ベースラインとの比較
    print("\n" + "=" * 80)
    print("Comparison with WaveFormer Baseline (72.33%) and Trial 46 (72.24%)")
    print("=" * 80)

    baseline_waveformer = 72.33
    baseline_trial46 = 72.24

    for trial_name, trial_results in results.items():
        test_acc = trial_results['test_acc']
        diff_waveformer = test_acc - baseline_waveformer
        diff_trial46 = test_acc - baseline_trial46

        status = "✓ BETTER" if diff_waveformer > 0 else "→ SAME" if diff_waveformer == 0 else "✗ WORSE"

        print(f"\n{trial_name}:")
        print(f"  Test Acc: {test_acc:.2f}%")
        print(f"  vs WaveFormer: {diff_waveformer:+.2f}% {status}")
        print(f"  vs Trial 46: {diff_trial46:+.2f}%")


if __name__ == "__main__":
    main()
