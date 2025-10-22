"""
EMG信号からの特徴量抽出
"""
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Optional


class EMGFeatureExtractor:
    """EMG信号から時間領域・周波数領域の特徴量を抽出"""

    def __init__(self, sampling_rate: int = 200):
        """
        Args:
            sampling_rate: サンプリング周波数 (Hz)
        """
        self.sampling_rate = sampling_rate

    def extract_time_domain_features(self, emg_signal: np.ndarray) -> np.ndarray:
        """
        時間領域の特徴量を抽出

        Args:
            emg_signal: shape (num_channels, sequence_length)

        Returns:
            features: shape (num_channels * num_features,)
        """
        features_list = []

        for channel in range(emg_signal.shape[0]):
            ch_signal = emg_signal[channel]

            # 1. MAV - Mean Absolute Value (平均絶対値)
            mav = np.mean(np.abs(ch_signal))

            # 2. RMS - Root Mean Square (二乗平均平方根)
            rms = np.sqrt(np.mean(ch_signal ** 2))

            # 3. VAR - Variance (分散)
            var = np.var(ch_signal)

            # 4. WL - Waveform Length (波形長)
            wl = np.sum(np.abs(np.diff(ch_signal)))

            # 5. ZC - Zero Crossing (ゼロクロッシング)
            zc = self._zero_crossings(ch_signal)

            # 6. SSC - Slope Sign Change (傾き符号変化)
            ssc = self._slope_sign_changes(ch_signal)

            # 7. WAMP - Willison Amplitude (ウィリソン振幅)
            wamp = self._willison_amplitude(ch_signal)

            # 8. Skewness (歪度)
            skewness = skew(ch_signal)

            # 9. Kurtosis (尖度)
            kurt = kurtosis(ch_signal)

            # 10. Peak-to-Peak Amplitude (最大-最小振幅)
            ptp = np.ptp(ch_signal)

            channel_features = [mav, rms, var, wl, zc, ssc, wamp,
                              skewness, kurt, ptp]
            features_list.extend(channel_features)

        return np.array(features_list)

    def extract_frequency_domain_features(self, emg_signal: np.ndarray) -> np.ndarray:
        """
        周波数領域の特徴量を抽出

        Args:
            emg_signal: shape (num_channels, sequence_length)

        Returns:
            features: shape (num_channels * num_features,)
        """
        features_list = []

        for channel in range(emg_signal.shape[0]):
            ch_signal = emg_signal[channel]

            # FFTでパワースペクトルを計算
            freqs, psd = signal.welch(ch_signal, fs=self.sampling_rate,
                                     nperseg=min(256, len(ch_signal)))

            # 1. MNF - Mean Frequency (平均周波数)
            mnf = np.sum(freqs * psd) / np.sum(psd)

            # 2. MDF - Median Frequency (中央周波数)
            cumsum_psd = np.cumsum(psd)
            mdf = freqs[np.where(cumsum_psd >= cumsum_psd[-1] / 2)[0][0]]

            # 3. Total Power (総パワー)
            total_power = np.sum(psd)

            # 4. Peak Frequency (ピーク周波数)
            peak_freq = freqs[np.argmax(psd)]

            # 5. Spectral Entropy (スペクトルエントロピー)
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

            channel_features = [mnf, mdf, total_power, peak_freq, spectral_entropy]
            features_list.extend(channel_features)

        return np.array(features_list)

    def extract_all_features(self, emg_signal: np.ndarray) -> np.ndarray:
        """
        時間領域と周波数領域の特徴量を全て抽出

        Args:
            emg_signal: shape (num_channels, sequence_length)

        Returns:
            features: shape (num_features,)
        """
        time_features = self.extract_time_domain_features(emg_signal)
        freq_features = self.extract_frequency_domain_features(emg_signal)

        all_features = np.concatenate([time_features, freq_features])

        return all_features

    def extract_features_batch(self, emg_signals: np.ndarray,
                               feature_type: str = 'all') -> np.ndarray:
        """
        バッチデータから特徴量を抽出

        Args:
            emg_signals: shape (num_samples, num_channels, sequence_length)
            feature_type: 'time', 'frequency', or 'all'

        Returns:
            features: shape (num_samples, num_features)
        """
        features_list = []

        for i in range(emg_signals.shape[0]):
            signal_sample = emg_signals[i]

            if feature_type == 'time':
                features = self.extract_time_domain_features(signal_sample)
            elif feature_type == 'frequency':
                features = self.extract_frequency_domain_features(signal_sample)
            else:  # 'all'
                features = self.extract_all_features(signal_sample)

            features_list.append(features)

        return np.array(features_list)

    # ヘルパー関数
    def _zero_crossings(self, signal: np.ndarray, threshold: float = 0.01) -> int:
        """ゼロクロッシングの数を計算"""
        signs = np.sign(signal)
        signs[signs == 0] = -1
        zc = np.sum(np.abs(np.diff(signs)) >= 2)
        return zc

    def _slope_sign_changes(self, signal: np.ndarray, threshold: float = 0.01) -> int:
        """傾き符号変化の数を計算"""
        diff_signal = np.diff(signal)
        signs = np.sign(diff_signal)
        signs[signs == 0] = -1
        ssc = np.sum(np.abs(np.diff(signs)) >= 2)
        return ssc

    def _willison_amplitude(self, signal: np.ndarray, threshold: float = 0.01) -> int:
        """ウィリソン振幅を計算"""
        diff_signal = np.abs(np.diff(signal))
        wamp = np.sum(diff_signal > threshold)
        return wamp


class EMGPreprocessor:
    """EMG信号の前処理"""

    def __init__(self, sampling_rate: int = 200):
        """
        Args:
            sampling_rate: サンプリング周波数 (Hz)
        """
        self.sampling_rate = sampling_rate

    def bandpass_filter(self, emg_signal: np.ndarray,
                       lowcut: float = 20.0,
                       highcut: float = 90.0,
                       order: int = 4) -> np.ndarray:
        """
        バンドパスフィルタを適用

        Args:
            emg_signal: shape (..., sequence_length)
            lowcut: 低域カットオフ周波数 (Hz)
            highcut: 高域カットオフ周波数 (Hz)
            order: フィルタ次数

        Returns:
            filtered_signal: shape (..., sequence_length)
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = signal.butter(order, [low, high], btype='band')

        # 最後の次元に沿ってフィルタリング
        filtered_signal = signal.filtfilt(b, a, emg_signal, axis=-1)

        return filtered_signal

    def notch_filter(self, emg_signal: np.ndarray,
                    freq: float = 50.0,
                    quality_factor: float = 30.0) -> np.ndarray:
        """
        ノッチフィルタを適用（電源ノイズ除去）

        Args:
            emg_signal: shape (..., sequence_length)
            freq: ノッチ周波数 (Hz)
            quality_factor: Q値

        Returns:
            filtered_signal: shape (..., sequence_length)
        """
        nyquist = 0.5 * self.sampling_rate
        freq_norm = freq / nyquist

        b, a = signal.iirnotch(freq_norm, quality_factor)

        filtered_signal = signal.filtfilt(b, a, emg_signal, axis=-1)

        return filtered_signal

    def normalize(self, emg_signal: np.ndarray,
                 method: str = 'zscore') -> np.ndarray:
        """
        信号を正規化

        Args:
            emg_signal: shape (num_channels, sequence_length)
            method: 'zscore' or 'minmax'

        Returns:
            normalized_signal: shape (num_channels, sequence_length)
        """
        if method == 'zscore':
            # Zスコア正規化（チャンネルごと）
            mean = np.mean(emg_signal, axis=-1, keepdims=True)
            std = np.std(emg_signal, axis=-1, keepdims=True)
            normalized_signal = (emg_signal - mean) / (std + 1e-8)

        elif method == 'minmax':
            # Min-Max正規化（チャンネルごと）
            min_val = np.min(emg_signal, axis=-1, keepdims=True)
            max_val = np.max(emg_signal, axis=-1, keepdims=True)
            normalized_signal = (emg_signal - min_val) / (max_val - min_val + 1e-8)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized_signal

    def preprocess(self, emg_signal: np.ndarray,
                  apply_bandpass: bool = True,
                  apply_notch: bool = True,
                  normalize: bool = True) -> np.ndarray:
        """
        完全な前処理パイプライン

        Args:
            emg_signal: shape (..., num_channels, sequence_length)
            apply_bandpass: バンドパスフィルタを適用するか
            apply_notch: ノッチフィルタを適用するか
            normalize: 正規化するか

        Returns:
            processed_signal: shape (..., num_channels, sequence_length)
        """
        processed_signal = emg_signal.copy()

        if apply_bandpass:
            processed_signal = self.bandpass_filter(processed_signal)

        if apply_notch:
            processed_signal = self.notch_filter(processed_signal)

        if normalize:
            # バッチデータの場合は各サンプルを正規化
            if processed_signal.ndim == 3:
                for i in range(processed_signal.shape[0]):
                    processed_signal[i] = self.normalize(processed_signal[i])
            else:
                processed_signal = self.normalize(processed_signal)

        return processed_signal


if __name__ == "__main__":
    # 使用例
    # ダミーデータで動作確認
    dummy_signal = np.random.randn(8, 200)  # 8チャンネル, 200サンプル

    # 特徴量抽出
    extractor = EMGFeatureExtractor(sampling_rate=200)
    time_features = extractor.extract_time_domain_features(dummy_signal)
    freq_features = extractor.extract_frequency_domain_features(dummy_signal)
    all_features = extractor.extract_all_features(dummy_signal)

    print(f"Time domain features shape: {time_features.shape}")
    print(f"Frequency domain features shape: {freq_features.shape}")
    print(f"All features shape: {all_features.shape}")

    # 前処理
    preprocessor = EMGPreprocessor(sampling_rate=200)
    processed_signal = preprocessor.preprocess(dummy_signal)
    print(f"\nProcessed signal shape: {processed_signal.shape}")
