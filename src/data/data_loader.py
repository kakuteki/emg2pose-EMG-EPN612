"""
EMG-EPN612データセットローダー
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class EMGDataLoader:
    """EMG-EPN612データセットのローダー"""

    def __init__(self, data_path: str, dataset_type: str = 'training'):
        """
        Args:
            data_path: データセットのルートパス
            dataset_type: 'training' or 'testing'
        """
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type

        if dataset_type == 'training':
            self.json_path = self.data_path / 'trainingJSON'
        else:
            self.json_path = self.data_path / 'testingJSON'

        self.gesture_labels = {
            0: "noGesture",
            1: "fist",
            2: "waveIn",
            3: "waveOut",
            4: "open",
            5: "pinch"
        }

    def load_user_data(self, user_folder: Path) -> Optional[Dict]:
        """単一ユーザーのデータを読み込む"""
        json_file = user_folder / f"{user_folder.name}.json"

        if not json_file.exists():
            return None

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return None

    def extract_emg_signals(self, user_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        EMG信号とラベルを抽出

        Returns:
            emg_signals: shape (num_samples, num_channels, sequence_length)
            labels: shape (num_samples,)
        """
        emg_signals_list = []
        labels_list = []

        if 'synchronizationGesture' not in user_data:
            return np.array([]), np.array([])

        sync_data = user_data['synchronizationGesture']['samples']

        for sample_key in sync_data:
            sample = sync_data[sample_key]

            if 'emg' not in sample or 'myoDetection' not in sample:
                continue

            # EMG信号を抽出（8チャンネル）
            emg = sample['emg']
            channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']

            # 全チャンネルが存在するか確認
            if not all(ch in emg for ch in channels):
                continue

            # 8チャンネルのEMG信号を配列に変換
            emg_array = np.array([emg[ch] for ch in channels])  # (8, seq_len)

            # ジェスチャーラベルを抽出
            gesture_labels = np.array(sample['myoDetection'])

            # シーケンスを固定長にセグメント化（ウィンドウサイズ: 200サンプル = 1秒）
            window_size = 200
            stride = 100  # オーバーラップあり

            seq_len = emg_array.shape[1]

            for start_idx in range(0, seq_len - window_size + 1, stride):
                end_idx = start_idx + window_size

                # ウィンドウ内のEMG信号
                window_emg = emg_array[:, start_idx:end_idx]

                # ウィンドウ内の最頻ラベル（モード）を取得
                window_labels = gesture_labels[start_idx:end_idx]
                # 65535（無効な値）を除外
                valid_labels = window_labels[window_labels != 65535]

                if len(valid_labels) == 0:
                    continue

                # 最頻ラベルを採用
                unique, counts = np.unique(valid_labels, return_counts=True)
                majority_label = unique[np.argmax(counts)]

                # ラベルが0-5の範囲内であることを確認
                if 0 <= majority_label <= 5:
                    emg_signals_list.append(window_emg)
                    labels_list.append(majority_label)

        if len(emg_signals_list) == 0:
            return np.array([]), np.array([])

        emg_signals = np.array(emg_signals_list)  # (num_samples, 8, 200)
        labels = np.array(labels_list)  # (num_samples,)

        return emg_signals, labels

    def load_dataset(self, max_users: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        データセット全体を読み込む

        Args:
            max_users: 読み込む最大ユーザー数（Noneの場合は全ユーザー）

        Returns:
            X: EMG信号 shape (num_samples, num_channels, sequence_length)
            y: ラベル shape (num_samples,)
            user_info: ユーザー情報のリスト
        """
        user_folders = sorted([d for d in self.json_path.iterdir() if d.is_dir()])

        if max_users is not None:
            user_folders = user_folders[:max_users]

        all_emg_signals = []
        all_labels = []
        all_user_info = []

        print(f"Loading {self.dataset_type} dataset...")
        for user_folder in tqdm(user_folders):
            user_data = self.load_user_data(user_folder)

            if user_data is None:
                continue

            # EMG信号とラベルを抽出
            emg_signals, labels = self.extract_emg_signals(user_data)

            if len(emg_signals) == 0:
                continue

            all_emg_signals.append(emg_signals)
            all_labels.append(labels)

            # ユーザー情報を保存
            if 'userInfo' in user_data:
                all_user_info.append(user_data['userInfo'])

        # 結合
        X = np.concatenate(all_emg_signals, axis=0)
        y = np.concatenate(all_labels, axis=0)

        print(f"\nLoaded {len(user_folders)} users")
        print(f"Total samples: {len(X)}")
        print(f"Data shape: {X.shape}")
        print(f"Label distribution:")
        for label in range(6):
            count = np.sum(y == label)
            percentage = count / len(y) * 100
            print(f"  {self.gesture_labels[label]} ({label}): {count} ({percentage:.1f}%)")

        return X, y, all_user_info


def create_data_split(X: np.ndarray, y: np.ndarray,
                      test_size: float = 0.2,
                      random_state: int = 42) -> Tuple:
    """
    データを訓練セットと検証セットに分割（層化サンプリング）

    Args:
        X: 特徴量
        y: ラベル
        test_size: テストセットの割合
        random_state: 乱数シード

    Returns:
        X_train, X_val, y_train, y_val
    """
    from sklearn.model_selection import train_test_split

    # Check if stratified split is possible (all classes must have >= 2 samples)
    unique, counts = np.unique(y, return_counts=True)
    min_samples = counts.min()

    if min_samples < 2:
        print(f"Warning: Some classes have < 2 samples (min={min_samples}). Using non-stratified split.")
        stratify_param = None
    else:
        stratify_param = y  # 層化サンプリング

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # 使用例
    data_path = "../../data/raw"

    # トレーニングデータを読み込む
    loader = EMGDataLoader(data_path, dataset_type='training')
    X_train_full, y_train_full, user_info = loader.load_dataset(max_users=10)  # テスト用に10ユーザーのみ

    # 訓練/検証分割
    X_train, X_val, y_train, y_val = create_data_split(X_train_full, y_train_full)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
