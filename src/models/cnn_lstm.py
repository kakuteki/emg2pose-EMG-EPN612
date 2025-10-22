"""
CNN-LSTM ハイブリッドモデル for EMG gesture recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    """
    CNN-LSTMハイブリッドモデル

    CNNで空間的特徴を抽出し、LSTMで時系列パターンを学習
    """

    def __init__(self,
                 input_channels: int = 8,
                 num_classes: int = 6,
                 cnn_channels: list = [32, 64, 128],
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 dropout: float = 0.5):
        """
        Args:
            input_channels: 入力チャンネル数（EMGチャンネル数）
            num_classes: 出力クラス数
            cnn_channels: CNNの各層のチャンネル数
            lstm_hidden_size: LSTMの隠れ層サイズ
            lstm_num_layers: LSTMの層数
            dropout: ドロップアウト率
        """
        super(CNNLSTM, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels

        for out_channels in cnn_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: shape (batch_size, channels, sequence_length)

        Returns:
            output: shape (batch_size, num_classes)
        """
        # CNN feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Transpose for LSTM: (batch, channels, seq) -> (batch, seq, channels)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last output
        x = lstm_out[:, -1, :]

        # Fully connected
        x = self.fc(x)

        return x


class SimpleCNN(nn.Module):
    """
    シンプルな1D CNNモデル（比較用）
    """

    def __init__(self,
                 input_channels: int = 8,
                 num_classes: int = 6,
                 dropout: float = 0.5):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Conv Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Conv Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Conv Block 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AttentionLSTM(nn.Module):
    """
    Attention機構付きLSTMモデル
    """

    def __init__(self,
                 input_channels: int = 8,
                 num_classes: int = 6,
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 dropout: float = 0.5):
        super(AttentionLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Transpose: (batch, channels, seq) -> (batch, seq, channels)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention weights
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)

        # Classification
        output = self.classifier(context)

        return output


def get_model(model_type: str = 'cnn_lstm', **kwargs):
    """
    モデルを取得

    Args:
        model_type: 'cnn_lstm', 'cnn', or 'attention_lstm'
        **kwargs: モデル固有のパラメータ

    Returns:
        model: PyTorchモデル
    """
    if model_type == 'cnn_lstm':
        return CNNLSTM(**kwargs)
    elif model_type == 'cnn':
        return SimpleCNN(**kwargs)
    elif model_type == 'attention_lstm':
        return AttentionLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # テスト
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ダミーデータ
    batch_size = 32
    channels = 8
    seq_length = 200
    num_classes = 6

    x = torch.randn(batch_size, channels, seq_length).to(device)

    # CNN-LSTMモデル
    print("Testing CNN-LSTM model...")
    model = CNNLSTM(input_channels=channels, num_classes=num_classes).to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # CNNモデル
    print("\nTesting Simple CNN model...")
    model = SimpleCNN(input_channels=channels, num_classes=num_classes).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Attention LSTMモデル
    print("\nTesting Attention LSTM model...")
    model = AttentionLSTM(input_channels=channels, num_classes=num_classes).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
