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


class BasicBlock1D(nn.Module):
    """
    ResNetの基本ブロック（1D版）
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.3):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttentionResNet18(nn.Module):
    """
    ResNet18-1D + Attention機構
    EMG信号用の1次元ResNet18アーキテクチャにAttention機構を追加
    """

    def __init__(self,
                 input_channels: int = 8,
                 num_classes: int = 6,
                 dropout: float = 0.5):
        """
        Args:
            input_channels: 入力チャンネル数（EMGチャンネル数）
            num_classes: 出力クラス数
            dropout: ドロップアウト率
        """
        super(AttentionResNet18, self).__init__()

        self.in_channels = 64
        self.dropout_rate = dropout

        # 初期畳み込み層
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2,
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNetブロック [2, 2, 2, 2] for ResNet18
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Attention機構
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # 分類層
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, out_channels, blocks, stride=1):
        """ResNetレイヤーを構築"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride,
                                   downsample, self.dropout_rate))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels,
                                       dropout=self.dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: shape (batch_size, channels, sequence_length)

        Returns:
            output: shape (batch_size, num_classes)
        """
        # 初期畳み込み
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNetブロック
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (batch, 512, seq_len)

        # Attention機構
        # Transpose for attention: (batch, 512, seq) -> (batch, seq, 512)
        x_t = x.transpose(1, 2)

        # Attention weights
        attention_weights = self.attention(x_t)  # (batch, seq, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum
        context = torch.sum(x_t * attention_weights, dim=1)  # (batch, 512)

        # 分類
        output = self.fc(context.unsqueeze(-1))  # Add dim for FC

        return output


class PositionalEncoding(nn.Module):
    """
    位置エンコーディング（Transformer用）
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 位置エンコーディングを計算
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer Encoderモデル for EMG時系列分類

    Multi-head self-attentionを使用して時系列パターンを学習
    """

    def __init__(self,
                 input_channels: int = 8,
                 num_classes: int = 6,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.5):
        """
        Args:
            input_channels: 入力チャンネル数（EMGチャンネル数）
            num_classes: 出力クラス数
            d_model: Transformerの特徴次元
            nhead: Multi-head attentionのヘッド数
            num_encoder_layers: Encoderレイヤーの数
            dim_feedforward: Feed-forwardネットワークの次元
            dropout: ドロップアウト率
        """
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        # 入力埋め込み層（チャンネル次元をd_modelに変換）
        self.input_embedding = nn.Linear(input_channels, d_model)

        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # (batch, seq, feature)の順序
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # グローバルプーリング（時系列次元を集約）
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
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
        # Transpose: (batch, channels, seq) -> (batch, seq, channels)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)

        # 入力埋め込み
        x = self.input_embedding(x)  # (batch, seq_len, d_model)

        # 位置エンコーディング
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # グローバルプーリング
        # Transpose: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x)  # (batch, d_model, 1)
        x = x.squeeze(-1)  # (batch, d_model)

        # 分類
        x = self.classifier(x)

        return x


class CausalConv1d(nn.Module):
    """
    Causal (non-leaking) 1D convolution
    過去の情報のみを使用する因果的畳み込み
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class ResidualBlock(nn.Module):
    """
    WaveNetの残差ブロック
    Dilated causal convolution + Gated activation + Residual & Skip connections
    """
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()

        # Dilated causal convolution
        self.dilated_conv = CausalConv1d(
            residual_channels, 2 * residual_channels,
            kernel_size, dilation
        )

        # 1x1 convolutions for residual and skip connections
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Dilated causal convolution
        conv_out = self.dilated_conv(x)

        # Gated activation: tanh(W_f * x) * sigmoid(W_g * x)
        tanh_out, sigmoid_out = conv_out.chunk(2, dim=1)
        gated = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)

        gated = self.dropout(gated)

        # Residual connection
        residual = self.residual_conv(gated)
        residual = residual + x

        # Skip connection
        skip = self.skip_conv(gated)

        return residual, skip


class WaveNet(nn.Module):
    """
    WaveNet for EMG時系列分類

    Dilated causal convolutions with residual and skip connections
    Originally designed for audio generation, adapted for time-series classification
    """
    def __init__(self,
                 input_channels: int = 8,
                 num_classes: int = 6,
                 residual_channels: int = 64,
                 skip_channels: int = 128,
                 kernel_size: int = 2,
                 num_layers: int = 10,
                 num_stacks: int = 3,
                 dropout: float = 0.2):
        """
        Args:
            input_channels: 入力チャンネル数（EMGチャンネル数）
            num_classes: 出力クラス数
            residual_channels: 残差接続のチャンネル数
            skip_channels: スキップ接続のチャンネル数
            kernel_size: 畳み込みカーネルサイズ
            num_layers: 各スタックのレイヤー数
            num_stacks: スタック数（同じdilationパターンを繰り返す回数）
            dropout: ドロップアウト率
        """
        super(WaveNet, self).__init__()

        self.input_channels = input_channels
        self.residual_channels = residual_channels

        # Input projection
        self.input_conv = nn.Conv1d(input_channels, residual_channels, 1)

        # Residual blocks with exponentially increasing dilation
        self.residual_blocks = nn.ModuleList()
        for stack in range(num_stacks):
            for layer in range(num_layers):
                dilation = 2 ** layer
                self.residual_blocks.append(
                    ResidualBlock(
                        residual_channels,
                        skip_channels,
                        kernel_size,
                        dilation,
                        dropout
                    )
                )

        # Output layers
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, skip_channels, 1)

        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(skip_channels, 256),
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
        # Input projection
        x = self.input_conv(x)

        # Accumulate skip connections
        skip_connections = []

        # Pass through residual blocks
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # Sum all skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)

        # Output layers with ReLU activations
        out = F.relu(skip_sum)
        out = self.output_conv1(out)
        out = F.relu(out)
        out = self.output_conv2(out)

        # Global pooling
        out = self.global_pool(out)

        # Classification
        out = self.classifier(out)

        return out


def get_model(model_type: str = 'cnn_lstm', **kwargs):
    """
    モデルを取得

    Args:
        model_type: 'cnn_lstm', 'cnn', 'attention_lstm', 'attention_resnet18', 'transformer', or 'wavenet'
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
    elif model_type == 'attention_resnet18':
        return AttentionResNet18(**kwargs)
    elif model_type == 'transformer':
        return TransformerModel(**kwargs)
    elif model_type == 'wavenet':
        return WaveNet(**kwargs)
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
