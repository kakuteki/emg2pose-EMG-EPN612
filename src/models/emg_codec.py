"""
EMG Codec Models inspired by TTS/Audio Codecs (SoundStream, EnCodec, VITS)

TTSパイプラインをEMG分類に適用:
  TTS:                          EMG:
  テキスト                      → EMG生信号
  ↓                             ↓
  言語処理                      → 前処理・正規化
  ↓                             ↓
  音韻/言語特徴                 → 時間・周波数特徴
  ↓                             ↓
  音響モデル                    → 潜在表現学習
  ↓                             ↓
  メルスペクトログラム          → 中間表現
  ↓                             ↓
  ボコーダー                    → デコーダー
  ↓                             ↓
  音声波形                      → 分類出力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer (VQ-VAE style)
    連続的な潜在表現を離散コードブックに量子化
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # コードブック
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        # inputs: [B, C, T]
        B, C, T = inputs.shape

        # [B, C, T] -> [B, T, C]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # Flatten
        flat_input = inputs.view(-1, self.embedding_dim)

        # コードブックとの距離計算
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # 最近傍のコードを選択
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 量子化
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Loss計算
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # [B, T, C] -> [B, C, T]
        quantized = quantized.permute(0, 2, 1).contiguous()

        return quantized, loss, encoding_indices.view(B, T)


class ResidualBlock(nn.Module):
    """残差ブロック"""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class EMGEncoder(nn.Module):
    """
    EMG Encoder (SoundStream/EnCodec inspired)
    生EMG信号を圧縮された潜在表現に変換
    """
    def __init__(self, in_channels=8, hidden_dim=128, latent_dim=64, num_residual_blocks=4):
        super().__init__()

        # 初期畳み込み (EMG前処理相当)
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim//2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU()
        )

        # ダウンサンプリング層 (特徴抽出相当)
        self.downsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ),
        ])

        # 残差ブロック (音響特徴抽出相当)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dilation=2**i) for i in range(num_residual_blocks)
        ])

        # 潜在表現への投影
        self.to_latent = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, 8, 200]
        x = self.input_conv(x)

        for downsample in self.downsample_blocks:
            x = downsample(x)

        for residual in self.residual_blocks:
            x = residual(x)

        latent = self.to_latent(x)
        return latent


class EMGDecoder(nn.Module):
    """
    EMG Decoder (Vocoder inspired)
    潜在表現から分類に必要な特徴を復元
    """
    def __init__(self, latent_dim=64, hidden_dim=128, out_channels=8, num_residual_blocks=4):
        super().__init__()

        # 潜在表現から特徴へ
        self.from_latent = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)

        # 残差ブロック
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dilation=2**i) for i in range(num_residual_blocks)
        ])

        # アップサンプリング層
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim//2),
                nn.ReLU()
            ),
        ])

        # 出力層
        self.output_conv = nn.Conv1d(hidden_dim//2, out_channels, kernel_size=7, padding=3)

    def forward(self, latent):
        x = self.from_latent(latent)

        for residual in self.residual_blocks:
            x = residual(x)

        for upsample in self.upsample_blocks:
            x = upsample(x)

        out = self.output_conv(x)
        return out


class EMGCodec(nn.Module):
    """
    Complete EMG Codec (VQ-VAE style)
    Encoder -> VQ -> Decoder
    """
    def __init__(self, in_channels=8, hidden_dim=128, latent_dim=64,
                 num_embeddings=512, num_classes=5):
        super().__init__()

        self.encoder = EMGEncoder(in_channels, hidden_dim, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = EMGDecoder(latent_dim, hidden_dim, in_channels)

        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, return_reconstruction=False):
        # Encode
        latent = self.encoder(x)

        # Quantize
        quantized, vq_loss, indices = self.vq(latent)

        # Classify
        logits = self.classifier(quantized)

        if return_reconstruction:
            # Decode (for reconstruction loss)
            reconstruction = self.decoder(quantized)
            return logits, vq_loss, reconstruction

        return logits, vq_loss


class AcousticModel(nn.Module):
    """
    Acoustic Model (TTS inspired)
    EMG特徴から中間表現（メルスペクトログラム相当）を生成
    """
    def __init__(self, in_channels=8, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()

        # 入力投影
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)

        # Transformer Encoder (音響モデリング)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(hidden_dim)

    def forward(self, x):
        # x: [B, C, T]
        x = self.input_proj(x)

        # [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1)

        # 位置エンコーディング
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)

        # [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)

        return x


class PositionalEncoding(nn.Module):
    """位置エンコーディング"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, T, C]
        return x + self.pe[:, :x.size(1), :]


class TTSStyleEMGClassifier(nn.Module):
    """
    Complete TTS-style pipeline for EMG classification

    Pipeline:
    1. Preprocessing (analogous to text preprocessing)
    2. Feature extraction (analogous to phoneme features)
    3. Acoustic model (analogous to acoustic model in TTS)
    4. Intermediate representation (analogous to mel-spectrogram)
    5. Decoder (analogous to vocoder)
    6. Classification output
    """
    def __init__(self, in_channels=8, hidden_dim=256, latent_dim=128,
                 num_embeddings=512, num_classes=5):
        super().__init__()

        # Stage 1: Preprocessing & Feature Extraction (前処理・特徴抽出)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim//2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Stage 2: Acoustic Model (音響モデル)
        self.acoustic_model = AcousticModel(hidden_dim, hidden_dim, num_heads=8, num_layers=4)

        # Stage 3: Intermediate Representation (中間表現生成)
        self.to_latent = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

        # Stage 4: Vector Quantization (離散化)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)

        # Stage 5: Decoder (デコーダー)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=1),
            ResidualBlock(hidden_dim, dilation=1),
            ResidualBlock(hidden_dim, dilation=2),
            ResidualBlock(hidden_dim, dilation=4),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Stage 6: Classification Head (分類ヘッド)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, x):
        # Stage 1: Feature extraction
        features = self.feature_extractor(x)

        # Stage 2: Acoustic modeling
        acoustic_features = self.acoustic_model(features)

        # Stage 3: Latent representation
        latent = self.to_latent(acoustic_features)

        # Stage 4: Quantization
        quantized, vq_loss, _ = self.vq(latent)

        # Stage 5: Decode
        decoded = self.decoder(quantized)

        # Stage 6: Classify
        logits = self.classifier(decoded)

        return logits, vq_loss


class MultiScaleCodec(nn.Module):
    """
    Multi-scale Codec inspired by multi-resolution TTS models
    複数のスケールでEMG信号をエンコード
    """
    def __init__(self, in_channels=8, hidden_dim=128, num_classes=5):
        super().__init__()

        # 3つの異なるスケール
        self.scale1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim)
        )

        self.scale2 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim)
        )

        self.scale3 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim)
        )

        # マルチスケール統合
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim * 2, kernel_size=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # 各スケールで処理
        s1 = self.scale1(x)  # [B, H, 200]
        s2 = self.scale2(x)  # [B, H, 100]
        s3 = self.scale3(x)  # [B, H, 50]

        # スケール2,3をスケール1に合わせる
        s2 = F.interpolate(s2, size=s1.size(-1), mode='linear', align_corners=False)
        s3 = F.interpolate(s3, size=s1.size(-1), mode='linear', align_corners=False)

        # 統合
        fused = torch.cat([s1, s2, s3], dim=1)
        features = self.fusion(fused)

        # 分類
        logits = self.classifier(features)

        return logits


def get_model(model_type='codec', **kwargs):
    """モデルファクトリー"""
    models = {
        'codec': EMGCodec,
        'tts_style': TTSStyleEMGClassifier,
        'multiscale': MultiScaleCodec,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](**kwargs)


if __name__ == "__main__":
    # テスト
    batch_size = 4
    in_channels = 8
    seq_len = 200
    num_classes = 5

    x = torch.randn(batch_size, in_channels, seq_len)

    print("Testing EMGCodec...")
    model1 = EMGCodec(in_channels=in_channels, num_classes=num_classes)
    logits1, vq_loss1 = model1(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {logits1.shape}")
    print(f"  VQ Loss: {vq_loss1.item():.4f}")

    print("\nTesting TTSStyleEMGClassifier...")
    model2 = TTSStyleEMGClassifier(in_channels=in_channels, num_classes=num_classes)
    logits2, vq_loss2 = model2(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {logits2.shape}")
    print(f"  VQ Loss: {vq_loss2.item():.4f}")

    print("\nTesting MultiScaleCodec...")
    model3 = MultiScaleCodec(in_channels=in_channels, num_classes=num_classes)
    logits3 = model3(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {logits3.shape}")

    print("\nAll tests passed!")
