"""
EMG-Diffusion Model for Gesture Classification

Architecture:
    EMG Signal → Transformer Feature Extractor → Features
    Features → Diffusion Model → Gesture Classification

Inspired by diffusion probabilistic models and their application to classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]


class TransformerFeatureExtractor(nn.Module):
    """
    Stage 1: Transformer-based feature extraction from EMG signals

    Architecture:
        EMG Input (B, 8, T) → Conv Embedding → Transformer Encoder → Feature Vector

    Args:
        in_channels: Number of EMG channels (default: 8)
        d_model: Transformer model dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of Transformer layers (default: 6)
        dim_feedforward: Feedforward network dimension (default: 1024)
        dropout: Dropout rate (default: 0.1)
        feature_dim: Output feature dimension (default: 128)
    """
    def __init__(self, in_channels=8, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.1, feature_dim=128):
        super().__init__()

        self.in_channels = in_channels
        self.d_model = d_model
        self.feature_dim = feature_dim

        # Input embedding: Conv1d to project EMG channels to d_model
        self.input_embedding = nn.Sequential(
            nn.Conv1d(in_channels, d_model // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(d_model, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: EMG input (batch_size, in_channels, seq_len)

        Returns:
            features: Extracted features (batch_size, feature_dim)
        """
        batch_size = x.size(0)

        # Embedding: (B, C, T) -> (B, d_model, T)
        x = self.input_embedding(x)

        # Reshape for Transformer: (B, d_model, T) -> (T, B, d_model)
        x = x.permute(2, 0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (T, B, d_model)

        # Global average pooling over time dimension
        x = x.mean(dim=0)  # (B, d_model)

        # Project to feature dimension
        features = self.feature_projection(x)  # (B, feature_dim)

        return features


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps"""
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        """
        Args:
            timesteps: (batch_size,) tensor of timestep indices

        Returns:
            embeddings: (batch_size, embedding_dim) time embeddings
        """
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.embedding_dim % 2 == 1:  # Handle odd embedding_dim
            emb = F.pad(emb, (0, 1))

        return emb


class DiffusionBlock(nn.Module):
    """Diffusion denoising block with time conditioning"""
    def __init__(self, feature_dim, time_dim, hidden_dim):
        super().__init__()

        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Time embedding processing
        self.time_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU()
        )

        # Combined processing
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x, time_emb):
        """
        Args:
            x: Features (batch_size, feature_dim)
            time_emb: Time embeddings (batch_size, time_dim)

        Returns:
            output: Denoised features (batch_size, feature_dim)
        """
        h_feat = self.feature_net(x)
        h_time = self.time_net(time_emb)
        h = h_feat + h_time
        output = self.combined_net(h)
        return output + x  # Residual connection


class DiffusionClassifier(nn.Module):
    """
    Stage 2: Diffusion-based classifier

    Uses denoising diffusion process for robust classification.
    The model learns to denoise features progressively from noise to clean predictions.

    Args:
        feature_dim: Input feature dimension (default: 128)
        num_classes: Number of gesture classes (default: 5)
        num_timesteps: Number of diffusion timesteps (default: 100)
        hidden_dim: Hidden layer dimension (default: 256)
        num_blocks: Number of diffusion blocks (default: 4)
    """
    def __init__(self, feature_dim=128, num_classes=5, num_timesteps=100,
                 hidden_dim=256, num_blocks=4):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        # Time embedding
        time_dim = hidden_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        # Diffusion blocks
        self.blocks = nn.ModuleList([
            DiffusionBlock(feature_dim, time_dim, hidden_dim)
            for _ in range(num_blocks)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Define beta schedule for diffusion
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule for beta values (improved stability over linear)
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward_diffusion(self, x0, t):
        """
        Forward diffusion process: add noise to features

        Args:
            x0: Clean features (batch_size, feature_dim)
            t: Timestep indices (batch_size,)

        Returns:
            xt: Noisy features
            noise: Added noise
        """
        noise = torch.randn_like(x0)

        # Get alpha_bar_t for the batch
        alpha_bar_t = self.alphas_cumprod[t].unsqueeze(1)

        # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        return xt, noise

    def denoise(self, xt, t):
        """
        Denoising process through diffusion blocks

        Args:
            xt: Noisy features (batch_size, feature_dim)
            t: Timestep indices (batch_size,)

        Returns:
            denoised: Denoised features (batch_size, feature_dim)
        """
        # Get time embeddings
        time_emb = self.time_embedding(t)

        # Pass through diffusion blocks
        h = xt
        for block in self.blocks:
            h = block(h, time_emb)

        return h

    def forward(self, features, training=True):
        """
        Args:
            features: Input features from Stage 1 (batch_size, feature_dim)
            training: Whether in training mode

        Returns:
            logits: Classification logits (batch_size, num_classes)
            loss_dict: Dictionary of losses (only during training)
        """
        batch_size = features.size(0)

        if training:
            # Sample random timesteps
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=features.device)

            # Forward diffusion: add noise
            noisy_features, noise = self.forward_diffusion(features, t)

            # Denoise
            denoised_features = self.denoise(noisy_features, t)

            # Compute denoising loss (predict the noise)
            denoise_loss = F.mse_loss(denoised_features, features)

            # Classification on denoised features
            logits = self.classifier(denoised_features)

            loss_dict = {'denoise_loss': denoise_loss}

            return logits, loss_dict
        else:
            # Inference: iterative denoising from random noise
            # For faster inference, we can skip this and directly classify
            # Or do a few denoising steps

            # Direct classification (faster)
            logits = self.classifier(features)

            return logits


class EMGDiffusionModel(nn.Module):
    """
    Complete EMG-Diffusion Model combining both stages

    Architecture:
        EMG Signal → Transformer Feature Extractor → Diffusion Classifier → Gesture Class

    Args:
        in_channels: Number of EMG channels (default: 8)
        num_classes: Number of gesture classes (default: 5)
        d_model: Transformer model dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of Transformer layers (default: 6)
        feature_dim: Feature dimension (default: 128)
        num_timesteps: Number of diffusion timesteps (default: 100)
        hidden_dim: Diffusion hidden dimension (default: 256)
    """
    def __init__(self, in_channels=8, num_classes=5, d_model=256, nhead=8,
                 num_layers=6, feature_dim=128, num_timesteps=100, hidden_dim=256):
        super().__init__()

        self.feature_extractor = TransformerFeatureExtractor(
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            feature_dim=feature_dim
        )

        self.diffusion_classifier = DiffusionClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            hidden_dim=hidden_dim
        )

    def forward(self, x, training=True):
        """
        Args:
            x: EMG input (batch_size, in_channels, seq_len)
            training: Whether in training mode

        Returns:
            logits: Classification logits (batch_size, num_classes)
            loss_dict: Dictionary of losses (only during training)
        """
        # Stage 1: Feature extraction
        features = self.feature_extractor(x)

        # Stage 2: Diffusion-based classification
        if training:
            logits, loss_dict = self.diffusion_classifier(features, training=True)
            return logits, loss_dict
        else:
            logits = self.diffusion_classifier(features, training=False)
            return logits

    def extract_features(self, x):
        """Extract features only (for analysis)"""
        return self.feature_extractor(x)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = EMGDiffusionModel(
        in_channels=8,
        num_classes=5,
        d_model=256,
        nhead=8,
        num_layers=6,
        feature_dim=128,
        num_timesteps=100,
        hidden_dim=256
    ).to(device)

    print(f"Model created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 16
    seq_len = 200  # Typical EMG sequence length
    x = torch.randn(batch_size, 8, seq_len).to(device)

    # Training mode
    model.train()
    logits, loss_dict = model(x, training=True)
    print(f"\nTraining mode:")
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss dict keys: {list(loss_dict.keys())}")

    # Inference mode
    model.eval()
    with torch.no_grad():
        logits = model(x, training=False)
    print(f"\nInference mode:")
    print(f"  Logits shape: {logits.shape}")

    print("\nModel test completed successfully!")
