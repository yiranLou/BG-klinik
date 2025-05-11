# neural_network/models_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """Transformer model for synergy-to-motion mapping."""

    def __init__(self, input_dim: int, output_dim: int,
                 d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, features)
        x = self.pos_encoder(x)

        output = self.transformer_encoder(x)
        output = output.mean(0)  # Global average pooling

        return self.output_projection(output)


class AttentionLSTM(nn.Module):
    """LSTM with self-attention mechanism."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 128, num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Self-attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)

        # Attention weights
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        weighted_output = torch.sum(lstm_out * attn_weights, dim=1)

        return self.output_projection(weighted_output)


class ConvLSTM(nn.Module):
    """1D CNN + LSTM hybrid model."""

    def __init__(self, input_dim: int, output_dim: int,
                 conv_channels: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 3, 3],
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        # Convolutional layers
        conv_layers = []
        in_channels = input_dim

        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # LSTM layers
        self.lstm = nn.LSTM(
            conv_channels[-1],
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        # Conv1D expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        conv_out = self.conv(x)
        conv_out = conv_out.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(conv_out)

        # Use last output
        last_output = lstm_out[:, -1, :]

        return self.output_projection(last_output)


class ResidualLSTM(nn.Module):
    """LSTM with residual connections."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 128, num_layers: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Stack of LSTM cells with residual connections
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Initialize hidden states
        h = [torch.zeros(batch_size, self.lstm_cells[0].hidden_size).to(x.device)
             for _ in range(len(self.lstm_cells))]
        c = [torch.zeros(batch_size, self.lstm_cells[0].hidden_size).to(x.device)
             for _ in range(len(self.lstm_cells))]

        # Project input
        x = self.input_projection(x)

        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]

            for i, (lstm_cell, layer_norm) in enumerate(zip(self.lstm_cells, self.layer_norms)):
                h[i], c[i] = lstm_cell(input_t, (h[i], c[i]))

                # Residual connection + layer norm
                if i > 0:
                    h[i] = layer_norm(h[i] + input_t)
                else:
                    h[i] = layer_norm(h[i])

                input_t = self.dropout(h[i])

            outputs.append(h[-1])

        # Use the last hidden state
        final_output = outputs[-1]

        return self.output_projection(final_output)


class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)

    def forward(self, x):
        outputs = []
        for model, weight in zip(self.models, self.weights):
            output = model(x) * weight
            outputs.append(output)

        return torch.stack(outputs).sum(dim=0)


# Enhanced feature extractor
class FeatureExtractor(nn.Module):
    """Extract additional features from synergy patterns."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, features = x.size()

        # Original features
        features_list = [x]

        # First-order differences
        if seq_len > 1:
            diff_1 = torch.zeros_like(x)
            diff_1[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
            features_list.append(diff_1)

        # Second-order differences
        if seq_len > 2:
            diff_2 = torch.zeros_like(x)
            diff_2[:, 2:, :] = diff_1[:, 2:, :] - diff_1[:, 1:-1, :]
            features_list.append(diff_2)

        # Rolling statistics (window size = 3)
        if seq_len >= 3:
            # Rolling mean
            rolling_mean = F.avg_pool1d(x.transpose(1, 2), kernel_size=3, stride=1, padding=1)
            rolling_mean = rolling_mean.transpose(1, 2)
            features_list.append(rolling_mean)

            # Rolling std
            rolling_std = torch.zeros_like(x)
            for i in range(1, seq_len - 1):
                window = x[:, i - 1:i + 2, :]
                rolling_std[:, i, :] = window.std(dim=1)
            features_list.append(rolling_std)

        # Interaction features (pairwise products)
        if features > 1:
            interactions = []
            for i in range(features):
                for j in range(i + 1, features):
                    interaction = x[:, :, i:i + 1] * x[:, :, j:j + 1]
                    interactions.append(interaction)

            if interactions:
                features_list.extend(interactions)

        # Concatenate all features
        enhanced_features = torch.cat(features_list, dim=2)

        return enhanced_features


# Updated model factory
def create_improved_model(model_type: str, **kwargs) -> nn.Module:
    """Create improved model based on type."""
    model_type = model_type.lower()

    if model_type == 'transformer':
        return TransformerModel(**kwargs)
    elif model_type == 'attention_lstm':
        return AttentionLSTM(**kwargs)
    elif model_type == 'conv_lstm':
        return ConvLSTM(**kwargs)
    elif model_type == 'residual_lstm':
        return ResidualLSTM(**kwargs)
    elif model_type == 'ensemble':
        # Create ensemble of different models
        base_models = [
            AttentionLSTM(**kwargs),
            TransformerModel(**kwargs),
            ConvLSTM(**kwargs)
        ]
        return EnsembleModel(base_models)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Combined model with feature extraction
class EnhancedModel(nn.Module):
    """Model with enhanced feature extraction."""

    def __init__(self, model_type: str, input_dim: int, output_dim: int, **kwargs):
        super().__init__()

        self.feature_extractor = FeatureExtractor(input_dim)

        # Calculate enhanced feature dimension
        # Original + diff1 + diff2 + rolling_mean + rolling_std + interactions
        enhanced_dim = input_dim + input_dim + input_dim + input_dim + input_dim
        if input_dim > 1:
            n_interactions = input_dim * (input_dim - 1) // 2
            enhanced_dim += n_interactions

        # Create the main model
        kwargs['input_dim'] = enhanced_dim
        kwargs['output_dim'] = output_dim
        self.model = create_improved_model(model_type, **kwargs)

    def forward(self, x):
        enhanced_features = self.feature_extractor(x)
        return self.model(enhanced_features)