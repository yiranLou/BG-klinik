# neural_network/additional_models.py
"""
Additional model architectures for time series prediction:
- Lightweight RNN variants (SimpleGRU, SimpleLSTM)
- Temporal Convolutional Network (TCN)
- Hybrid CNN-RNN
- Bayesian Structural Time Series (BSTS) wrapper
- XGBoost and LightGBM wrappers
- SVM and Random Forest wrappers (new)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
import xgboost as xgb
import lightgbm as lgb
from torch.nn.utils import weight_norm
from sklearn.base import BaseEstimator
from statsmodels.tsa.statespace.structural import UnobservedComponents
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


class SimpleGRU(nn.Module):
    """Simple GRU model for sequence-to-value prediction."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        # GRU output: [batch_size, seq_len, hidden_dim * num_directions]
        output, _ = self.gru(x)

        # Use the output from the last time step
        # Shape: [batch_size, hidden_dim * num_directions]
        last_output = output[:, -1, :]

        # Final projection
        return self.fc(last_output)


class SimpleLSTM(nn.Module):
    """Simple LSTM model for sequence-to-value prediction."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        # LSTM output: [batch_size, seq_len, hidden_dim * num_directions]
        output, _ = self.lstm(x)

        # Use the output from the last time step
        # Shape: [batch_size, hidden_dim * num_directions]
        last_output = output[:, -1, :]

        # Final projection
        return self.fc(last_output)


class Chomp1d(nn.Module):
    """Removes padding from the end of the sequence to ensure causal convolutions."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Temporal convolutional block with dilated causal convolutions."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int,
                 dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()

        # First dilated convolution
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated convolution
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 1x1 convolution for residual connection if input and output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # Main path
        out = self.net(x)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network with dilated causal convolutions."""

    def __init__(self, input_dim: int, hidden_channels: List[int],
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()

        layers = []
        num_levels = len(hidden_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else hidden_channels[i - 1]
            out_channels = hidden_channels[i]

            # Add temporal block with appropriate dilation
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # TCN expects input of shape [batch_size, channels, seq_len]
        # Our input is [batch_size, seq_len, channels]
        return self.network(x.transpose(1, 2)).transpose(1, 2)


class TCNModel(nn.Module):
    """Complete TCN model for sequence-to-value prediction."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_channels: List[int] = [32, 64, 128, 64],
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()

        # TCN network
        self.tcn = TemporalConvNet(
            input_dim=input_dim,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.LayerNorm(hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, output_dim)
        )

    def forward(self, x):
        # Process with TCN
        tcn_output = self.tcn(x)

        # Use last time step for output
        last_output = tcn_output[:, -1, :]

        # Final projection
        return self.fc(last_output)


class HybridCNNRNN(nn.Module):
    """Hybrid model combining CNN feature extraction with RNN sequence modeling."""

    def __init__(self, input_dim: int, output_dim: int,
                 cnn_channels: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 3, 3],
                 rnn_hidden: int = 64,
                 rnn_layers: int = 2,
                 rnn_type: str = 'lstm',
                 dropout: float = 0.2):
        super().__init__()

        # CNN layers for feature extraction
        cnn_layers = []
        in_channels = input_dim

        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, kernel_sizes)):
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # RNN for sequence modeling
        rnn_class = nn.LSTM if rnn_type.lower() == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            input_size=cnn_channels[-1],
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.LayerNorm(rnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, output_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        # CNN expects [batch_size, channels, seq_len]
        x = x.transpose(1, 2)
        cnn_output = self.cnn(x)

        # RNN expects [batch_size, seq_len, features]
        rnn_input = cnn_output.transpose(1, 2)

        # RNN output: [batch_size, seq_len, hidden_size * num_directions]
        rnn_output, _ = self.rnn(rnn_input)

        # Use the output from the last time step
        last_output = rnn_output[:, -1, :]

        # Final projection
        return self.fc(last_output)


class BSTSRegressor(BaseEstimator):
    """
    Bayesian Structural Time Series wrapper for sklearn compatibility.
    Uses statsmodels UnobservedComponents as the backend.
    """

    def __init__(self, level: bool = True, trend: bool = True,
                 seasonal: int = None, freq_seasonal: List = None,
                 cycle: bool = False, autoregressive: Optional[int] = None,
                 exog_dim: Optional[int] = None, mle_regression: bool = True,
                 sequence_length: int = 10):
        self.level = level
        self.trend = trend
        self.seasonal = seasonal
        self.freq_seasonal = freq_seasonal
        self.cycle = cycle
        self.autoregressive = autoregressive
        self.exog_dim = exog_dim
        self.mle_regression = mle_regression
        self.sequence_length = sequence_length
        self.models = None

    def fit(self, X, y):
        """
        Fit BSTS model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]
            y: Target values of shape [n_samples, n_outputs]
        """
        # Number of outputs
        n_outputs = y.shape[1]
        self.models = []

        # Reshape X to use only the last point of each sequence
        n_samples = X.shape[0]
        X_last = X[:, -1, :]

        # Fit a separate model for each output dimension
        for i in range(n_outputs):
            # Create model
            model = UnobservedComponents(
                y[:, i],
                exog=X_last if self.exog_dim is not None else None,
                level=self.level,
                trend=self.trend,
                seasonal=self.seasonal,
                freq_seasonal=self.freq_seasonal,
                cycle=self.cycle,
                autoregressive=self.autoregressive,
                mle_regression=self.mle_regression
            )

            # Fit model with maximum likelihood
            fitted_model = model.fit(disp=False)
            self.models.append(fitted_model)

        return self

    def predict(self, X):
        """
        Make predictions with the fitted BSTS model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]

        Returns:
            Predictions of shape [n_samples, n_outputs]
        """
        if self.models is None:
            raise ValueError("Model has not been fitted yet.")

        # Number of samples
        n_samples = X.shape[0]
        n_outputs = len(self.models)

        # Reshape X to use only the last point of each sequence
        X_last = X[:, -1, :]

        # Make predictions
        predictions = np.zeros((n_samples, n_outputs))

        for i, model in enumerate(self.models):
            # Use in-sample prediction for simplicity
            predictions[:, i] = model.predict(exog=X_last if self.exog_dim is not None else None)

        return predictions


class XGBoostWrapper(BaseEstimator):
    """Wrapper for XGBoost to handle sequence input."""

    def __init__(self, sequence_length: int = 10, num_outputs: int = 1,
                 max_depth: int = 6, learning_rate: float = 0.1,
                 n_estimators: int = 100, objective: str = 'reg:squarederror',
                 **xgb_params):
        self.sequence_length = sequence_length
        self.num_outputs = num_outputs
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.xgb_params = xgb_params
        self.models = None

    def _reshape_features(self, X):
        """Reshape 3D sequence input to 2D feature matrix for XGBoost."""
        # X shape: [n_samples, seq_len, n_features]
        n_samples, seq_len, n_features = X.shape

        # Flatten sequence dimension to create feature vector
        # New shape: [n_samples, seq_len * n_features]
        return X.reshape(n_samples, seq_len * n_features)

    def fit(self, X, y):
        """
        Fit XGBoost model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]
            y: Target values of shape [n_samples, n_outputs]
        """
        # Reshape input features
        X_flat = self._reshape_features(X)

        # Train a separate model for each output
        self.models = []
        for i in range(self.num_outputs):
            model = xgb.XGBRegressor(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                objective=self.objective,
                **self.xgb_params
            )
            model.fit(X_flat, y[:, i])
            self.models.append(model)

        return self

    def predict(self, X):
        """
        Make predictions with the fitted XGBoost model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]

        Returns:
            Predictions of shape [n_samples, n_outputs]
        """
        if self.models is None:
            raise ValueError("Model has not been fitted yet.")

        # Reshape input features
        X_flat = self._reshape_features(X)

        # Make predictions for each output
        predictions = np.zeros((X.shape[0], self.num_outputs))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_flat)

        return predictions


class LightGBMWrapper(BaseEstimator):
    """Wrapper for LightGBM to handle sequence input."""

    def __init__(self, sequence_length: int = 10, num_outputs: int = 1,
                 num_leaves: int = 31, learning_rate: float = 0.1,
                 n_estimators: int = 100, objective: str = 'regression',
                 **lgb_params):
        self.sequence_length = sequence_length
        self.num_outputs = num_outputs
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.lgb_params = lgb_params
        self.models = None

    def _reshape_features(self, X):
        """Reshape 3D sequence input to 2D feature matrix for LightGBM."""
        # X shape: [n_samples, seq_len, n_features]
        n_samples, seq_len, n_features = X.shape

        # Flatten sequence dimension to create feature vector
        # New shape: [n_samples, seq_len * n_features]
        return X.reshape(n_samples, seq_len * n_features)

    def fit(self, X, y):
        """
        Fit LightGBM model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]
            y: Target values of shape [n_samples, n_outputs]
        """
        # Reshape input features
        X_flat = self._reshape_features(X)

        # Train a separate model for each output
        self.models = []
        for i in range(self.num_outputs):
            model = lgb.LGBMRegressor(
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                objective=self.objective,
                **self.lgb_params
            )
            model.fit(X_flat, y[:, i])
            self.models.append(model)

        return self

    def predict(self, X):
        """
        Make predictions with the fitted LightGBM model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]

        Returns:
            Predictions of shape [n_samples, n_outputs]
        """
        if self.models is None:
            raise ValueError("Model has not been fitted yet.")

        # Reshape input features
        X_flat = self._reshape_features(X)

        # Make predictions for each output
        predictions = np.zeros((X.shape[0], self.num_outputs))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_flat)

        return predictions


class SVMWrapper(BaseEstimator):
    """Wrapper for Support Vector Regression."""
    
    def __init__(self, sequence_length: int = 10, num_outputs: int = 1,
                 C: float = 1.0, kernel: str = 'rbf', 
                 gamma: str = 'scale', epsilon: float = 0.1,
                 **svm_params):
        self.sequence_length = sequence_length
        self.num_outputs = num_outputs
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.epsilon = epsilon
        self.svm_params = svm_params
        self.models = None

    def _reshape_features(self, X):
        """Reshape 3D sequence input to 2D feature matrix for SVM."""
        # X shape: [n_samples, seq_len, n_features]
        n_samples, seq_len, n_features = X.shape

        # Flatten sequence dimension to create feature vector
        # New shape: [n_samples, seq_len * n_features]
        return X.reshape(n_samples, seq_len * n_features)

    def fit(self, X, y):
        """
        Fit SVM model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]
            y: Target values of shape [n_samples, n_outputs]
        """
        # Reshape input features
        X_flat = self._reshape_features(X)

        # Train a separate model for each output
        self.models = []
        for i in range(self.num_outputs):
            model = SVR(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                epsilon=self.epsilon,
                **self.svm_params
            )
            model.fit(X_flat, y[:, i])
            self.models.append(model)

        return self

    def predict(self, X):
        """
        Make predictions with the fitted SVM model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]

        Returns:
            Predictions of shape [n_samples, n_outputs]
        """
        if self.models is None:
            raise ValueError("Model has not been fitted yet.")

        # Reshape input features
        X_flat = self._reshape_features(X)

        # Make predictions for each output
        predictions = np.zeros((X.shape[0], self.num_outputs))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_flat)

        return predictions


class RandomForestWrapper(BaseEstimator):
    """Wrapper for Random Forest Regression."""
    
    def __init__(self, sequence_length: int = 10, num_outputs: int = 1,
                 n_estimators: int = 100, max_depth: int = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 **rf_params):
        self.sequence_length = sequence_length
        self.num_outputs = num_outputs
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.rf_params = rf_params
        self.models = None

    def _reshape_features(self, X):
        """Reshape 3D sequence input to 2D feature matrix for Random Forest."""
        # X shape: [n_samples, seq_len, n_features]
        n_samples, seq_len, n_features = X.shape

        # Flatten sequence dimension to create feature vector
        # New shape: [n_samples, seq_len * n_features]
        return X.reshape(n_samples, seq_len * n_features)

    def fit(self, X, y):
        """
        Fit Random Forest model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]
            y: Target values of shape [n_samples, n_outputs]
        """
        # Reshape input features
        X_flat = self._reshape_features(X)

        # Train a separate model for each output
        self.models = []
        for i in range(self.num_outputs):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                **self.rf_params
            )
            model.fit(X_flat, y[:, i])
            self.models.append(model)

        return self

    def predict(self, X):
        """
        Make predictions with the fitted Random Forest model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]

        Returns:
            Predictions of shape [n_samples, n_outputs]
        """
        if self.models is None:
            raise ValueError("Model has not been fitted yet.")

        # Reshape input features
        X_flat = self._reshape_features(X)

        # Make predictions for each output
        predictions = np.zeros((X.shape[0], self.num_outputs))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_flat)

        return predictions


# Wrapper for PyTorch models to provide a unified interface with sklearn-like models
class PyTorchModelWrapper(BaseEstimator):
    """Wrapper for PyTorch models to provide sklearn-like interface."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None,
                 epochs: int = 100, batch_size: int = 32,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 patience: int = 10, gradient_clip: float = 1.0):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.gradient_clip = gradient_clip

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.history = []

    def fit(self, X, y):
        """
        Fit PyTorch model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]
            y: Target values of shape [n_samples, n_outputs]
        """
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Training loop
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # Create random indices for batches
            indices = torch.randperm(X_tensor.size(0)).to(self.device)
            batch_losses = []

            # Process in batches
            for start_idx in range(0, X_tensor.size(0), self.batch_size):
                end_idx = min(start_idx + self.batch_size, X_tensor.size(0))
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

                batch_losses.append(loss.item())

            # Average loss for the epoch
            avg_loss = np.mean(batch_losses)
            self.history.append(avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        return self

    def predict(self, X):
        """
        Make predictions with the fitted PyTorch model.

        Args:
            X: Input features of shape [n_samples, seq_len, n_features]

        Returns:
            Predictions of shape [n_samples, n_outputs]
        """
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = []

            # Process in batches
            for start_idx in range(0, X_tensor.size(0), self.batch_size):
                end_idx = min(start_idx + self.batch_size, X_tensor.size(0))
                X_batch = X_tensor[start_idx:end_idx]

                # Forward pass
                outputs = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())

        return np.vstack(predictions)


# Factory function to create additional models
def create_additional_model(model_type: str, **kwargs) -> Union[BaseEstimator, nn.Module]:
    """Create model based on type."""
    model_type = model_type.lower()

    if model_type == 'gru':
        return SimpleGRU(**kwargs)
    elif model_type == 'lstm':
        return SimpleLSTM(**kwargs)
    elif model_type == 'tcn':
        return TCNModel(**kwargs)
    elif model_type == 'cnn_rnn':
        return HybridCNNRNN(**kwargs)
    elif model_type == 'bsts':
        return BSTSRegressor(**kwargs)
    elif model_type == 'xgboost':
        return XGBoostWrapper(**kwargs)
    elif model_type == 'lightgbm':
        return LightGBMWrapper(**kwargs)
    elif model_type == 'svm':
        return SVMWrapper(**kwargs)
    elif model_type == 'random_forest':
        return RandomForestWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")