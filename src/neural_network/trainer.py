# neural_network/trainer.py
# Fixed version with better training and evaluation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import time
from tqdm import tqdm
from sklearn.metrics import r2_score


class CombinedLoss(nn.Module):
    """Combined loss function with MSE and MAE."""

    def __init__(self, mse_weight: float = 0.8, smooth_weight: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.mse_weight = mse_weight
        self.mae_weight = 1.0 - mse_weight - smooth_weight
        self.smooth_weight = smooth_weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)

        # Smoothness loss (penalize large changes between consecutive predictions)
        if pred.dim() > 1 and pred.size(0) > 1:
            pred_diff = pred[1:] - pred[:-1]
            smooth_loss = torch.mean(pred_diff ** 2)
        else:
            smooth_loss = torch.tensor(0.0).to(pred.device)

        total_loss = (self.mse_weight * mse_loss +
                      self.mae_weight * mae_loss +
                      self.smooth_weight * smooth_loss)

        return total_loss


class ImprovedTrainer:
    """Improved trainer with advanced optimization techniques."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'train_mae': [],
            'val_mae': [],
            'lr': [],
            'train_time': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 200,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 30,
              gradient_clip: float = 1.0,
              warmup_epochs: int = 10,
              save_path: Optional[str] = None) -> Dict:
        """
        Train the model with improved optimization.
        """
        # Setup optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6
        )

        # Loss function
        criterion = CombinedLoss(mse_weight=0.8, smooth_weight=0.1)

        # Metrics
        mse_metric = nn.MSELoss()
        mae_metric = nn.L1Loss()

        # Early stopping
        no_improve_count = 0

        # Best loss tracking
        best_val_loss = float('inf')

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()

            # Warmup learning rate
            if epoch < warmup_epochs:
                warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_mse = 0.0
            train_mae = 0.0
            train_steps = 0

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            for batch_x, batch_y in progress_bar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                # Calculate metrics
                with torch.no_grad():
                    mse = mse_metric(outputs, batch_y)
                    mae = mae_metric(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

                optimizer.step()

                train_loss += loss.item()
                train_mse += mse.item()
                train_mae += mae.item()
                train_steps += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mse': f'{mse.item():.4f}',
                    'mae': f'{mae.item():.4f}'
                })

            avg_train_loss = train_loss / train_steps
            avg_train_mse = train_mse / train_steps
            avg_train_mae = train_mae / train_steps

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_mse = 0.0
            val_mae = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    mse = mse_metric(outputs, batch_y)
                    mae = mae_metric(outputs, batch_y)

                    val_loss += loss.item()
                    val_mse += mse.item()
                    val_mae += mae.item()
                    val_steps += 1

            avg_val_loss = val_loss / val_steps
            avg_val_mse = val_mse / val_steps
            avg_val_mae = val_mae / val_steps

            # Update scheduler
            if epoch >= warmup_epochs:
                scheduler.step()

            # Record history
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_mse'].append(avg_train_mse)
            self.history['val_mse'].append(avg_val_mse)
            self.history['train_mae'].append(avg_train_mae)
            self.history['val_mae'].append(avg_val_mae)
            self.history['lr'].append(current_lr)
            self.history['train_time'].append(epoch_time)

            # Print progress
            print(f'Epoch {epoch + 1}/{epochs}: '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, '
                  f'LR: {current_lr:.6f}, '
                  f'Time: {epoch_time:.2f}s')

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.history['best_val_loss'] = avg_val_loss
                self.history['best_epoch'] = epoch + 1
                no_improve_count = 0

                # Save best model
                if save_path:
                    self.save_model(save_path)
                    print(f'Saved best model with val loss: {avg_val_loss:.6f}')
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        return self.history

    def evaluate_with_uncertainty(self, test_loader: DataLoader,
                                  n_samples: int = 10,
                                  scalers: Dict = None) -> Dict:
        """
        Evaluate model with uncertainty estimation and proper inverse scaling.
        """
        self.model.eval()  # Set to eval mode for deterministic output

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Single forward pass for evaluation
                output = self.model(batch_x)

                all_predictions.append(output.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        # Concatenate results
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)

        # Apply inverse scaling if scalers are provided
        if scalers and 'y' in scalers:
            predictions = scalers['y'].inverse_transform(predictions)
            targets = scalers['y'].inverse_transform(targets)

        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)

        # R-squared for each output
        r2_scores = []
        for i in range(targets.shape[1]):
            y_true = targets[:, i]
            y_pred = predictions[:, i]

            # Avoid division by zero
            if np.var(y_true) == 0:
                r2_scores.append(0.0)
            else:
                r2 = r2_score(y_true, y_pred)
                r2_scores.append(r2)

        # Calculate uncertainty (standard deviation of predictions)
        uncertainty_mean = np.mean(np.std(predictions, axis=0))

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_scores': r2_scores,
            'r2_mean': np.mean(r2_scores),
            'predictions_mean': predictions,
            'targets': targets,
            'uncertainty_mean': uncertainty_mean
        }

    def save_model(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)

    def load_model(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)