# run_neural_network_comparison.py
"""
Improved neural network training with advanced models and techniques.
Implementation for synergy-to-motion mapping with extensive experimentation.
Fixed version with better error handling and stability.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import sys
import os
import warnings
import traceback
import gc
from torch.cuda.amp import autocast, GradScaler
import codecs

warnings.filterwarnings('ignore')

# Fix encoding issues
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_network.data_loader import MotionDataLoader
from src.neural_network.models import (
    TransformerModel, AttentionLSTM, ConvLSTM, ResidualLSTM,
    EnsembleModel, FeatureExtractor, EnhancedModel, create_improved_model
)
from src.neural_network.trainer import ImprovedTrainer, CombinedLoss
from src.neural_network.evaluator import ModelEvaluator
from src.neural_network.data_augmentation import (
    DataAugmentation, MixupAugmentation, CutMixAugmentation, AugmentedDataset
)


class ComprehensiveNeuralNetworkPipeline:
    """Comprehensive pipeline for neural network training and evaluation with improved stability."""

    def __init__(self, config: Dict):
        """Initialize pipeline with configuration."""
        self.config = config
        self.output_path = Path(config['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Set up logging with UTF-8 encoding
        self._setup_logging()

        # Device setup with better error handling
        self._setup_device()

        # Initialize components
        self.models = {}
        self.trainers = {}
        self.histories = {}
        self.results = {}

        # Add checkpoint tracking
        self.checkpoints = {}

    def _setup_logging(self):
        """Set up logging configuration with UTF-8 encoding."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_path / f'training_{timestamp}.log'

        # Create a custom logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create formatters and handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)

        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _setup_device(self):
        """Set up computing device with better error handling."""
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
                self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

                # Optimize CUDA settings
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.deterministic = False

                # Clear cache
                torch.cuda.empty_cache()
                gc.collect()

                # Enable mixed precision training if supported
                self.use_amp = True
                self.logger.info("Mixed precision training enabled")

            else:
                self.device = torch.device('cpu')
                self.logger.info("Using CPU device")
                self.use_amp = False

        except Exception as e:
            self.logger.warning(f"Error setting up CUDA: {str(e)}")
            self.device = torch.device('cpu')
            self.use_amp = False

    def load_and_prepare_data(self) -> Dict:
        """Load and prepare data with augmentation."""
        self.logger.info("Loading and preparing data...")

        try:
            # Initialize data loader
            data_loader = MotionDataLoader(
                self.config['synergy_path'],
                self.config['ik_path']
            )

            # Prepare data splits
            data_dict = data_loader.prepare_data(
                test_split=self.config.get('test_split', 0.2),
                val_split=self.config.get('val_split', 0.1),
                sequence_length=self.config['sequence_length']
            )

            # Apply data augmentation if enabled
            if self.config.get('use_augmentation', False):
                data_dict['train'] = self._apply_augmentation(data_dict['train'])

            # Save metadata
            self.metadata = data_dict['metadata']
            self.scalers = data_dict['scalers']

            # Important: Check if data is properly scaled
            self._check_data_scaling(data_dict)

            # Log data information
            self.logger.info(f"Input dimension: {self.metadata['input_dim']}")
            self.logger.info(f"Output dimension: {self.metadata['output_dim']}")
            self.logger.info(f"Sequence length: {self.config['sequence_length']}")
            self.logger.info(f"Training samples: {len(data_dict['train'].dataset)}")
            self.logger.info(f"Validation samples: {len(data_dict['val'].dataset)}")
            self.logger.info(f"Test samples: {len(data_dict['test'].dataset)}")

            return data_dict

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _check_data_scaling(self, data_dict: Dict):
        """Check if data is properly scaled."""
        # Check training data
        sample_x, sample_y = data_dict['train'].dataset[0]

        self.logger.info(f"Data scaling check:")
        self.logger.info(f"Input range: [{sample_x.min():.3f}, {sample_x.max():.3f}]")
        self.logger.info(f"Output range: [{sample_y.min():.3f}, {sample_y.max():.3f}]")

        # Check if data is normalized
        if sample_x.max() > 10 or sample_x.min() < -10:
            self.logger.warning("Input data may not be properly normalized!")
        if sample_y.max() > 10 or sample_y.min() < -10:
            self.logger.warning("Output data may not be properly normalized!")

    def _apply_augmentation(self, train_loader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """Apply data augmentation to training data."""
        self.logger.info("Applying data augmentation...")

        try:
            # Create augmentation objects with safer parameters
            basic_augment = DataAugmentation(
                noise_level=self.config.get('noise_level', 0.01),
                time_warp_sigma=self.config.get('time_warp_sigma', 0.05),
                scale_sigma=self.config.get('scale_sigma', 0.05),
                augment_prob=self.config.get('augment_prob', 0.3)
            )

            mixup = MixupAugmentation(alpha=self.config.get('mixup_alpha', 0.1))
            cutmix = CutMixAugmentation(alpha=self.config.get('cutmix_alpha', 0.1))

            # Wrap dataset with augmentation
            augmented_dataset = AugmentedDataset(
                train_loader.dataset,
                augmentations=basic_augment,
                mixup=mixup,
                cutmix=cutmix
            )

            # Create new data loader with smaller batch size for stability
            return torch.utils.data.DataLoader(
                augmented_dataset,
                batch_size=self.config.get('batch_size', 16),
                shuffle=True,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=True if self.device.type == 'cuda' else False,
                persistent_workers=False
            )
        except Exception as e:
            self.logger.warning(f"Error applying augmentation: {str(e)}")
            return train_loader

    def create_models(self) -> Dict[str, nn.Module]:
        """Create all models for comparison with error handling."""
        self.logger.info("Creating models...")

        models_to_create = self.config.get('models_to_compare', ['attention_lstm'])

        for model_type in models_to_create:
            try:
                self.logger.info(f"Creating {model_type} model...")

                # Get model-specific kwargs
                model_kwargs = self._get_model_kwargs(model_type)

                # Create model
                if self.config.get('use_enhanced_features', False):
                    model = EnhancedModel(
                        model_type=model_type,
                        **self._get_enhanced_model_kwargs()
                    )
                else:
                    model = create_improved_model(model_type, **model_kwargs)

                # Move to device
                model = model.to(self.device)

                # Model statistics
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.logger.info(f"{model_type} parameters: {total_params:,}")

                self.models[model_type] = model

            except Exception as e:
                self.logger.error(f"Failed to create {model_type} model: {str(e)}")
                self.logger.error(traceback.format_exc())
                continue

        if not self.models:
            raise RuntimeError("No models were successfully created")

        return self.models

    def _get_enhanced_model_kwargs(self) -> Dict:
        """Get enhanced model kwargs with safer defaults."""
        return {
            'input_dim': self.metadata['input_dim'],
            'output_dim': self.metadata['output_dim'],
            'hidden_dim': self.config.get('hidden_dim', 64),
            'num_layers': self.config.get('num_layers', 2),
            'dropout': self.config.get('dropout', 0.2),
            'nhead': self.config.get('nhead', 4),
            'conv_channels': self.config.get('conv_channels', [16, 32, 64]),
            'kernel_sizes': self.config.get('kernel_sizes', [3, 3, 3])
        }

    def _get_model_kwargs(self, model_type: str) -> Dict:
        """Get model-specific keyword arguments with safer defaults."""
        base_kwargs = {
            'input_dim': self.metadata['input_dim'],
            'output_dim': self.metadata['output_dim'],
            'dropout': self.config.get('dropout', 0.2)
        }

        if model_type == 'transformer':
            return {
                **base_kwargs,
                'd_model': self.config.get('hidden_dim', 64),
                'nhead': self.config.get('nhead', 4),
                'num_layers': self.config.get('num_layers', 2)
            }
        elif model_type in ['attention_lstm', 'residual_lstm']:
            return {
                **base_kwargs,
                'hidden_dim': self.config.get('hidden_dim', 64),
                'num_layers': self.config.get('num_layers', 2)
            }
        elif model_type == 'conv_lstm':
            return {
                **base_kwargs,
                'conv_channels': self.config.get('conv_channels', [16, 32, 64]),
                'kernel_sizes': self.config.get('kernel_sizes', [3, 3, 3]),
                'lstm_hidden': self.config.get('hidden_dim', 64),
                'lstm_layers': self.config.get('num_layers', 2)
            }
        else:
            return {
                **base_kwargs,
                'hidden_dim': self.config.get('hidden_dim', 64),
                'num_layers': self.config.get('num_layers', 2)
            }

    def train_models(self, data_dict: Dict) -> Dict[str, Dict]:
        """Train all models with improved error handling and checkpointing."""
        self.logger.info("Starting model training...")

        for model_name, model in self.models.items():
            try:
                self.logger.info(f"\nTraining {model_name}...")

                # Create trainer with model-specific settings
                trainer = ImprovedTrainer(model, self.device)

                # Model-specific hyperparameters (safer defaults)
                lr = self.config.get(f'{model_name}_lr', self.config.get('learning_rate', 1e-4))
                weight_decay = self.config.get(f'{model_name}_weight_decay',
                                               self.config.get('weight_decay', 1e-5))

                # Train model with checkpointing
                start_time = time.time()
                history = self._train_with_checkpointing(
                    trainer=trainer,
                    model_name=model_name,
                    train_loader=data_dict['train'],
                    val_loader=data_dict['val'],
                    epochs=self.config.get('epochs', 100),
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    patience=self.config.get('patience', 20),
                    gradient_clip=self.config.get('gradient_clip', 0.5),
                    warmup_epochs=self.config.get('warmup_epochs', 5)
                )
                training_time = time.time() - start_time

                # Store results
                self.trainers[model_name] = trainer
                self.histories[model_name] = history
                history['training_time'] = training_time

                self.logger.info(f"{model_name} training completed in {training_time/60:.2f} minutes")
                self.logger.info(f"Best validation loss: {history['best_val_loss']:.6f} at epoch {history['best_epoch']}")

                # Clear GPU cache after each model
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                self.logger.error(traceback.format_exc())
                continue

        return self.histories

    def _train_with_checkpointing(self, trainer, model_name, train_loader, val_loader,
                                  epochs, learning_rate, weight_decay, patience,
                                  gradient_clip, warmup_epochs):
        """Train model with checkpointing and error recovery."""
        checkpoint_path = self.output_path / f'{model_name}_checkpoint.pth'
        best_model_path = self.output_path / f'{model_name}_best.pth'

        # Train model using improved trainer
        save_path = str(best_model_path)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            gradient_clip=gradient_clip,
            warmup_epochs=warmup_epochs,
            save_path=save_path
        )

        return history

    def evaluate_models(self, test_loader: torch.utils.data.DataLoader) -> Dict:
        """Comprehensive evaluation of all models with error handling."""
        self.logger.info("\nEvaluating models...")

        try:
            evaluator = ModelEvaluator(str(self.output_path))

            # Load best models
            valid_models = {}
            for model_name, trainer in self.trainers.items():
                try:
                    best_path = self.output_path / f'{model_name}_best.pth'
                    if best_path.exists():
                        trainer.load_model(str(best_path))
                    valid_models[model_name] = trainer.model
                except Exception as e:
                    self.logger.warning(f"Failed to load best model for {model_name}: {str(e)}")
                    continue

            if not valid_models:
                self.logger.error("No valid models found for evaluation")
                return {}

            # Compare models
            results_df = evaluator.compare_models(
                valid_models,
                test_loader,
                self.metadata['joint_names']
            )

            # Generate visualizations with error handling
            try:
                evaluator.plot_model_comparison(results_df)
            except Exception as e:
                self.logger.warning(f"Failed to generate comparison plot: {str(e)}")

            # Get proper time points for visualization
            try:
                predictions, targets = self._get_sample_predictions(test_loader)
                n_samples = len(targets)
                timestep = 0.01
                time_points = np.arange(n_samples) * timestep

                evaluator.plot_predictions(
                    valid_models,
                    test_loader,
                    self.metadata['joint_names'],
                    time_points,
                    self.scalers
                )
            except Exception as e:
                self.logger.warning(f"Failed to generate prediction plots: {str(e)}")

            # Additional plots
            try:
                evaluator.plot_error_distribution(
                    valid_models,
                    test_loader,
                    self.metadata['joint_names']
                )
                evaluator.plot_residuals(
                    valid_models,
                    test_loader,
                    self.metadata['joint_names']
                )
                evaluator.create_training_comparison_plot(self.histories)
            except Exception as e:
                self.logger.warning(f"Failed to generate additional plots: {str(e)}")

            # Generate report
            evaluator.generate_report(results_df)

            # Detailed evaluation with uncertainty and proper scaling
            self.logger.info("\nPerforming detailed evaluation with uncertainty estimation...")
            for model_name, trainer in self.trainers.items():
                try:
                    # Pass scalers for proper inverse transformation
                    results = trainer.evaluate_with_uncertainty(
                        test_loader,
                        n_samples=10,
                        scalers=self.scalers
                    )
                    self.results[model_name] = results

                    self.logger.info(f"\n{model_name} results:")
                    self.logger.info(f"RMSE: {results['rmse']:.4f}")
                    self.logger.info(f"MAE: {results['mae']:.4f}")
                    self.logger.info(f"R-squared (mean): {results['r2_mean']:.4f}")
                    self.logger.info(f"Uncertainty: {results['uncertainty_mean']:.4f}")

                    # Check individual joint R² scores
                    self.logger.info("Individual joint R-squared scores:")
                    for i, r2 in enumerate(results['r2_scores']):
                        if i < len(self.metadata['joint_names']):
                            joint_name = self.metadata['joint_names'][i]
                        else:
                            joint_name = f"Joint {i}"

                        if r2 < 0:
                            self.logger.warning(f"{joint_name}: R-squared = {r2:.4f} (negative)")
                        else:
                            self.logger.info(f"{joint_name}: R-squared = {r2:.4f}")

                except Exception as e:
                    self.logger.warning(f"Failed to evaluate {model_name}: {str(e)}")
                    self.logger.warning(traceback.format_exc())
                    continue

            return self.results

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}

    def _get_sample_predictions(self, test_loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get sample predictions to determine proper dimensions."""
        # Use any model to get predictions
        model_name = list(self.trainers.keys())[0]
        model = self.trainers[model_name].model
        model.eval()

        predictions = []
        targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)

                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y.numpy())

        return np.vstack(predictions), np.vstack(targets)

    def generate_final_report(self):
        """Generate comprehensive final report with error handling."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config,
                'metadata': {
                    'input_dim': self.metadata.get('input_dim', 'N/A'),
                    'output_dim': self.metadata.get('output_dim', 'N/A'),
                    'sequence_length': self.config.get('sequence_length', 'N/A'),
                    'joint_names': self.metadata.get('joint_names', [])
                },
                'training_results': {},
                'evaluation_results': {}
            }

            # Add training history
            for model_name, history in self.histories.items():
                if history.get('val_loss'):
                    report['training_results'][model_name] = {
                        'best_epoch': history.get('best_epoch', 0),
                        'best_val_loss': history.get('best_val_loss', float('inf')),
                        'training_time': history.get('training_time', 0),
                        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None
                    }

            # Add evaluation results
            for model_name, results in self.results.items():
                report['evaluation_results'][model_name] = {
                    'rmse': float(results.get('rmse', 0)),
                    'mae': float(results.get('mae', 0)),
                    'r2_mean': float(results.get('r2_mean', 0)),
                    'uncertainty_mean': float(results.get('uncertainty_mean', 0)),
                    'r2_scores': [float(score) for score in results.get('r2_scores', [])]
                }

            # Save report
            report_path = self.output_path / 'final_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"\nFinal report saved to: {report_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate final report: {str(e)}")
            self.logger.error(traceback.format_exc())

    def run_full_pipeline(self):
        """Execute the full training and evaluation pipeline with error recovery."""
        start_time = time.time()

        try:
            # Load and prepare data
            data_dict = self.load_and_prepare_data()

            # Create models
            self.create_models()

            # Train models
            self.train_models(data_dict)

            # Evaluate models
            if self.trainers:
                self.evaluate_models(data_dict['test'])
            else:
                self.logger.warning("No trained models available for evaluation")

            # Generate final report
            self.generate_final_report()

            total_time = time.time() - start_time
            self.logger.info(f"\nPipeline completed in {total_time/60:.2f} minutes")

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Still try to save what we have
            self.generate_final_report()
            raise


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Comprehensive neural network training pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--synergy_path', type=str,
                        default=r"C:\temporary_file\BG_klinik\newPipeline\data\processed\synergies\synergy_extraction_n3\activation_patterns_smooth.csv",
                        help='Path to synergy activation patterns')
    parser.add_argument('--ik_path', type=str,
                        default=r"C:\temporary_file\BG_klinik\newPipeline\data\processed\Ik_result\N10_IK\kinematics_result01.mot",
                        help='Path to IK results')
    parser.add_argument('--output_path', type=str,
                        default=r"C:\temporary_file\BG_klinik\newPipeline\data\results\neural_network_improved",
                        help='Path to save results')
    parser.add_argument('--models_to_compare', nargs='+', default=['attention_lstm'],
                        help='List of models to compare')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Sequence length for models')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for models')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                        help='Use data augmentation')
    parser.add_argument('--use_enhanced_features', action='store_true', default=False,
                        help='Use enhanced feature extraction')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--gradient_clip', type=float, default=0.5,
                        help='Gradient clipping value')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads for transformer')
    parser.add_argument('--conv_channels', nargs='+', type=int, default=[16, 32, 64],
                        help='Convolutional channels for ConvLSTM')
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[3, 3, 3],
                        help='Kernel sizes for ConvLSTM')

    args = parser.parse_args()

    # Create configuration
    if args.config:
        # Load configuration from file
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Create configuration from arguments
        config = {
            'synergy_path': args.synergy_path,
            'ik_path': args.ik_path,
            'output_path': args.output_path,
            'models_to_compare': args.models_to_compare,
            'sequence_length': args.sequence_length,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'dropout': args.dropout,
            'use_augmentation': args.use_augmentation,
            'use_enhanced_features': args.use_enhanced_features,
            'patience': args.patience,
            'weight_decay': args.weight_decay,
            'gradient_clip': args.gradient_clip,
            'warmup_epochs': args.warmup_epochs,
            'nhead': args.nhead,
            'conv_channels': args.conv_channels,
            'kernel_sizes': args.kernel_sizes,
            'test_split': 0.2,
            'val_split': 0.1,
            # Data augmentation parameters (safer defaults)
            'noise_level': 0.01,
            'time_warp_sigma': 0.05,
            'scale_sigma': 0.05,
            'augment_prob': 0.3,
            'mixup_alpha': 0.1,
            'cutmix_alpha': 0.1
        }

    # Save configuration
    output_path = Path(config['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Create and run pipeline
    pipeline = ComprehensiveNeuralNetworkPipeline(config)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()