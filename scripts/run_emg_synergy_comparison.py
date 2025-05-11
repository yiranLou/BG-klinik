# scripts/run_emg_synergy_comparison.py
"""
Compare neural network performance using raw EMG vs synergy inputs.
Research Question 1: How does muscle synergy reduce EMG dimensionality effectively?
"""

import torch
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
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

warnings.filterwarnings('ignore')

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_network.data_loader import MotionDataLoader
from src.neural_network.data_loader_emg import EMGDataLoader
from src.neural_network.models import AttentionLSTM, create_improved_model
from src.neural_network.trainer import ImprovedTrainer
from src.neural_network.evaluator import ModelEvaluator


class DetailedModelEvaluator(ModelEvaluator):
    """Enhanced evaluator that provides detailed joint-by-joint metrics."""

    def __init__(self, output_path: str):
        super().__init__(output_path)

    def evaluate_model_detailed(self, model: torch.nn.Module,
                                test_loader: torch.utils.data.DataLoader,
                                joint_names: List[str],
                                model_name: str) -> pd.DataFrame:
        """
        Evaluate model performance for each joint individually.

        Args:
            model: Neural network model
            test_loader: Test data loader
            joint_names: Names of joints being predicted
            model_name: Name of the model for reporting

        Returns:
            DataFrame with detailed metrics for each joint
        """
        # Get predictions
        predictions, targets = self._get_predictions(model, test_loader)

        results = []

        # Calculate metrics for each joint
        for i, joint_name in enumerate(joint_names):
            y_true = targets[:, i]
            y_pred = predictions[:, i]

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)

            # Calculate R² score
            if np.var(y_true) == 0:
                r2 = 0.0
            else:
                r2 = r2_score(y_true, y_pred)

            # Calculate correlation
            if np.std(y_true) == 0 or np.std(y_pred) == 0:
                correlation = 0.0
            else:
                correlation, _ = stats.pearsonr(y_true, y_pred)

            results.append({
                'Model': model_name,
                'Joint': joint_name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Correlation': correlation
            })

        results_df = pd.DataFrame(results)

        # Save to CSV
        csv_path = self.output_path / f'{model_name}_detailed_results.csv'
        results_df.to_csv(csv_path, index=False)

        return results_df


class EMGSynergyComparison:
    """Compare EMG and Synergy-based neural network approaches."""

    def __init__(self, config: Dict):
        """Initialize comparison framework."""
        self.config = config
        self.output_path = Path(config['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logging()

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Results storage
        self.results = {
            'emg': {},
            'synergy': {}
        }
        self.models = {}
        self.trainers = {}
        self.histories = {}
        self.data = {}
        self.detailed_results = {}  # Store detailed results

    def _setup_logging(self):
        """Set up logging configuration."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_path / f'comparison_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> Tuple[Dict, Dict]:
        """Load both EMG and synergy data."""
        self.logger.info("Loading EMG data...")

        # Load EMG data
        emg_loader = EMGDataLoader(
            emg_path=self.config['emg_path'],
            ik_path=self.config['ik_path'],
            model_version=self.config.get('model_version', 'full_arm')
        )

        emg_data = emg_loader.prepare_data(
            test_split=self.config.get('test_split', 0.2),
            val_split=self.config.get('val_split', 0.1),
            sequence_length=self.config['sequence_length'],
            batch_size=self.config['batch_size'],
            downsample_rate=self.config.get('downsample_rate', 10)
        )

        self.logger.info("Loading synergy data...")

        # Load synergy data
        synergy_loader = MotionDataLoader(
            synergy_path=self.config['synergy_path'],
            ik_path=self.config['ik_path'],
            model_version=self.config.get('model_version', 'full_arm')
        )

        synergy_data = synergy_loader.prepare_data(
            test_split=self.config.get('test_split', 0.2),
            val_split=self.config.get('val_split', 0.1),
            sequence_length=self.config['sequence_length'],
            batch_size=self.config['batch_size']
        )

        # Store data for later use
        self.data['emg'] = emg_data
        self.data['synergy'] = synergy_data

        # Store muscle groups for analysis
        self.muscle_groups = emg_data['metadata'].get('muscle_groups', {})

        return emg_data, synergy_data

    def create_models(self, emg_data: Dict, synergy_data: Dict) -> None:
        """Create models for both EMG and synergy inputs."""
        # EMG model
        self.logger.info(f"Creating EMG model (input_dim: {emg_data['metadata']['input_dim']})")
        self.models['emg'] = create_improved_model(
            model_type=self.config['model_type'],
            input_dim=emg_data['metadata']['input_dim'],
            output_dim=emg_data['metadata']['output_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)

        # Synergy model
        self.logger.info(f"Creating synergy model (input_dim: {synergy_data['metadata']['input_dim']})")
        self.models['synergy'] = create_improved_model(
            model_type=self.config['model_type'],
            input_dim=synergy_data['metadata']['input_dim'],
            output_dim=synergy_data['metadata']['output_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)

        # Log model parameters
        for name, model in self.models.items():
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"{name} model parameters: {total_params:,}")

    def train_models(self, emg_data: Dict, synergy_data: Dict) -> None:
        """Train both models and compare performance."""
        data_dict = {'emg': emg_data, 'synergy': synergy_data}

        for data_type, data in data_dict.items():
            self.logger.info(f"\nTraining {data_type} model...")

            # Create trainer
            trainer = ImprovedTrainer(self.models[data_type], self.device)

            # Train model
            start_time = time.time()
            history = trainer.train(
                train_loader=data['train'],
                val_loader=data['val'],
                epochs=self.config['epochs'],
                learning_rate=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                patience=self.config['patience'],
                gradient_clip=self.config.get('gradient_clip', 1.0),
                warmup_epochs=self.config.get('warmup_epochs', 10),
                save_path=self.output_path / f'{data_type}_best.pth'
            )
            training_time = time.time() - start_time

            self.trainers[data_type] = trainer
            self.histories[data_type] = history
            history['training_time'] = training_time

            self.logger.info(f"{data_type} training completed in {training_time / 60:.2f} minutes")

    def evaluate_models(self, emg_data: Dict, synergy_data: Dict) -> None:
        """Evaluate and compare model performance with detailed joint-by-joint analysis."""
        detailed_evaluator = DetailedModelEvaluator(str(self.output_path))

        # Evaluate both models
        test_loaders = {'emg': emg_data['test'], 'synergy': synergy_data['test']}
        scalers = {'emg': emg_data['scalers'], 'synergy': synergy_data['scalers']}

        for data_type, test_loader in test_loaders.items():
            self.logger.info(f"\nEvaluating {data_type} model...")

            # Load best model
            self.trainers[data_type].load_model(str(self.output_path / f'{data_type}_best.pth'))

            # Get detailed results by joint
            joint_names = self.data[data_type]['metadata']['joint_names']
            detailed_results = detailed_evaluator.evaluate_model_detailed(
                self.models[data_type],
                test_loader,
                joint_names,
                f"{data_type}_{self.config['model_type']}"
            )
            self.detailed_results[data_type] = detailed_results

            # Overall evaluation
            results = self.trainers[data_type].evaluate_with_uncertainty(
                test_loader,
                n_samples=10,
                scalers=scalers[data_type]
            )

            self.results[data_type] = results

            self.logger.info(f"{data_type} results:")
            self.logger.info(f"  RMSE: {results['rmse']:.4f}")
            self.logger.info(f"  MAE: {results['mae']:.4f}")
            self.logger.info(f"  R²: {results['r2_mean']:.4f}")

        # Compare results
        self.compare_results()

    def compare_results(self) -> None:
        """Generate comparison analysis and visualizations."""
        # Create comparison metrics dataframe
        comparison_data = []

        for data_type in ['emg', 'synergy']:
            results = self.results[data_type]
            history = self.histories[data_type]

            comparison_data.append({
                'Method': 'Raw EMG' if data_type == 'emg' else 'Synergy',
                'Input Dimensions': 11 if data_type == 'emg' else self.config.get('n_synergies', 3),
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'R²': results['r2_mean'],
                'Training Time (min)': history['training_time'] / 60,
                'Best Epoch': history['best_epoch'],
                'Final Val Loss': history['val_loss'][-1],
                'Parameters': sum(p.numel() for p in self.models[data_type].parameters() if p.requires_grad)
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(self.output_path / 'comparison_metrics.csv', index=False)

        # Create comparison visualizations
        self._plot_training_comparison()
        self._plot_performance_comparison(comparison_df)
        self._plot_predictions_comparison_extended()
        self._plot_muscle_contribution_analysis()
        self._plot_joint_performance_comparison()

        # Generate report with detailed results
        self._generate_report(comparison_df)

    def _plot_training_comparison(self) -> None:
        """Plot training curves comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Training loss
        for data_type, history in self.histories.items():
            epochs = range(1, len(history['train_loss']) + 1)
            label = 'Raw EMG' if data_type == 'emg' else 'Synergy'
            ax1.plot(epochs, history['train_loss'], '-', label=f'{label} Train')
            ax1.plot(epochs, history['val_loss'], '--', label=f'{label} Val')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Learning rate
        for data_type, history in self.histories.items():
            if 'lr' in history:
                epochs = range(1, len(history['lr']) + 1)
                label = 'Raw EMG' if data_type == 'emg' else 'Synergy'
                ax2.plot(epochs, history['lr'], '-', label=label)

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_path / 'training_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_joint_performance_comparison(self) -> None:
        """Plot joint-by-joint performance comparison between EMG and Synergy models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # Get detailed results
        emg_detailed = self.detailed_results['emg']
        synergy_detailed = self.detailed_results['synergy']

        # Merge on joint name
        comparison = pd.merge(emg_detailed, synergy_detailed, on='Joint', suffixes=('_EMG', '_Synergy'))

        # Plot R² comparison
        joints = comparison['Joint'].str.split('/').str[-2] + ' ' + comparison['Joint'].str.split('/').str[-1]
        x = np.arange(len(joints))
        width = 0.35

        ax1.bar(x - width / 2, comparison['R2_EMG'], width, label='Raw EMG', color='#1f77b4', alpha=0.7)
        ax1.bar(x + width / 2, comparison['R2_Synergy'], width, label='Synergy', color='#ff7f0e', alpha=0.7)
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score by Joint')
        ax1.set_xticks(x)
        ax1.set_xticklabels(joints, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot RMSE comparison
        ax2.bar(x - width / 2, comparison['RMSE_EMG'], width, label='Raw EMG', color='#1f77b4', alpha=0.7)
        ax2.bar(x + width / 2, comparison['RMSE_Synergy'], width, label='Synergy', color='#ff7f0e', alpha=0.7)
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE by Joint')
        ax2.set_xticks(x)
        ax2.set_xticklabels(joints, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_path / 'joint_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create heatmap showing improvement
        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate improvement (positive means synergy is better)
        improvement_data = pd.DataFrame({
            'Joint': joints,
            'R² Improvement': comparison['R2_Synergy'] - comparison['R2_EMG'],
            'RMSE Improvement': comparison['RMSE_EMG'] - comparison['RMSE_Synergy'],  # Lower is better
            'MAE Improvement': comparison['MAE_EMG'] - comparison['MAE_Synergy']  # Lower is better
        })

        # Create heatmap
        improvement_matrix = improvement_data.set_index('Joint')[
            ['R² Improvement', 'RMSE Improvement', 'MAE Improvement']].T
        sns.heatmap(improvement_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'Improvement (Positive = Synergy Better)'})
        plt.title('Performance Improvement: Synergy vs Raw EMG')
        plt.tight_layout()
        plt.savefig(self.output_path / 'performance_improvement_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Plot performance metrics comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        metrics = ['RMSE', 'MAE', 'R²', 'Training Time (min)']

        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = comparison_df[metric].values
            methods = comparison_df['Method'].values

            bars = ax.bar(methods, values, color=['#1f77b4', '#ff7f0e'], alpha=0.7)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                format_str = f'{value:.3f}' if metric in ['RMSE', 'MAE', 'R²'] else f'{value:.1f}'
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        format_str, ha='center', va='bottom', fontweight='bold')

            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_predictions_comparison_extended(self) -> None:
        """Plot predictions comparison for key joints."""
        # Get predictions with proper scaling
        emg_test_loader = self.data['emg']['test']
        synergy_test_loader = self.data['synergy']['test']

        emg_predictions, emg_targets = self._get_predictions_for_visualization(
            emg_test_loader, self.models['emg'], self.data['emg']['scalers']
        )
        synergy_predictions, synergy_targets = self._get_predictions_for_visualization(
            synergy_test_loader, self.models['synergy'], self.data['synergy']['scalers']
        )

        # Define key joints for visualization (based on best performance from report)
        key_joints = {
            '/jointset/shoulder1/ra_sh_elv/value': 'Shoulder Elevation Angle',
            '/jointset/shoulder1/ra_sh_elv/speed': 'Shoulder Elevation Speed',
            '/jointset/RA1H_RA2U/ra_el_e_f/value': 'Elbow Flexion/Extension Angle',
            '/jointset/RA1H_RA2U/ra_el_e_f/speed': 'Elbow Flexion/Extension Speed',
            '/jointset/RA2R_RA3L/ra_wr_rd_ud/value': 'Wrist Radial/Ulnar Deviation Angle',
            '/jointset/RA2R_RA3L/ra_wr_rd_ud/speed': 'Wrist Radial/Ulnar Deviation Speed',
            '/jointset/RA2R_RA3L/ra_wr_e_f/value': 'Wrist Flexion/Extension Angle',
            '/jointset/RA2R_RA3L/ra_wr_e_f/speed': 'Wrist Flexion/Extension Speed',
        }

        # Get joint names from metadata
        joint_names = self.data['emg']['metadata']['joint_names']

        # Find indices of key joints
        joint_indices = []
        joint_labels = []
        for joint_path, label in key_joints.items():
            try:
                idx = joint_names.index(joint_path)
                joint_indices.append(idx)
                joint_labels.append(label)
            except ValueError:
                self.logger.warning(f"Joint {joint_path} not found in joint names")

        # Create subplot for each joint
        n_joints = len(joint_indices)
        fig, axes = plt.subplots(n_joints, 1, figsize=(12, 4 * n_joints))
        if n_joints == 1:
            axes = [axes]

        # Plot first 1000 samples for clarity
        n_samples = min(1000, len(emg_predictions))
        time = np.arange(n_samples) * 0.1  # 10Hz sampling

        for i, (joint_idx, label) in enumerate(zip(joint_indices, joint_labels)):
            ax = axes[i]

            # Plot ground truth
            ax.plot(time, emg_targets[:n_samples, joint_idx], 'k-', linewidth=2, label='Ground Truth')

            # Plot predictions
            ax.plot(time, emg_predictions[:n_samples, joint_idx], 'b--', linewidth=2, label='Raw EMG')
            ax.plot(time, synergy_predictions[:n_samples, joint_idx], 'r--', linewidth=2, label='Synergy')

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value')
            ax.set_title(f'{label} Predictions')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_path / 'predictions_comparison_extended.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _get_predictions_for_visualization(self, data_loader, model, scaler=None):
        """Get predictions from model for visualization."""
        model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)

                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y.numpy())

        predictions = np.vstack(predictions)
        targets = np.vstack(targets)

        # Apply inverse scaling if provided
        if scaler and 'y' in scaler:
            predictions = scaler['y'].inverse_transform(predictions)
            targets = scaler['y'].inverse_transform(targets)

        return predictions, targets

    def _plot_muscle_contribution_analysis(self) -> None:
        """Analyze and plot muscle contribution patterns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Create muscle contribution visualization
        muscles = ['BIClong', 'BICshort', 'BRA', 'TRIlong', 'TRIlat', 'TRImed',
                   'ECRL', 'ECRB', 'ECU', 'FCR', 'FCU']

        # Define muscle groups and their functions
        muscle_functions = {
            'Elbow Flexors': ['BIClong', 'BICshort', 'BRA'],
            'Elbow Extensors': ['TRIlong', 'TRIlat', 'TRImed'],
            'Wrist Extensors': ['ECRL', 'ECRB', 'ECU'],
            'Wrist Flexors': ['FCR', 'FCU']
        }

        # Create color map
        colors = []
        group_colors = {'Elbow Flexors': '#1f77b4', 'Elbow Extensors': '#ff7f0e',
                        'Wrist Extensors': '#2ca02c', 'Wrist Flexors': '#d62728'}

        for muscle in muscles:
            for group, group_muscles in muscle_functions.items():
                if muscle in group_muscles:
                    colors.append(group_colors[group])
                    break

        # Plot muscle grouping
        y_pos = np.arange(len(muscles))
        ax1.barh(y_pos, [1] * len(muscles), color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(muscles)
        ax1.set_xlabel('Muscle Groups')
        ax1.set_title('EMG Channel Functional Grouping')

        # Add legend
        for group, color in group_colors.items():
            ax1.barh([], [], color=color, label=group, alpha=0.7)
        ax1.legend()

        # Plot dimensionality comparison
        dimensions = ['Raw EMG', 'Synergy']
        values = [11, 3]
        bars = ax2.bar(dimensions, values, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        ax2.set_ylabel('Number of Input Dimensions')
        ax2.set_title('Dimensionality Comparison')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=14)

        # Add reduction percentage
        reduction = (values[0] - values[1]) / values[0] * 100
        ax2.text(0.5, max(values) * 0.5, f'{reduction:.1f}% reduction',
                 ha='center', va='center', fontsize=16, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.tight_layout()
        plt.savefig(self.output_path / 'muscle_contribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_report(self, comparison_df: pd.DataFrame) -> None:
        """Generate comprehensive comparison report with detailed joint analysis."""
        report_lines = [
            "# EMG vs Synergy Neural Network Comparison Report",
            "",
            "## Research Question",
            "How does muscle synergy (NMF) effectively reduce EMG dimensionality for neural control?",
            "",
            "## Experimental Setup",
            f"- Task: Elbow flexion/extension movement",
            f"- EMG channels: 11 (BIClong, BICshort, BRA, TRIlong, TRIlat, TRImed, ECRL, ECRB, ECU, FCR, FCU)",
            f"- Synergies: {self.config.get('n_synergies', 3)} (extracted using NMF)",
            f"- Neural Network: {self.config['model_type']}",
            f"- Sequence length: {self.config['sequence_length']}",
            f"- Hidden dimension: {self.config['hidden_dim']}",
            f"- Number of layers: {self.config['num_layers']}",
            "",
            "## Summary Results",
            "",
            comparison_df.to_markdown(index=False),
            "",
            "## Detailed Results by Joint - EMG Model",
            "",
            "| Model | Joint | MSE | RMSE | MAE | R² | Correlation |",
            "|-------|-------|-----|------|-----|-----|-------------|"
        ]

        # Add EMG detailed results
        for _, row in self.detailed_results['emg'].iterrows():
            report_lines.append(
                f"| {row['Model']} | {row['Joint']} | {row['MSE']:.6f} | {row['RMSE']:.6f} | "
                f"{row['MAE']:.6f} | {row['R2']:.6f} | {row['Correlation']:.6f} |"
            )

        report_lines.extend([
            "",
            "## Detailed Results by Joint - Synergy Model",
            "",
            "| Model | Joint | MSE | RMSE | MAE | R² | Correlation |",
            "|-------|-------|-----|------|-----|-----|-------------|"
        ])

        # Add Synergy detailed results
        for _, row in self.detailed_results['synergy'].iterrows():
            report_lines.append(
                f"| {row['Model']} | {row['Joint']} | {row['MSE']:.6f} | {row['RMSE']:.6f} | "
                f"{row['MAE']:.6f} | {row['R2']:.6f} | {row['Correlation']:.6f} |"
            )

        # Add performance comparison
        report_lines.extend([
            "",
            "## Performance Comparison by Joint",
            "",
            self._create_joint_comparison_table(),
            "",
            "## Key Findings",
            ""
        ])

        # Calculate improvements
        emg_row = comparison_df[comparison_df['Method'] == 'Raw EMG'].iloc[0]
        synergy_row = comparison_df[comparison_df['Method'] == 'Synergy'].iloc[0]

        rmse_improvement = (emg_row['RMSE'] - synergy_row['RMSE']) / emg_row['RMSE'] * 100
        mae_improvement = (emg_row['MAE'] - synergy_row['MAE']) / emg_row['MAE'] * 100
        training_speedup = emg_row['Training Time (min)'] / synergy_row['Training Time (min)']
        param_reduction = (emg_row['Parameters'] - synergy_row['Parameters']) / emg_row['Parameters'] * 100

        report_lines.extend([
            f"1. **Dimensionality Reduction**: {emg_row['Input Dimensions']} → {synergy_row['Input Dimensions']} dimensions ({(1 - synergy_row['Input Dimensions'] / emg_row['Input Dimensions']) * 100:.1f}% reduction)",
            f"2. **Overall Performance**:",
            f"   - RMSE: {'improved' if rmse_improvement > 0 else 'worsened'} by {abs(rmse_improvement):.1f}%",
            f"   - MAE: {'improved' if mae_improvement > 0 else 'worsened'} by {abs(mae_improvement):.1f}%",
            f"   - R²: EMG {emg_row['R²']:.3f} vs Synergy {synergy_row['R²']:.3f}",
            f"3. **Training Efficiency**:",
            f"   - Training time: {training_speedup:.1f}x faster with synergy",
            f"   - Model parameters: {param_reduction:.1f}% reduction",
            "",
            "## Top Performing Joints",
            ""
        ])

        # Add top performing joints analysis
        report_lines.extend(self._get_top_performing_joints_analysis())

        report_lines.extend([
            "",
            "## Conclusion",
            "",
            f"The synergy-based approach using {synergy_row['Input Dimensions']} dimensions achieved "
            f"{'comparable' if abs(rmse_improvement) < 5 else 'better' if rmse_improvement > 0 else 'worse'} "
            f"performance while reducing computational complexity by {param_reduction:.1f}% and training "
            f"time by {(1 - 1 / training_speedup) * 100:.1f}%.",
            "",
            "The results demonstrate that muscle synergy extraction effectively captures the essential "
            "control patterns while significantly reducing the dimensionality of the control space."
        ])

        with open(self.output_path / 'comparison_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

    def _create_joint_comparison_table(self) -> str:
        """Create a comparison table of joint performance."""
        emg_detailed = self.detailed_results['emg']
        synergy_detailed = self.detailed_results['synergy']

        # Merge results
        comparison = pd.merge(emg_detailed, synergy_detailed, on='Joint', suffixes=('_EMG', '_Synergy'))

        # Calculate improvements
        comparison['R²_Improvement'] = comparison['R2_Synergy'] - comparison['R2_EMG']
        comparison['RMSE_Improvement'] = comparison['RMSE_EMG'] - comparison['RMSE_Synergy']

        # Create table
        table_lines = [
            "| Joint | EMG R² | Synergy R² | R² Improvement | EMG RMSE | Synergy RMSE | RMSE Improvement |",
            "|-------|--------|------------|----------------|----------|--------------|------------------|"
        ]

        for _, row in comparison.iterrows():
            joint_display = row['Joint'].split('/')[-2] + ' ' + row['Joint'].split('/')[-1]
            table_lines.append(
                f"| {joint_display} | {row['R2_EMG']:.3f} | {row['R2_Synergy']:.3f} | "
                f"{row['R²_Improvement']:.3f} | {row['RMSE_EMG']:.3f} | "
                f"{row['RMSE_Synergy']:.3f} | {row['RMSE_Improvement']:.3f} |"
            )

        return '\n'.join(table_lines)

    def _get_top_performing_joints_analysis(self) -> List[str]:
        """Get analysis of top performing joints for both models."""
        analysis_lines = []

        # Get detailed results
        emg_detailed = self.detailed_results['emg']
        synergy_detailed = self.detailed_results['synergy']

        # Sort by R² for each model
        emg_top = emg_detailed.nlargest(5, 'R2')
        synergy_top = synergy_detailed.nlargest(5, 'R2')

        analysis_lines.append("### EMG Model - Top 5 Joints by R²:")
        analysis_lines.append("")
        for i, (_, row) in enumerate(emg_top.iterrows(), 1):
            joint_display = row['Joint'].split('/')[-2] + ' ' + row['Joint'].split('/')[-1]
            analysis_lines.append(f"{i}. {joint_display}: R² = {row['R2']:.3f}, RMSE = {row['RMSE']:.3f}")

        analysis_lines.append("")
        analysis_lines.append("### Synergy Model - Top 5 Joints by R²:")
        analysis_lines.append("")
        for i, (_, row) in enumerate(synergy_top.iterrows(), 1):
            joint_display = row['Joint'].split('/')[-2] + ' ' + row['Joint'].split('/')[-1]
            analysis_lines.append(f"{i}. {joint_display}: R² = {row['R2']:.3f}, RMSE = {row['RMSE']:.3f}")

        return analysis_lines

    def run_comparison(self) -> None:
        """Run the complete comparison experiment."""
        self.logger.info("Starting EMG vs Synergy comparison experiment...")

        # Load data
        emg_data, synergy_data = self.load_data()

        # Create models
        self.create_models(emg_data, synergy_data)

        # Train models
        self.train_models(emg_data, synergy_data)

        # Evaluate and compare with detailed joint analysis
        self.evaluate_models(emg_data, synergy_data)

        self.logger.info("Comparison experiment completed!")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compare EMG vs Synergy neural network approaches')
    parser.add_argument('--emg_path', type=str,
                        default=r"C:\temporary_file\BG_klinik\newPipeline\data\processed\emg_processed\emg_norm_db4.sto",
                        help='Path to EMG data')
    parser.add_argument('--synergy_path', type=str,
                        default=r"C:\temporary_file\BG_klinik\newPipeline\data\processed\synergies\synergy_extraction_n3\activation_patterns_smooth.csv",
                        help='Path to synergy data')
    parser.add_argument('--ik_path', type=str,
                        default=r"C:\temporary_file\BG_klinik\newPipeline\data\processed\Ik_result\N10_IK\kinematics_result01.mot",
                        help='Path to IK results')
    parser.add_argument('--output_path', type=str,
                        default=r"C:\temporary_file\BG_klinik\newPipeline\data\results\emg_synergy_comparison",
                        help='Path to save results')
    parser.add_argument('--model_type', type=str, default='attention_lstm',
                        help='Neural network model type')
    parser.add_argument('--n_synergies', type=int, default=3,
                        help='Number of synergies')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Sequence length for LSTM')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--downsample_rate', type=int, default=10,
                        help='Downsampling rate for EMG data')

    args = parser.parse_args()

    # Create configuration
    config = {
        'emg_path': args.emg_path,
        'synergy_path': args.synergy_path,
        'ik_path': args.ik_path,
        'output_path': args.output_path,
        'model_type': args.model_type,
        'n_synergies': args.n_synergies,
        'sequence_length': args.sequence_length,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'patience': args.patience,
        'downsample_rate': args.downsample_rate,
        'test_split': 0.2,
        'val_split': 0.1,
        'weight_decay': 1e-5,
        'gradient_clip': 0.5,
        'warmup_epochs': 5,
        'model_version': 'full_arm'
    }

    # Save configuration
    output_path = Path(config['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Run comparison
    comparison = EMGSynergyComparison(config)
    comparison.run_comparison()


if __name__ == "__main__":
    main()