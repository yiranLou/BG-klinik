# neural_network/evaluator_improved.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from scipy import stats


class ModelEvaluator:
    """Evaluate and compare neural network models."""

    def __init__(self, output_path: str):
        """
        Initialize evaluator.

        Args:
            output_path: Path to save evaluation results
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def compare_models(self,
                       models: Dict[str, torch.nn.Module],
                       test_loader: torch.utils.data.DataLoader,
                       joint_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple models on test set.

        Args:
            models: Dictionary of model_name: model
            test_loader: Test data loader
            joint_names: Names of joints being predicted

        Returns:
            Comparison DataFrame
        """
        results = []

        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")

            # Get predictions
            predictions, targets = self._get_predictions(model, test_loader)

            # Calculate metrics for each joint
            for i, joint_name in enumerate(joint_names):
                y_true = targets[:, i]
                y_pred = predictions[:, i]

                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                # Additional metrics
                correlation, _ = stats.pearsonr(y_true, y_pred)
                max_error = np.max(np.abs(y_true - y_pred))

                # Percentage errors
                mask = np.abs(y_true) > 1e-6  # Avoid division by zero
                if np.any(mask):
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                else:
                    mape = np.nan

                results.append({
                    'Model': model_name,
                    'Joint': joint_name,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'Correlation': correlation,
                    'MaxError': max_error,
                    'MAPE': mape
                })

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Save results
        results_df.to_csv(self.output_path / 'model_comparison.csv', index=False)

        return results_df

    def _get_predictions(self, model: torch.nn.Module,
                         data_loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions."""
        model.eval()
        device = next(model.parameters()).device

        predictions = []
        targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)

                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y.numpy())

        return np.vstack(predictions), np.vstack(targets)

    def plot_model_comparison(self, results_df: pd.DataFrame) -> None:
        """Plot model comparison results."""
        metrics = ['RMSE', 'MAE', 'R2', 'Correlation']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Neural Network Model Performance Comparison', fontsize=16)
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Create grouped bar plot
            pivot_df = results_df.pivot(index='Joint', columns='Model', values=metric)
            pivot_df.plot(kind='bar', ax=ax)

            ax.set_title(f'{metric} by Joint and Model', fontsize=14)
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Model')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_path / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_predictions(self,
                         models: Dict[str, torch.nn.Module],
                         data_loader: torch.utils.data.DataLoader,
                         joint_names: List[str],
                         time_points: np.ndarray,
                         scalers: Dict = None) -> None:
        """Plot predictions vs ground truth for each model."""
        # Get predictions for each model
        model_predictions = {}
        for model_name, model in models.items():
            predictions, targets = self._get_predictions(model, data_loader)
            model_predictions[model_name] = predictions

        # Inverse transform if scalers provided
        if scalers and 'y' in scalers:
            targets = scalers['y'].inverse_transform(targets)
            for model_name in model_predictions:
                model_predictions[model_name] = scalers['y'].inverse_transform(
                    model_predictions[model_name]
                )

        # Create subplots for each joint
        n_joints = len(joint_names)
        n_cols = 2
        n_rows = (n_joints + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Model Predictions vs Ground Truth for Each Joint', fontsize=16)

        if n_joints == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, joint_name in enumerate(joint_names):
            ax = axes[i]

            # Plot ground truth
            ax.plot(time_points[:len(targets)], targets[:, i],
                    'k-', linewidth=2, label='Ground Truth')

            # Plot predictions for each model
            colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
            for (model_name, predictions), color in zip(model_predictions.items(), colors):
                ax.plot(time_points[:len(predictions)], predictions[:, i],
                        '--', color=color, linewidth=2, label=f'{model_name} Prediction')

            # Clean up joint name for display
            display_name = joint_name.split('/')[-2] + ' ' + joint_name.split('/')[-1]
            ax.set_title(f'{display_name}', fontsize=12)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_path / 'predictions_vs_ground_truth.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_error_distribution(self,
                                models: Dict[str, torch.nn.Module],
                                data_loader: torch.utils.data.DataLoader,
                                joint_names: List[str]) -> None:
        """Plot error distribution for each model and joint."""
        # Calculate errors for each model
        model_errors = {}

        for model_name, model in models.items():
            predictions, targets = self._get_predictions(model, data_loader)
            errors = predictions - targets
            model_errors[model_name] = errors

        # Create violin plots
        n_joints = len(joint_names)
        n_cols = 2
        n_rows = (n_joints + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        fig.suptitle('Prediction Error Distribution by Model and Joint', fontsize=16)

        if n_joints == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, joint_name in enumerate(joint_names):
            ax = axes[i]

            # Prepare data for violin plot
            data = []
            for model_name, errors in model_errors.items():
                joint_errors = errors[:, i]
                data.extend([(model_name, error) for error in joint_errors])

            df = pd.DataFrame(data, columns=['Model', 'Error'])

            # Create violin plot
            sns.violinplot(x='Model', y='Error', data=df, ax=ax)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

            # Clean up joint name for display
            display_name = joint_name.split('/')[-2] + ' ' + joint_name.split('/')[-1]
            ax.set_title(f'{display_name}', fontsize=12)
            ax.set_ylabel('Prediction Error')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_path / 'error_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_residuals(self,
                       models: Dict[str, torch.nn.Module],
                       data_loader: torch.utils.data.DataLoader,
                       joint_names: List[str]) -> None:
        """Plot residuals vs predicted values."""
        n_models = len(models)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        fig.suptitle('Residuals vs Predicted Values', fontsize=16)

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, model) in enumerate(models.items()):
            ax = axes[idx]
            predictions, targets = self._get_predictions(model, data_loader)

            # Plot residuals for each joint
            colors = plt.cm.Set1(np.linspace(0, 1, len(joint_names)))

            for i, (joint_name, color) in enumerate(zip(joint_names, colors)):
                residuals = targets[:, i] - predictions[:, i]
                display_name = joint_name.split('/')[-2] + ' ' + joint_name.split('/')[-1]

                ax.scatter(predictions[:, i], residuals, alpha=0.5, s=10,
                           color=color, label=display_name)

            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Predicted Value')
            ax.set_ylabel('Residual')
            ax.set_title(f'{model_name} Residuals')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_path / 'residuals_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, results_df: pd.DataFrame) -> None:
        """Generate evaluation report."""
        # Calculate summary statistics
        summary = results_df.groupby('Model').agg({
            'RMSE': 'mean',
            'MAE': 'mean',
            'R2': 'mean',
            'Correlation': 'mean',
            'MaxError': 'mean',
            'MAPE': 'mean'
        }).round(4)

        # Find best model for each metric
        best_models = {
            'RMSE': summary['RMSE'].idxmin(),
            'MAE': summary['MAE'].idxmin(),
            'R2': summary['R2'].idxmax(),
            'Correlation': summary['Correlation'].idxmax()
        }

        # Generate report
        report_lines = [
            "# Neural Network Model Comparison Report",
            "",
            "## Summary Statistics",
            "",
            "### Model Performance (averaged across all joints)",
            ""
        ]

        # Add summary table
        report_lines.append("| Model | RMSE | MAE | R² | Correlation | Max Error | MAPE (%) |")
        report_lines.append("|-------|------|-----|-----|-------------|-----------|----------|")
        for model in summary.index:
            rmse = summary.loc[model, 'RMSE']
            mae = summary.loc[model, 'MAE']
            r2 = summary.loc[model, 'R2']
            corr = summary.loc[model, 'Correlation']
            max_err = summary.loc[model, 'MaxError']
            mape = summary.loc[model, 'MAPE']
            report_lines.append(f"| {model} | {rmse} | {mae} | {r2} | {corr} | {max_err} | {mape:.1f} |")

        report_lines.extend([
            "",
            "## Best Models",
            "",
            f"- Best RMSE: {best_models['RMSE']} ({summary.loc[best_models['RMSE'], 'RMSE']})",
            f"- Best MAE: {best_models['MAE']} ({summary.loc[best_models['MAE'], 'MAE']})",
            f"- Best R²: {best_models['R2']} ({summary.loc[best_models['R2'], 'R2']})",
            f"- Best Correlation: {best_models['Correlation']} ({summary.loc[best_models['Correlation'], 'Correlation']})",
            "",
            "## Detailed Results by Joint",
            ""
        ])

        # Add detailed results table
        report_lines.append("| Model | Joint | MSE | RMSE | MAE | R² | Correlation |")
        report_lines.append("|-------|-------|-----|------|-----|-----|-------------|")
        for _, row in results_df.iterrows():
            model = row['Model']
            joint = row['Joint']
            mse = round(row['MSE'], 6)
            rmse = round(row['RMSE'], 6)
            mae = round(row['MAE'], 6)
            r2 = round(row['R2'], 6)
            corr = round(row['Correlation'], 6)
            report_lines.append(f"| {model} | {joint} | {mse} | {rmse} | {mae} | {r2} | {corr} |")

        # Save report with UTF-8 encoding
        with open(self.output_path / 'evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

    def create_training_comparison_plot(self, histories: Dict[str, Dict]) -> None:
        """Compare training histories of different models."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History Comparison Between Models', fontsize=16)

        colors = plt.cm.Set1(np.linspace(0, 1, len(histories)))

        # Loss comparison
        ax = axes[0, 0]
        for (model_name, history), color in zip(histories.items(), colors):
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], '-', color=color,
                    label=f'{model_name} Train')
            ax.plot(epochs, history['val_loss'], '--', color=color,
                    label=f'{model_name} Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate comparison (if available)
        ax = axes[0, 1]
        for (model_name, history), color in zip(histories.items(), colors):
            if 'lr' in history:
                epochs = range(1, len(history['lr']) + 1)
                ax.plot(epochs, history['lr'], '-', color=color,
                        label=model_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Training time comparison
        ax = axes[1, 0]
        for (model_name, history), color in zip(histories.items(), colors):
            epochs = range(1, len(history['train_time']) + 1)
            ax.plot(epochs, history['train_time'], '-', color=color,
                    label=model_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time per Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Best validation loss
        ax = axes[1, 1]
        model_names = []
        best_val_losses = []
        for model_name, history in histories.items():
            model_names.append(model_name)
            best_val_losses.append(history.get('best_val_loss', min(history['val_loss'])))

        bars = ax.bar(model_names, best_val_losses, color=colors)
        ax.set_ylabel('Best Validation Loss')
        ax.set_title('Best Validation Loss by Model')

        # Add value labels on bars
        for bar, val in zip(bars, best_val_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.4f}', ha='center', va='bottom')

        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_path / 'training_history_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()