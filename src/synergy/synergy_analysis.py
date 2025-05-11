# synergy/synergy_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr
import torch


class SynergyAnalyzer:
    """Analyze and visualize extracted muscle synergies."""

    def __init__(self, synergy_path: str):
        """
        Initialize synergy analyzer.

        Args:
            synergy_path: Path to synergy extraction results
        """
        self.synergy_path = Path(synergy_path)
        self.synergies = None
        self.activations = None
        self.activations_smooth = None
        self.metrics = None

    def load_results(self) -> None:
        """Load synergy extraction results."""
        # Load synergy weights
        self.synergies = pd.read_csv(self.synergy_path / 'synergy_weights.csv', index_col=0)

        # Load activation patterns
        self.activations = pd.read_csv(self.synergy_path / 'activation_patterns.csv', index_col=0)

        # Load smooth activations if available
        smooth_path = self.synergy_path / 'activation_patterns_smooth.csv'
        if smooth_path.exists():
            self.activations_smooth = pd.read_csv(smooth_path, index_col=0)

        # Load metrics
        self.metrics = pd.read_csv(self.synergy_path / 'metrics.csv')

        print(f"Loaded {self.synergies.shape[1]} synergies from {self.synergies.shape[0]} muscles")

    def plot_synergy_weights(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot synergy weight matrix as a heatmap."""
        plt.figure(figsize=figsize)

        # Create heatmap
        sns.heatmap(self.synergies, annot=True, fmt='.3f', cmap='viridis',
                    cbar_kws={'label': 'Weight'})

        plt.title('Muscle Synergy Weights')
        plt.ylabel('Muscles')
        plt.xlabel('Synergies')
        plt.tight_layout()
        plt.savefig(self.synergy_path / 'synergy_weights.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_synergy_weights_bar(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """Plot synergy weights as bar plots."""
        n_synergies = self.synergies.shape[1]
        fig, axes = plt.subplots(n_synergies, 1, figsize=figsize, sharex=True)

        if n_synergies == 1:
            axes = [axes]

        for i, col in enumerate(self.synergies.columns):
            ax = axes[i]
            weights = self.synergies[col].sort_values(ascending=False)

            # Create bar plot
            bars = ax.bar(range(len(weights)), weights, color='skyblue', edgecolor='navy')

            # Customize
            ax.set_xticks(range(len(weights)))
            ax.set_xticklabels(weights.index, rotation=45, ha='right')
            ax.set_ylabel('Weight')
            ax.set_title(f'{col}')
            ax.grid(axis='y', alpha=0.3)

            # Highlight significant contributions
            for j, bar in enumerate(bars):
                if weights.iloc[j] > 0.2:  # Threshold for significant contribution
                    bar.set_color('orange')

        plt.suptitle('Muscle Synergy Weights by Synergy', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.synergy_path / 'synergy_weights_bar.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_activation_patterns(self, use_smooth: bool = True,
                                 figsize: Tuple[int, int] = (14, 8)) -> None:
        """Plot activation patterns over time."""
        activations = self.activations_smooth if use_smooth and self.activations_smooth is not None else self.activations

        n_synergies = activations.shape[1]
        fig, axes = plt.subplots(n_synergies, 1, figsize=figsize, sharex=True)

        if n_synergies == 1:
            axes = [axes]

        for i, col in enumerate(activations.columns):
            ax = axes[i]
            time = activations.index
            values = activations[col]

            # Plot activation
            ax.plot(time, values, linewidth=2, color='darkblue')
            ax.fill_between(time, 0, values, alpha=0.3, color='lightblue')

            # Customize
            ax.set_ylabel('Activation')
            ax.set_title(f'{col} Activation Pattern')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, None)

        axes[-1].set_xlabel('Time (s)')
        plt.suptitle('Synergy Activation Patterns', fontsize=14)
        plt.tight_layout()

        suffix = '_smooth' if use_smooth else ''
        plt.savefig(self.synergy_path / f'activation_patterns{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_reconstruction_quality(self, emg_data: np.ndarray,
                                    time: np.ndarray,
                                    muscle_idx: int = 0) -> None:
        """Plot original vs reconstructed EMG for a specific muscle."""
        # Reconstruct EMG
        W = self.synergies.values
        H = self.activations.values.T
        reconstructed = W @ H

        # Select muscle
        original = emg_data[muscle_idx, :]
        recon = reconstructed[muscle_idx, :]
        muscle_name = self.synergies.index[muscle_idx]

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(time, original, label='Original', linewidth=2, alpha=0.7)
        plt.plot(time, recon, label='Reconstructed', linewidth=2, alpha=0.7)

        # Calculate correlation
        corr, _ = pearsonr(original, recon)

        plt.title(f'EMG Reconstruction - {muscle_name} (r = {corr:.3f})')
        plt.xlabel('Time (s)')
        plt.ylabel('EMG Activation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(self.synergy_path / f'reconstruction_{muscle_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_synergy_contribution(self) -> pd.DataFrame:
        """Analyze the contribution of each synergy to each muscle."""
        # Calculate relative contributions
        contributions = self.synergies.div(self.synergies.sum(axis=1), axis=0)

        # Find dominant synergy for each muscle
        dominant_synergy = contributions.idxmax(axis=1)
        max_contribution = contributions.max(axis=1)

        # Create summary dataframe
        summary = pd.DataFrame({
            'dominant_synergy': dominant_synergy,
            'max_contribution': max_contribution
        })

        # Add contribution from each synergy
        for col in contributions.columns:
            summary[f'{col}_contribution'] = contributions[col]

        summary.to_csv(self.synergy_path / 'synergy_contributions.csv')
        return summary

    def calculate_synergy_similarity(self) -> np.ndarray:
        """Calculate similarity between synergies."""
        W = self.synergies.values
        n_synergies = W.shape[1]

        # Calculate cosine similarity
        similarity_matrix = np.zeros((n_synergies, n_synergies))

        for i in range(n_synergies):
            for j in range(n_synergies):
                w1, w2 = W[:, i], W[:, j]
                similarity = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
                similarity_matrix[i, j] = similarity

        # Plot similarity matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    xticklabels=self.synergies.columns,
                    yticklabels=self.synergies.columns,
                    center=0, vmin=-1, vmax=1)
        plt.title('Synergy Similarity Matrix (Cosine Similarity)')
        plt.tight_layout()
        plt.savefig(self.synergy_path / 'synergy_similarity.png', dpi=300, bbox_inches='tight')
        plt.close()

        return similarity_matrix

    def plot_activation_correlation(self) -> None:
        """Plot correlation between activation patterns."""
        # Calculate correlation matrix
        corr_matrix = self.activations.corr()

        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1)
        plt.title('Activation Pattern Correlations')
        plt.tight_layout()
        plt.savefig(self.synergy_path / 'activation_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self) -> None:
        """Generate a comprehensive analysis report."""
        report_lines = [
            "# Muscle Synergy Analysis Report",
            "",
            "## Summary",
            f"- Number of synergies: {self.synergies.shape[1]}",
            f"- Number of muscles: {self.synergies.shape[0]}",
            f"- Variance Accounted For (VAF): {self.metrics['vaf'].iloc[0]:.3f}",
            f"- R-squared: {self.metrics['r2'].iloc[0]:.3f}",
            f"- Reconstruction Error: {self.metrics['reconstruction_error'].iloc[0]:.6f}",
            ""
        ]

        # Add muscle contributions
        contributions = self.analyze_synergy_contribution()
        report_lines.extend([
            "## Muscle-Synergy Relationships",
            ""
        ])

        for muscle in contributions.index:
            dominant = contributions.loc[muscle, 'dominant_synergy']
            contrib = contributions.loc[muscle, 'max_contribution']
            report_lines.append(f"- {muscle}: Dominated by {dominant} ({contrib:.3f})")

        # Save report
        with open(self.synergy_path / 'analysis_report.md', 'w') as f:
            f.write('\n'.join(report_lines))

        print("Analysis report generated")

    def run_full_analysis(self) -> None:
        """Run complete synergy analysis pipeline."""
        print("Running full synergy analysis...")

        # Load results
        self.load_results()

        # Generate all plots
        print("Generating visualizations...")
        self.plot_synergy_weights()
        self.plot_synergy_weights_bar()
        self.plot_activation_patterns(use_smooth=True)
        self.plot_activation_patterns(use_smooth=False)
        self.plot_activation_correlation()

        # Perform analyses
        print("Analyzing synergy relationships...")
        self.analyze_synergy_contribution()
        self.calculate_synergy_similarity()

        # Generate report
        self.generate_report()

        print(f"Analysis complete. Results saved to: {self.synergy_path}")