# synergy/synergy_extraction.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import scipy.signal as signal
from .nmf import NMFSynergy
import os


class SynergyExtractor:
    """Extract muscle synergies from processed EMG data."""

    def __init__(self, data_path: str, output_path: str):
        """
        Initialize synergy extractor.

        Args:
            data_path: Path to processed EMG data (.sto file)
            output_path: Path to save synergy results
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.emg_data = None
        self.muscle_names = None
        self.time = None
        self.synergies = None
        self.activations = None

    def load_sto_file(self) -> None:
        """Load EMG data from OpenSim .sto file."""
        print(f"Loading data from: {self.data_path}")

        # Read the header to get muscle names
        with open(self.data_path, 'r') as f:
            lines = f.readlines()

        # Find the header line (usually starts with "time")
        header_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('time'):
                header_idx = i
                break

        # Extract muscle names
        header = lines[header_idx].strip().split('\t')
        self.muscle_names = header[1:]  # Exclude 'time'

        # Load data using pandas
        df = pd.read_csv(self.data_path, sep='\t', skiprows=header_idx)

        # Extract time and EMG data
        self.time = df.iloc[:, 0].values
        self.emg_data = df.iloc[:, 1:].values.T  # Transpose to get muscles x time

        print(f"Loaded EMG data: {self.emg_data.shape[0]} muscles x {self.emg_data.shape[1]} time points")
        print(f"Muscles: {self.muscle_names}")

    def preprocess_for_nmf(self, downsample_factor: int = 10,
                           smooth_window: int = 5) -> np.ndarray:
        """
        Preprocess EMG data for NMF analysis.

        Args:
            downsample_factor: Factor by which to downsample data
            smooth_window: Window size for smoothing

        Returns:
            Preprocessed EMG data
        """
        if self.emg_data is None:
            raise ValueError("EMG data not loaded. Call load_sto_file() first.")

        emg_processed = self.emg_data.copy()

        # Ensure non-negative values
        emg_processed = np.maximum(emg_processed, 0)

        # Smooth the data
        if smooth_window > 1:
            for i in range(emg_processed.shape[0]):
                emg_processed[i, :] = signal.savgol_filter(
                    emg_processed[i, :], smooth_window,
                    polyorder=2, mode='nearest'
                )

        # Downsample to reduce computational load
        if downsample_factor > 1:
            emg_processed = emg_processed[:, ::downsample_factor]
            self.time_downsampled = self.time[::downsample_factor]
        else:
            self.time_downsampled = self.time

        print(f"Preprocessed data shape: {emg_processed.shape}")
        return emg_processed

    def extract_synergies(self, n_synergies: int = 3,
                          use_gpu: bool = True,
                          max_iter: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract muscle synergies using NMF.

        Args:
            n_synergies: Number of synergies to extract
            use_gpu: Whether to use GPU acceleration
            max_iter: Maximum iterations for NMF

        Returns:
            W: Synergy weight matrix
            H: Activation coefficient matrix
        """
        # Preprocess data
        emg_processed = self.preprocess_for_nmf()

        # Initialize NMF
        nmf = NMFSynergy(
            n_components=n_synergies,
            max_iter=max_iter,
            use_gpu=use_gpu,
            init='random',
            random_state=42
        )

        # Extract synergies
        print(f"Extracting {n_synergies} synergies...")
        W, H = nmf.fit_transform(emg_processed)

        # Calculate metrics
        metrics = nmf.calculate_metrics(emg_processed)
        print(f"Variance Accounted For (VAF): {metrics['vaf']:.3f}")
        print(f"R-squared: {metrics['r2']:.3f}")

        # Store results
        self.synergies = W
        self.activations = H
        self.metrics = metrics

        return W, H

    def determine_optimal_synergies(self, max_synergies: int = 8,
                                    vaf_threshold: float = 0.9) -> int:
        """
        Determine optimal number of synergies based on VAF criterion.

        Args:
            max_synergies: Maximum number of synergies to test
            vaf_threshold: Target variance accounted for

        Returns:
            Optimal number of synergies
        """
        emg_processed = self.preprocess_for_nmf()
        vaf_values = []

        for n in range(1, max_synergies + 1):
            print(f"Testing {n} synergies...")
            nmf = NMFSynergy(n_components=n, use_gpu=True, max_iter=300)
            nmf.fit_transform(emg_processed)
            metrics = nmf.calculate_metrics(emg_processed)
            vaf_values.append(metrics['vaf'])

            if metrics['vaf'] >= vaf_threshold:
                print(f"Reached VAF threshold at {n} synergies")
                break

        # Save VAF curve
        self.save_vaf_curve(vaf_values)

        # Find elbow point or threshold crossing
        optimal_n = next((i + 1 for i, vaf in enumerate(vaf_values)
                          if vaf >= vaf_threshold), len(vaf_values))

        print(f"Optimal number of synergies: {optimal_n}")
        return optimal_n

    def smooth_activations(self, window_size: int = 5) -> np.ndarray:
        """Apply additional smoothing to activation patterns."""
        if self.activations is None:
            raise ValueError("No activations found. Extract synergies first.")

        H_smooth = np.zeros_like(self.activations)

        for i in range(self.activations.shape[0]):
            H_smooth[i, :] = signal.savgol_filter(
                self.activations[i, :], window_size,
                polyorder=2, mode='nearest'
            )

        return H_smooth

    def save_results(self, tag: str = "") -> None:
        """Save synergy extraction results."""
        if self.synergies is None or self.activations is None:
            raise ValueError("No results to save. Extract synergies first.")

        # Create subdirectory for this extraction
        save_dir = self.output_path / f"synergy_extraction_{tag}" if tag else self.output_path
        save_dir.mkdir(exist_ok=True)

        # Save synergy weights
        synergy_df = pd.DataFrame(self.synergies,
                                  index=self.muscle_names,
                                  columns=[f'Synergy_{i + 1}' for i in range(self.synergies.shape[1])])
        synergy_df.to_csv(save_dir / 'synergy_weights.csv')

        # Save activation patterns
        activation_df = pd.DataFrame(self.activations.T,
                                     index=self.time_downsampled,
                                     columns=[f'Synergy_{i + 1}' for i in range(self.activations.shape[0])])
        activation_df.index.name = 'time'
        activation_df.to_csv(save_dir / 'activation_patterns.csv')

        # Save smooth activations
        H_smooth = self.smooth_activations()
        activation_smooth_df = pd.DataFrame(H_smooth.T,
                                            index=self.time_downsampled,
                                            columns=[f'Synergy_{i + 1}' for i in range(H_smooth.shape[0])])
        activation_smooth_df.index.name = 'time'
        activation_smooth_df.to_csv(save_dir / 'activation_patterns_smooth.csv')

        # Save metrics
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(save_dir / 'metrics.csv', index=False)

        print(f"Results saved to: {save_dir}")

    def save_vaf_curve(self, vaf_values: List[float]) -> None:
        """Save VAF values for different numbers of synergies."""
        vaf_df = pd.DataFrame({
            'n_synergies': range(1, len(vaf_values) + 1),
            'vaf': vaf_values
        })
        vaf_df.to_csv(self.output_path / 'vaf_curve.csv', index=False)

    def run_extraction_pipeline(self, n_synergies: Optional[int] = None,
                                determine_optimal: bool = True) -> None:
        """
        Run the complete synergy extraction pipeline.

        Args:
            n_synergies: Number of synergies (if None, will determine optimal)
            determine_optimal: Whether to determine optimal number of synergies
        """
        # Load data
        self.load_sto_file()

        # Determine optimal number of synergies if needed
        if determine_optimal and n_synergies is None:
            n_synergies = self.determine_optimal_synergies()
        elif n_synergies is None:
            n_synergies = 3  # Default

        # Extract synergies
        self.extract_synergies(n_synergies=n_synergies)

        # Save results
        self.save_results(tag=f"n{n_synergies}")