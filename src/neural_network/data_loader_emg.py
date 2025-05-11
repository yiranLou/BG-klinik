# neural_network/data_loader_emg.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Import the base classes
from .data_loader import MotionDataLoader, MotionDataset


class EMGDataLoader(MotionDataLoader):
    """Data loader for EMG-to-motion mapping (without NMF)."""

    def __init__(self, emg_path: str, ik_path: str, model_version: str = "full_arm"):
        """
        Initialize EMG data loader.

        Args:
            emg_path: Path to processed EMG data (12 channels)
            ik_path: Path to IK results
            model_version: Model version for joint selection
        """
        # Initialize parent class with dummy synergy path
        super().__init__(synergy_path=emg_path, ik_path=ik_path, model_version=model_version)
        self.emg_path = Path(emg_path)
        self.emg_channel_names = []

    def load_emg_data(self) -> None:
        """Load multi-channel EMG data from .sto file."""
        print(f"Loading EMG data from: {self.emg_path}")

        # Read the .sto file
        with open(self.emg_path, 'r') as f:
            lines = f.readlines()

        # Find header information
        header_end_idx = None
        for i, line in enumerate(lines):
            if line.strip() == 'endheader':
                header_end_idx = i
                break

        # Read data - use whitespace separator for .sto files
        df = pd.read_csv(self.emg_path, sep='\s+', skiprows=header_end_idx + 1)

        # Extract time and EMG channels
        self.time_synergy = df.iloc[:, 0].values  # Use time_synergy to maintain compatibility
        self.synergy_data = df.iloc[:, 1:].values  # 11 EMG channels (BIClong to FCU)

        print(f"Loaded EMG data: {self.synergy_data.shape}")
        print(f"EMG channels: {df.columns[1:].tolist()}")

        # Store channel names
        self.emg_channel_names = df.columns[1:].tolist()

    def load_synergy_data(self) -> None:
        """Override to load EMG data instead of synergy."""
        self.load_emg_data()

    def get_muscle_function_groups(self) -> Dict[str, List[str]]:
        """Define muscle functional groups based on arm movements."""
        muscle_groups = {
            'elbow_flexors': ['BIClong', 'BICshort', 'BRA'],  # 肘屈肌群
            'elbow_extensors': ['TRIlong', 'TRIlat', 'TRImed'],  # 肘伸肌群
            'wrist_flexors': ['FCR', 'FCU'],  # 腕屈肌群
            'wrist_extensors': ['ECRL', 'ECRB', 'ECU'],  # 腕伸肌群
        }
        return muscle_groups

    def prepare_data(self, test_split: float = 0.2,
                     val_split: float = 0.1,
                     sequence_length: int = 10,
                     batch_size: int = 32,
                     downsample_rate: int = 10) -> Dict[str, DataLoader]:
        """
        Prepare data loaders with optional downsampling.

        Args:
            downsample_rate: Downsampling rate for EMG data (to match synergy processing)
        """
        # Load data
        self.load_synergy_data()  # This now loads EMG data
        self.load_ik_data()

        # Downsample EMG data to match synergy processing (10Hz)
        if downsample_rate > 1:
            self.synergy_data = self.synergy_data[::downsample_rate]
            self.time_synergy = self.time_synergy[::downsample_rate]
            print(f"Downsampled EMG data to {len(self.time_synergy)} samples")

        # Align time series
        X, y = self.align_time_series()

        # Split data
        n_samples = len(X)
        n_test = int(n_samples * test_split)
        n_train = n_samples - n_test
        n_val = int(n_train * val_split)
        n_train = n_train - n_val

        # Create splits
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

        # Fit scalers on training data
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        # Transform validation and test data
        X_val_scaled = self.scaler_x.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val)
        X_test_scaled = self.scaler_x.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test)

        # Create datasets
        train_dataset = MotionDataset(X_train_scaled, y_train_scaled, sequence_length)
        val_dataset = MotionDataset(X_val_scaled, y_val_scaled, sequence_length)
        test_dataset = MotionDataset(X_test_scaled, y_test_scaled, sequence_length)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"\nData statistics:")
        print(f"  Training samples: {n_train}")
        print(f"  Validation samples: {n_val}")
        print(f"  Test samples: {n_test}")
        print(f"  Input dimension: {X.shape[1]} (EMG channels)")
        print(f"  Output dimension: {y.shape[1]} (joints)")
        print(f"  Sequence length: {sequence_length}")

        # Display muscle function groups
        muscle_groups = self.get_muscle_function_groups()
        print(f"\nMuscle functional groups:")
        for group_name, muscles in muscle_groups.items():
            print(f"  {group_name}: {muscles}")

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'scalers': {'X': self.scaler_x, 'y': self.scaler_y},
            'metadata': {
                'input_dim': X.shape[1],
                'output_dim': y.shape[1],
                'sequence_length': sequence_length,
                'joint_names': self.joint_names,
                'input_names': self.emg_channel_names,  # EMG channel names
                'muscle_groups': muscle_groups
            }
        }