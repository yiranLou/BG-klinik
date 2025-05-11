# neural_network/data_loader.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class MotionDataset(Dataset):
    """Dataset for synergy-to-motion mapping."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 sequence_length: int = 10,
                 stride: int = 1):
        """
        Initialize dataset.

        Args:
            X: Input features (synergy activations)
            y: Target values (joint angles/velocities)
            sequence_length: Length of input sequences for LSTM
            stride: Stride between sequences
        """
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.stride = stride

        # Create sequences
        self.sequences = []
        for i in range(0, len(X) - sequence_length + 1, stride):
            seq_x = X[i:i + sequence_length]
            seq_y = y[i + sequence_length - 1]  # Predict at the end of sequence
            self.sequences.append((seq_x, seq_y))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_x, seq_y = self.sequences[idx]
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)


class MotionDataLoader:
    """Data loader for synergy-to-motion mapping."""

    def __init__(self, synergy_path: str, ik_path: str, model_version: str = "full_arm"):
        """
        Initialize data loader.

        Args:
            synergy_path: Path to synergy activation patterns
            ik_path: Path to IK results (joint angles/velocities)
            model_version: "simple_arm" for basic model, "full_arm" for hand-included model
        """
        self.synergy_path = Path(synergy_path)
        self.ik_path = Path(ik_path)
        self.model_version = model_version

        # Data containers
        self.synergy_data = None
        self.ik_data = None
        self.joint_names = None
        self.time_synergy = None
        self.time_ik = None

        # Preprocessing
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def load_synergy_data(self) -> None:
        """Load synergy activation patterns."""
        print(f"Loading synergy data from: {self.synergy_path}")

        # Load smooth activation patterns
        df = pd.read_csv(self.synergy_path, index_col=0)
        self.time_synergy = df.index.values
        self.synergy_data = df.values

        print(f"Loaded synergy data: {self.synergy_data.shape}")
        print(f"Synergy columns: {df.columns.tolist()}")

    def get_relevant_joint_columns(self) -> List[str]:
        """
        Get relevant joint columns based on muscle synergy analysis and model version.
        """
        if self.model_version == "simple_arm":
            # 原始的简单手臂模型关节
            return self._get_simple_arm_joints()
        else:
            # 包含手部的完整模型关节
            return self._get_full_arm_joints()

    def _get_simple_arm_joints(self) -> List[str]:
        """Get joint columns for simple arm model (without detailed hand)."""
        return [
            '/jointset/elbow/elbow_flexion/value',
            '/jointset/elbow/elbow_flexion/speed',
            '/jointset/radiocarpal/flexion/value',
            '/jointset/radiocarpal/flexion/speed',
            '/jointset/radiocarpal/deviation/value',
            '/jointset/radiocarpal/deviation/speed',
            '/jointset/radioulnar/pro_sup/value',
            '/jointset/radioulnar/pro_sup/speed',
            '/jointset/shoulder0/elv_angle/value',
            '/jointset/shoulder0/elv_angle/speed',
            '/jointset/shoulder1/shoulder_elv/value',
            '/jointset/shoulder1/shoulder_elv/speed',
            '/jointset/shoulder2/shoulder_rot/value',
            '/jointset/shoulder2/shoulder_rot/speed'
        ]

    def _get_full_arm_joints(self) -> List[str]:
        """Get joint columns for full arm model (with detailed hand)."""
        joint_columns = []

        # 1. 主要手臂关节（与肌肉协同直接相关）
        # 肘关节 - 对应Synergy_1(TRI群)和Synergy_2(BIC群)
        joint_columns.extend([
            '/jointset/RA1H_RA2U/ra_el_e_f/value',  # 肘屈伸角度
            '/jointset/RA1H_RA2U/ra_el_e_f/speed',  # 肘屈伸角速度
        ])

        # 前臂旋转 - 可能受PT(旋前圆肌)影响
        joint_columns.extend([
            '/jointset/RA2U_RA2R/ra_wr_sup_pro/value',  # 前臂旋前/旋后角度
            '/jointset/RA2U_RA2R/ra_wr_sup_pro/speed',  # 前臂旋前/旋后角速度
        ])

        # 手腕关节 - 对应Synergy_1(FCR/FCU)和Synergy_3(ECR群)
        joint_columns.extend([
            '/jointset/RA2R_RA3L/ra_wr_e_f/value',  # 腕屈伸角度
            '/jointset/RA2R_RA3L/ra_wr_e_f/speed',  # 腕屈伸角速度
            '/jointset/RA2R_RA3L/ra_wr_rd_ud/value',  # 腕尺桡偏角度
            '/jointset/RA2R_RA3L/ra_wr_rd_ud/speed',  # 腕尺桡偏角速度
        ])

        # 2. 肩关节（可能受协同影响）
        joint_columns.extend([
            '/jointset/shoulder0/ra_sh_elv_angle/value',
            '/jointset/shoulder0/ra_sh_elv_angle/speed',
            '/jointset/shoulder1/ra_sh_elv/value',
            '/jointset/shoulder1/ra_sh_elv/speed',
            '/jointset/shoulder2/ra_sh_rot/value',
            '/jointset/shoulder2/ra_sh_rot/speed',
        ])

        # 3. 手指关节（可选，根据您的研究需求）
        # 如果您的肌肉协同分析涉及手指控制，可以包含这些关节
        if self._include_finger_joints():
            # 拇指关节
            joint_columns.extend([
                '/jointset/RA3T_RA4M1_PHANT/ra_cmc1_f_e/value',  # 拇指腕掌关节屈伸
                '/jointset/RA3T_RA4M1_PHANT/ra_cmc1_f_e/speed',
                '/jointset/RA4M1_RA5P1/ra_mcp1_e_f/value',  # 拇指掌指关节屈伸
                '/jointset/RA4M1_RA5P1/ra_mcp1_e_f/speed',
                '/jointset/RA5P1_RA6D1/ra_ip1_e_f/value',  # 拇指指间关节屈伸
                '/jointset/RA5P1_RA6D1/ra_ip1_e_f/speed',
            ])

            # 食指到小指的MCP关节（掌指关节）
            for finger in range(2, 6):  # 2-5表示食指到小指
                joint_columns.extend([
                    f'/jointset/RA4M{finger}_RA5P{finger}/ra_mcp{finger}_e_f/value',
                    f'/jointset/RA4M{finger}_RA5P{finger}/ra_mcp{finger}_e_f/speed',
                ])

        return joint_columns

    def _include_finger_joints(self) -> bool:
        """
        Determine whether to include finger joints based on synergy analysis.
        如果您的EMG数据包含控制手指的肌肉（如FDS、FDP、EDC等），返回True
        """
        # 暂时设为False，如果需要手指数据可以改为True
        return False

    def load_ik_data(self, joint_columns: Optional[List[str]] = None) -> None:
        """Load IK results (joint angles and velocities)."""
        print(f"Loading IK data from: {self.ik_path}")
        print(f"Model version: {self.model_version}")

        # First, read header to get column names
        with open(self.ik_path, 'r') as f:
            lines = f.readlines()

        # Find the header line
        header_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('time'):
                header_idx = i
                break

        # Read the full file
        df = pd.read_csv(self.ik_path, sep='\t', skiprows=header_idx)
        self.time_ik = df.iloc[:, 0].values

        # Use default joint columns if none specified
        if joint_columns is None:
            joint_columns = self.get_relevant_joint_columns()

        # Extract specified columns
        available_columns = [col for col in joint_columns if col in df.columns]
        missing_columns = [col for col in joint_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")

        if not available_columns:
            # 检查是否使用了错误的模型版本
            print("No columns found. Available columns sample:")
            print(df.columns[:20].tolist())
            raise ValueError(f"None of the specified columns found in IK data.")

        self.ik_data = df[available_columns].values
        self.joint_names = available_columns

        print(f"Loaded IK data: {self.ik_data.shape}")
        print(f"Selected joints: {len(self.joint_names)} joints")
        for i, name in enumerate(self.joint_names):
            print(f"  [{i}] {name}")

    def align_time_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """Align synergy and IK time series."""
        # Find common time range
        t_start = max(self.time_synergy[0], self.time_ik[0])
        t_end = min(self.time_synergy[-1], self.time_ik[-1])

        # Interpolate both to common time grid
        common_time = np.linspace(t_start, t_end, min(len(self.time_synergy), len(self.time_ik)))

        # Interpolate synergy data
        synergy_aligned = np.zeros((len(common_time), self.synergy_data.shape[1]))
        for i in range(self.synergy_data.shape[1]):
            synergy_aligned[:, i] = np.interp(common_time, self.time_synergy, self.synergy_data[:, i])

        # Interpolate IK data
        ik_aligned = np.zeros((len(common_time), self.ik_data.shape[1]))
        for i in range(self.ik_data.shape[1]):
            ik_aligned[:, i] = np.interp(common_time, self.time_ik, self.ik_data[:, i])

        self.time_aligned = common_time

        print(f"\nTime series alignment:")
        print(f"  Original synergy: {len(self.time_synergy)} samples")
        print(f"  Original IK: {len(self.time_ik)} samples")
        print(f"  Aligned: {len(common_time)} samples")
        print(f"  Time range: [{t_start:.3f}, {t_end:.3f}] seconds")

        return synergy_aligned, ik_aligned

    def prepare_data(self, test_split: float = 0.2,
                     val_split: float = 0.1,
                     sequence_length: int = 10,
                     batch_size: int = 32) -> Dict[str, DataLoader]:
        """
        Prepare data loaders for training.
        """
        # Load data
        self.load_synergy_data()
        self.load_ik_data()

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
        print(f"  Input dimension: {X.shape[1]} (synergies)")
        print(f"  Output dimension: {y.shape[1]} (joints)")
        print(f"  Sequence length: {sequence_length}")

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
                'synergy_names': [f'Synergy_{i + 1}' for i in range(X.shape[1])]
            }
        }

    def visualize_data_mapping(self) -> None:
        """Visualize the relationship between synergies and joints."""
        print("\n=== Synergy-Joint Mapping ===")
        print("Based on muscle composition:")
        print("- Synergy_1 (TRI群+FCR/FCU): 肘伸展 + 腕屈曲")
        print("- Synergy_2 (BIC群): 肘屈曲")
        print("- Synergy_3 (ECR群): 腕伸展")

        if self.model_version == "full_arm":
            print("\nUsing full arm model with detailed hand joints")
        else:
            print("\nUsing simple arm model without detailed hand joints")

        if self.joint_names:
            print("\nSelected joints for prediction:")
            for i, joint in enumerate(self.joint_names):
                joint_desc = self._get_joint_description(joint)
                print(f"  [{i:2d}] {joint_desc}")

    def _get_joint_description(self, joint_path: str) -> str:
        """Get human-readable description of joint."""
        descriptions = {
            # Simple arm model joints
            '/jointset/elbow/elbow_flexion/value': '肘关节屈伸角度',
            '/jointset/elbow/elbow_flexion/speed': '肘关节屈伸角速度',
            '/jointset/radiocarpal/flexion/value': '腕关节屈伸角度',
            '/jointset/radiocarpal/flexion/speed': '腕关节屈伸角速度',
            '/jointset/radiocarpal/deviation/value': '腕关节尺桡偏角度',
            '/jointset/radiocarpal/deviation/speed': '腕关节尺桡偏角速度',
            '/jointset/radioulnar/pro_sup/value': '前臂旋前/旋后角度',
            '/jointset/radioulnar/pro_sup/speed': '前臂旋前/旋后角速度',
            '/jointset/shoulder0/elv_angle/value': '肩抬高角度',
            '/jointset/shoulder0/elv_angle/speed': '肩抬高角速度',
            '/jointset/shoulder1/shoulder_elv/value': '肩屈伸角度',
            '/jointset/shoulder1/shoulder_elv/speed': '肩屈伸角速度',
            '/jointset/shoulder2/shoulder_rot/value': '肩旋转角度',
            '/jointset/shoulder2/shoulder_rot/speed': '肩旋转角速度',

            # Full arm model joints
            '/jointset/RA1H_RA2U/ra_el_e_f/value': '肘关节屈伸角度',
            '/jointset/RA1H_RA2U/ra_el_e_f/speed': '肘关节屈伸角速度',
            '/jointset/RA2U_RA2R/ra_wr_sup_pro/value': '前臂旋前/旋后角度',
            '/jointset/RA2U_RA2R/ra_wr_sup_pro/speed': '前臂旋前/旋后角速度',
            '/jointset/RA2R_RA3L/ra_wr_e_f/value': '腕关节屈伸角度',
            '/jointset/RA2R_RA3L/ra_wr_e_f/speed': '腕关节屈伸角速度',
            '/jointset/RA2R_RA3L/ra_wr_rd_ud/value': '腕关节尺桡偏角度',
            '/jointset/RA2R_RA3L/ra_wr_rd_ud/speed': '腕关节尺桡偏角速度',
            '/jointset/shoulder0/ra_sh_elv_angle/value': '肩抬高角度',
            '/jointset/shoulder0/ra_sh_elv_angle/speed': '肩抬高角速度',
            '/jointset/shoulder1/ra_sh_elv/value': '肩屈伸角度',
            '/jointset/shoulder1/ra_sh_elv/speed': '肩屈伸角速度',
            '/jointset/shoulder2/ra_sh_rot/value': '肩旋转角度',
            '/jointset/shoulder2/ra_sh_rot/speed': '肩旋转角速度',

            # Hand joints
            '/jointset/RA3T_RA4M1_PHANT/ra_cmc1_f_e/value': '拇指腕掌关节屈伸角度',
            '/jointset/RA3T_RA4M1_PHANT/ra_cmc1_f_e/speed': '拇指腕掌关节屈伸角速度',
            '/jointset/RA4M1_RA5P1/ra_mcp1_e_f/value': '拇指掌指关节屈伸角度',
            '/jointset/RA4M1_RA5P1/ra_mcp1_e_f/speed': '拇指掌指关节屈伸角速度',
            '/jointset/RA5P1_RA6D1/ra_ip1_e_f/value': '拇指指间关节屈伸角度',
            '/jointset/RA5P1_RA6D1/ra_ip1_e_f/speed': '拇指指间关节屈伸角速度',
        }

        # 为其他手指生成描述
        finger_names = {2: '食指', 3: '中指', 4: '无名指', 5: '小指'}
        for finger in range(2, 6):
            descriptions[
                f'/jointset/RA4M{finger}_RA5P{finger}/ra_mcp{finger}_e_f/value'] = f'{finger_names[finger]}掌指关节屈伸角度'
            descriptions[
                f'/jointset/RA4M{finger}_RA5P{finger}/ra_mcp{finger}_e_f/speed'] = f'{finger_names[finger]}掌指关节屈伸角速度'

        return descriptions.get(joint_path, joint_path)