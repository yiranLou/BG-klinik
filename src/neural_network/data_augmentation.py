# neural_network/data_augmentation.py
import numpy as np
import torch
from scipy.interpolate import interp1d
from typing import Tuple, Optional


class DataAugmentation:
    """Data augmentation techniques for time series data."""

    def __init__(self,
                 noise_level: float = 0.05,
                 time_warp_sigma: float = 0.2,
                 scale_sigma: float = 0.1,
                 augment_prob: float = 0.5):
        """
        Initialize data augmentation.

        Args:
            noise_level: Standard deviation of Gaussian noise
            time_warp_sigma: Magnitude of time warping
            scale_sigma: Magnitude of magnitude scaling
            augment_prob: Probability of applying augmentation
        """
        self.noise_level = noise_level
        self.time_warp_sigma = time_warp_sigma
        self.scale_sigma = scale_sigma
        self.augment_prob = augment_prob

    def add_noise(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to the data."""
        noise = np.random.normal(0, self.noise_level, x.shape)
        return x + noise

    def time_warp(self, x: np.ndarray) -> np.ndarray:
        """Apply time warping to the data."""
        orig_steps = np.arange(x.shape[0])

        # Generate smooth random warping
        random_warps = np.random.normal(loc=1.0, scale=self.time_warp_sigma,
                                        size=(len(orig_steps),))

        # Ensure monotonicity and bounds
        random_warps = np.cumsum(random_warps)
        random_warps = random_warps / random_warps[-1] * (len(orig_steps) - 1)

        # Interpolate
        warped = np.zeros_like(x)
        for i in range(x.shape[1]):
            f = interp1d(orig_steps, x[:, i], kind='linear',
                         fill_value='extrapolate')
            warped[:, i] = f(random_warps)

        return warped

    def magnitude_scale(self, x: np.ndarray) -> np.ndarray:
        """Apply magnitude scaling to the data."""
        scale = np.random.normal(loc=1.0, scale=self.scale_sigma,
                                 size=(1, x.shape[1]))
        return x * scale

    def window_slice(self, x: np.ndarray) -> np.ndarray:
        """Randomly slice a window from the data."""
        if x.shape[0] <= 4:
            return x

        # Random window size (70% to 100% of original)
        window_size = np.random.randint(int(0.7 * x.shape[0]), x.shape[0])

        # Random start position
        start = np.random.randint(0, x.shape[0] - window_size + 1)

        return x[start:start + window_size]

    def augment(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply random augmentation to the data."""
        if np.random.rand() > self.augment_prob:
            return x, y

        # Choose augmentation techniques randomly
        augmented_x = x.copy()

        if np.random.rand() < 0.5:
            augmented_x = self.add_noise(augmented_x)

        if np.random.rand() < 0.3:
            augmented_x = self.time_warp(augmented_x)

        if np.random.rand() < 0.3:
            augmented_x = self.magnitude_scale(augmented_x)

        return augmented_x, y


class MixupAugmentation:
    """Mixup augmentation for time series."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, x1: torch.Tensor, y1: torch.Tensor,
                 x2: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2

        return mixed_x, mixed_y


class CutMixAugmentation:
    """CutMix augmentation for time series."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, x1: torch.Tensor, y1: torch.Tensor,
                 x2: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cutmix augmentation."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        # Handle both 2D and 3D tensors
        if x1.dim() == 3:  # (batch, seq_len, features)
            seq_len = x1.size(1)
            cut_len = int(seq_len * (1 - lam))

            if cut_len > 0:
                cut_start = np.random.randint(0, seq_len - cut_len + 1)
                cut_end = cut_start + cut_len

                mixed_x = x1.clone()
                mixed_x[:, cut_start:cut_end, :] = x2[:, cut_start:cut_end, :]

                # Adjust target based on the proportion of mixing
                mixed_y = lam * y1 + (1 - lam) * y2
            else:
                mixed_x = x1
                mixed_y = y1

        elif x1.dim() == 2:  # (seq_len, features)
            seq_len = x1.size(0)
            cut_len = int(seq_len * (1 - lam))

            if cut_len > 0:
                cut_start = np.random.randint(0, seq_len - cut_len + 1)
                cut_end = cut_start + cut_len

                mixed_x = x1.clone()
                mixed_x[cut_start:cut_end, :] = x2[cut_start:cut_end, :]

                # Adjust target based on the proportion of mixing
                mixed_y = lam * y1 + (1 - lam) * y2
            else:
                mixed_x = x1
                mixed_y = y1
        else:
            # For other dimensions, just do regular mixup
            mixed_x = lam * x1 + (1 - lam) * x2
            mixed_y = lam * y1 + (1 - lam) * y2

        return mixed_x, mixed_y


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset with online augmentation."""

    def __init__(self,
                 original_dataset: torch.utils.data.Dataset,
                 augmentations: Optional[DataAugmentation] = None,
                 mixup: Optional[MixupAugmentation] = None,
                 cutmix: Optional[CutMixAugmentation] = None,
                 training: bool = True):
        self.dataset = original_dataset
        self.augmentations = augmentations
        self.mixup = mixup
        self.cutmix = cutmix
        self.training = training

    def __len__(self):
        return len(self.dataset)

    def train(self, mode: bool = True):
        """Set the training mode."""
        self.training = mode

    def eval(self):
        """Set the evaluation mode."""
        self.training = False

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        # Convert to numpy for augmentation
        x_np = x.numpy()
        y_np = y.numpy()

        # Apply basic augmentations
        if self.augmentations is not None and self.training:
            x_np, y_np = self.augmentations.augment(x_np, y_np)

        # Convert back to tensor
        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()

        # Apply mixup or cutmix (only during training)
        if self.training and (self.mixup is not None or self.cutmix is not None):
            # Get another random sample
            idx2 = np.random.randint(0, len(self.dataset))
            x2, y2 = self.dataset[idx2]

            if self.mixup is not None and np.random.rand() < 0.5:
                x, y = self.mixup(x, y, x2, y2)
            elif self.cutmix is not None:
                x, y = self.cutmix(x, y, x2, y2)

        return x, y