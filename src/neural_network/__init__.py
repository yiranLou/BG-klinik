# neural_network/__init__.py
from .models import (
    TransformerModel,
    AttentionLSTM,
    ConvLSTM,
    ResidualLSTM,
    EnsembleModel,
    FeatureExtractor,
    EnhancedModel,
    create_improved_model
)
from .data_loader import MotionDataLoader, MotionDataset
from .data_loader_emg import EMGDataLoader  # 添加这行
from .trainer import ImprovedTrainer, CombinedLoss
from .evaluator import ModelEvaluator
from .data_augmentation import (
    DataAugmentation,
    MixupAugmentation,
    CutMixAugmentation,
    AugmentedDataset
)

__all__ = [
    # 模型
    'TransformerModel',
    'AttentionLSTM',
    'ConvLSTM',
    'ResidualLSTM',
    'EnsembleModel',
    'FeatureExtractor',
    'EnhancedModel',
    'create_improved_model',

    # 数据处理
    'MotionDataLoader',
    'MotionDataset',
    'EMGDataLoader',  # 添加这行

    # 训练与评估
    'ImprovedTrainer',
    'CombinedLoss',
    'ModelEvaluator',

    # 数据增强
    'DataAugmentation',
    'MixupAugmentation',
    'CutMixAugmentation',
    'AugmentedDataset'
]