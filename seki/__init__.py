"""
SEKI (Symmetry-Equivariant Karcher Interpolation) 包
用於流形上的對稱等變 Karcher 插值實驗
"""

# 嘗試導入各模組，如果失敗則設為 None
try:
    from .augmentation import (
        seki_augmentation,
        mixup_augmentation,
        manifold_mixup_augmentation,
        r_mixup_augmentation,
        group_equivariant_augmentation
    )
except ImportError as e:
    print(f"Warning: Could not import augmentation module: {e}")
    seki_augmentation = None
    mixup_augmentation = None
    manifold_mixup_augmentation = None
    r_mixup_augmentation = None
    group_equivariant_augmentation = None

try:
    from .models import (
        StandardMLP,
        ManifoldMixupMLP, 
        GroupEquivariantMLP,
        SPDNet
    )
except ImportError as e:
    print(f"Warning: Could not import models module: {e}")
    StandardMLP = None
    ManifoldMixupMLP = None
    GroupEquivariantMLP = None
    SPDNet = None

try:
    from .evaluation import (
        evaluate_accuracy,
        expected_calibration_error,
        rotation_robustness_test,
        compute_comprehensive_metrics,
        compare_methods
    )
except ImportError as e:
    print(f"Warning: Could not import evaluation module: {e}")
    evaluate_accuracy = None
    expected_calibration_error = None
    rotation_robustness_test = None
    compute_comprehensive_metrics = None
    compare_methods = None

try:
    from .training import (
        TrainingConfig,
        train_standard_model,
        train_with_mixup,
        train_with_seki,
        cross_validate
    )
except ImportError as e:
    print(f"Warning: Could not import training module: {e}")
    TrainingConfig = None
    train_standard_model = None
    train_with_mixup = None
    train_with_seki = None
    cross_validate = None

__version__ = "0.1.0"
__author__ = "SEKI Team"

__all__ = [
    # Augmentation methods
    'seki_augmentation',
    'mixup_augmentation', 
    'manifold_mixup_augmentation',
    'r_mixup_augmentation',
    'group_equivariant_augmentation',
    
    # Models
    'StandardMLP',
    'ManifoldMixupMLP',
    'GroupEquivariantMLP', 
    'SPDNet',
    
    # Evaluation
    'evaluate_accuracy',
    'expected_calibration_error',
    'rotation_robustness_test',
    'compute_comprehensive_metrics',
    'compare_methods',
    
    # Training
    'TrainingConfig',
    'train_standard_model',
    'train_with_mixup',
    'train_with_seki',
    'cross_validate'
]

__version__ = "0.1.0"
__author__ = "SEKI Team"
__email__ = "your-email@example.com"

from .augmentation import (
    seki_augmentation,
    mixup_augmentation,
    r_mixup_augmentation,
    group_equivariant_augmentation
)

from .models import (
    StandardMLP,
    ManifoldMixupMLP,
    GroupEquivariantMLP,
    SPDNet
)

from .evaluation import (
    compute_comprehensive_metrics,
    expected_calibration_error,
    rotation_robustness_test
)

from .training import (
    train_standard_model,
    train_manifold_mixup_model,
    train_spd_model,
    TrainingConfig
)

__all__ = [
    # Augmentation methods
    'seki_augmentation',
    'mixup_augmentation', 
    'r_mixup_augmentation',
    'group_equivariant_augmentation',
    
    # Models
    'StandardMLP',
    'ManifoldMixupMLP',
    'GroupEquivariantMLP',
    'SPDNet',
    
    # Evaluation
    'compute_comprehensive_metrics',
    'expected_calibration_error',
    'rotation_robustness_test',
    
    # Training
    'train_standard_model',
    'train_manifold_mixup_model',
    'train_spd_model',
    'TrainingConfig',
]
