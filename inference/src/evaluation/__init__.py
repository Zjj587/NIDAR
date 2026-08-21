"""
强度评估模块
包含点云球面投影、强度图生成、评估指标计算等功能
"""

from .spherical_projection import (
    project_to_range_image,
    project_to_range_image_waymo,
    lidar_to_front_view,
)
from .metrics import (
    compute_intensity_metrics,
    compute_all_metrics,
)
from .intensity_evaluator import IntensityEvaluator

__all__ = [
    'project_to_range_image',
    'project_to_range_image_waymo',
    'lidar_to_front_view',
    'compute_intensity_metrics',
    'compute_all_metrics',
    'IntensityEvaluator',
]
