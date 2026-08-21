"""
配置文件加载工具
"""
import os
import yaml
from pathlib import Path


def expand_env_vars(value):
    """Expand ${VAR} and ~ in YAML string values without changing their type."""
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def load_config(config_path):
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径 (str 或 Path)

    Returns:
        config: 配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config = expand_env_vars(config)

    # 转换路径为Path对象
    if 'dataset' in config:
        # Waymo/nuScenes 格式
        for key in ['image_dir', 'pointcloud_dir', 'calib_dir', 'work_dir']:
            if key in config['dataset']:
                config['dataset'][key] = Path(config['dataset'][key])

        # KITTI-360 格式
        if 'data_root' in config['dataset']:
            config['dataset']['data_root'] = Path(config['dataset']['data_root'])
        if 'work_dir' in config['dataset']:
            config['dataset']['work_dir'] = Path(config['dataset']['work_dir'])

    return config


def validate_config(config):
    """
    验证配置文件的完整性

    Args:
        config: 配置字典

    Raises:
        ValueError: 如果配置缺少必要字段
    """
    required_fields = ['dataset', 'stage1', 'stage2', 'stage3']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"配置文件缺少必要字段: {field}")

    dataset_fields = ['image_dir', 'pointcloud_dir', 'calib_dir', 'work_dir']
    for field in dataset_fields:
        if field not in config['dataset']:
            raise ValueError(f"dataset配置缺少必要字段: {field}")

    return True
