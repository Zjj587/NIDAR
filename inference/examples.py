#!/usr/bin/env python3
"""
使用示例脚本
演示如何单独使用各个模块
"""
import sys
import os
from pathlib import Path
import numpy as np

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def example_1_stn_inference():
    """示例1: 使用STN模型将RGB转换为伪NIR"""
    print("\n" + "=" * 60)
    print("示例 1: STN 推理")
    print("=" * 60)

    from src.models.stn_wrapper import STNWrapper
    from src.utils.image_utils import load_image, save_image

    # 初始化模型
    stn = STNWrapper(
        checkpoint_path=os.path.expandvars('${WEIGHTS_ROOT}/stn/47.pth'),
        image_shape=(384, 1248),
        device='cuda'  # 或 'cpu'
    )

    # 处理单张图像
    rgb_image = load_image('path/to/rgb_image.jpg', color_mode='RGB')
    pseudo_nir = stn.process_image(rgb_image, exp_ratio=0.5)
    save_image(pseudo_nir, 'output_nir.png', color_mode='GRAY')

    print("✓ 伪NIR图像已保存")


def example_2_load_waymo_data():
    """示例2: 使用Waymo数据集封装加载数据"""
    print("\n" + "=" * 60)
    print("示例 2: 加载 Waymo 数据")
    print("=" * 60)

    from src.datasets.waymo_dataset import WaymoDatasetWrapper

    # 初始化数据集
    dataset = WaymoDatasetWrapper(
        image_dir=Path('/path/to/waymo/images'),
        pointcloud_dir=Path('/path/to/waymo/pointclouds'),
    )

    print(f"发现 {len(dataset.frames)} 帧数据")

    # 加载第一帧
    frame_id = dataset.frames[0]
    frame_info = dataset.get_frame_info(frame_id)
    print(f"\n帧 {frame_id}:")
    print(f"  相机: {frame_info['cameras']}")

    # 加载图像
    cam_id = frame_info['cameras'][0]
    image = dataset.load_image(frame_id, cam_id, color_mode='RGB')
    print(f"  图像形状: {image.shape}")

    # 加载点云
    points = dataset.load_pointcloud(frame_id)
    print(f"  点云大小: {points.shape}")

    # 加载标定
    calib = dataset.load_calibration(frame_id, cam_id)
    print(f"  内参: {calib['intrinsic']}")


def example_3_projection():
    """示例3: 投影点云到图像"""
    print("\n" + "=" * 60)
    print("示例 3: 点云投影")
    print("=" * 60)

    from src.projection.projection_utils import (
        project_points_to_image,
        assign_intensity_from_image
    )
    from src.datasets.waymo_dataset import WaymoDatasetWrapper
    from src.utils.image_utils import load_image

    # 加载数据
    dataset = WaymoDatasetWrapper(
        image_dir=Path('/path/to/waymo/images'),
        pointcloud_dir=Path('/path/to/waymo/pointclouds'),
    )

    frame_id = dataset.frames[0]
    cam_id = 1

    # 加载点云和标定
    points = dataset.load_pointcloud(frame_id)
    calib = dataset.load_calibration(frame_id, cam_id)
    lidar_calib = dataset.load_lidar_calibration(frame_id)

    # 投影
    valid_points, uv_coords, point_indices = project_points_to_image(
        points=points,
        intrinsic=calib['intrinsic'],
        extrinsic=calib['extrinsic'],
        image_width=calib['width'],
        image_height=calib['height'],
        ego2world=lidar_calib['ego2world']
    )

    print(f"投影成功: {len(point_indices)}/{len(points)} 点")
    print(f"UV范围: [{uv_coords[:, 0].min():.1f}, {uv_coords[:, 0].max():.1f}] x "
          f"[{uv_coords[:, 1].min():.1f}, {uv_coords[:, 1].max():.1f}]")

    # 从反射率图像提取强度
    reflectance_img = load_image('path/to/reflectance.png', color_mode='GRAY')
    points_with_intensity = assign_intensity_from_image(
        points=points,
        reflectance_image=reflectance_img,
        intrinsic=calib['intrinsic'],
        extrinsic=calib['extrinsic'],
        ego2world=lidar_calib['ego2world']
    )

    print(f"强度范围: [{points_with_intensity[:, 3].min():.2f}, "
          f"{points_with_intensity[:, 3].max():.2f}]")


def example_4_multi_camera_fusion():
    """示例4: 多相机强度融合"""
    print("\n" + "=" * 60)
    print("示例 4: 多相机强度融合")
    print("=" * 60)

    from src.projection.projection_utils import merge_multi_camera_intensity
    from src.datasets.waymo_dataset import WaymoDatasetWrapper
    from src.utils.image_utils import load_image

    # 加载数据
    dataset = WaymoDatasetWrapper(
        image_dir=Path('/path/to/waymo/images'),
        pointcloud_dir=Path('/path/to/waymo/pointclouds'),
    )

    frame_id = dataset.frames[0]
    points = dataset.load_pointcloud(frame_id)
    lidar_calib = dataset.load_lidar_calibration(frame_id)

    # 收集所有相机的反射率图像和标定
    frame_info = dataset.get_frame_info(frame_id)
    reflectance_images = []
    intrinsics = []
    extrinsics = []

    for cam_id in frame_info['cameras']:
        # 加载反射率图像
        ref_path = f'path/to/reflectance/frame_{frame_id:06d}_cam{cam_id}_r.png'
        ref_img = load_image(ref_path, color_mode='GRAY')
        reflectance_images.append(ref_img)

        # 加载标定
        calib = dataset.load_calibration(frame_id, cam_id)
        intrinsics.append(calib['intrinsic'])
        extrinsics.append(calib['extrinsic'])

    # 融合
    points_fused = merge_multi_camera_intensity(
        points=points,
        reflectance_images=reflectance_images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        ego2world=lidar_calib['ego2world'],
        merge_strategy='max'  # 或 'mean', 'first'
    )

    print(f"融合完成: {len(reflectance_images)} 个相机")
    print(f"强度统计:")
    print(f"  最小值: {points_fused[:, 3].min():.2f}")
    print(f"  最大值: {points_fused[:, 3].max():.2f}")
    print(f"  平均值: {points_fused[:, 3].mean():.2f}")
    print(f"  中位数: {np.median(points_fused[:, 3]):.2f}")


def example_5_batch_processing():
    """示例5: 批量处理多帧"""
    print("\n" + "=" * 60)
    print("示例 5: 批量处理")
    print("=" * 60)

    from src.datasets.waymo_dataset import WaymoDatasetWrapper
    from src.models.stn_wrapper import STNWrapper
    from src.utils.image_utils import save_image
    from tqdm import tqdm

    # 初始化
    dataset = WaymoDatasetWrapper(
        image_dir=Path('/path/to/waymo/images'),
        pointcloud_dir=Path('/path/to/waymo/pointclouds'),
    )

    stn = STNWrapper(
        checkpoint_path=os.path.expandvars('${WEIGHTS_ROOT}/stn/47.pth'),
        image_shape=(384, 1248),
        device='cuda'
    )

    output_dir = Path('output_batch')
    output_dir.mkdir(exist_ok=True)

    # 批量处理
    for frame_id in tqdm(dataset.frames[:10], desc="Processing"):
        frame_info = dataset.get_frame_info(frame_id)

        for cam_id in frame_info['cameras']:
            # 加载RGB
            rgb_img = dataset.load_image(frame_id, cam_id, color_mode='RGB')

            # STN推理
            pseudo_nir = stn.process_image(rgb_img, exp_ratio=0.5)

            # 保存
            output_path = output_dir / f"frame_{frame_id:06d}_cam{cam_id}_nir.png"
            save_image(pseudo_nir, output_path, color_mode='GRAY')

    print(f"✓ 处理完成,输出: {output_dir}")


def example_6_custom_config():
    """示例6: 程序化创建配置"""
    print("\n" + "=" * 60)
    print("示例 6: 程序化配置")
    print("=" * 60)

    import yaml

    # 创建配置字典
    config = {
        'dataset': {
            'image_dir': '${DATA_ROOT}/waymo/example_scene/raw/images',
            'pointcloud_dir': '${DATA_ROOT}/waymo/example_scene/raw/pointclouds',
            'work_dir': '${OUTPUT_ROOT}/waymo_run1',
        },
        'stage1': {
            'checkpoint': '${WEIGHTS_ROOT}/stn/47.pth',
            'exp_ratio': 0.5,
            'bit_depth': 8,
            'use_gpu': True,
            'skip_existing': True,
        },
        'stage2': {
            'linear': False,
            'skip_existing': True,
        },
        'stage3': {
            'merge_strategy': 'max',
            'save_format': 'npy',
            'save_ply': True,
            'skip_existing': True,
        },
        'logging': {
            'level': 'INFO',
        }
    }

    # 保存为YAML
    output_config = Path('configs/generated_config.yaml')
    output_config.parent.mkdir(exist_ok=True)

    with open(output_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ 配置已保存: {output_config}")

    # 使用配置运行
    from src.utils.config import load_config
    loaded_config = load_config(output_config)
    print(f"✓ 配置已加载: {loaded_config['dataset']['work_dir']}")


def example_7_read_outputs():
    """示例7: 读取和分析输出结果"""
    print("\n" + "=" * 60)
    print("示例 7: 读取输出结果")
    print("=" * 60)

    from src.projection.projection_utils import load_pointcloud
    import matplotlib.pyplot as plt

    # 读取点云
    pointcloud_path = Path('work/stage3_pointcloud/frame_000001_pseudo_intensity.npy')
    points = load_pointcloud(pointcloud_path)

    print(f"点云形状: {points.shape}")
    print(f"X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Z范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print(f"强度范围: [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}]")

    # 统计分析
    intensity = points[:, 3]
    print(f"\n强度统计:")
    print(f"  均值: {intensity.mean():.2f}")
    print(f"  标准差: {intensity.std():.2f}")
    print(f"  中位数: {np.median(intensity):.2f}")
    print(f"  非零点: {(intensity > 0).sum()} / {len(intensity)}")

    # 可视化直方图
    plt.figure(figsize=(10, 6))
    plt.hist(intensity[intensity > 0], bins=50, edgecolor='black')
    plt.xlabel('Pseudo Intensity')
    plt.ylabel('Count')
    plt.title('Pseudo Intensity Distribution')
    plt.savefig('intensity_histogram.png')
    print(f"\n✓ 直方图已保存: intensity_histogram.png")


def main():
    """主函数 - 运行所有示例"""
    examples = [
        ("STN推理", example_1_stn_inference),
        ("加载Waymo数据", example_2_load_waymo_data),
        ("点云投影", example_3_projection),
        ("多相机融合", example_4_multi_camera_fusion),
        ("批量处理", example_5_batch_processing),
        ("程序化配置", example_6_custom_config),
        ("读取输出", example_7_read_outputs),
    ]

    print("\n" + "=" * 60)
    print("Waymo 伪强度管道 - 使用示例")
    print("=" * 60)
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\n注意: 这些示例需要修改路径才能运行")
    print("请根据实际情况修改代码中的路径\n")

    # 选择运行示例 (这里只演示结构)
    # 实际使用时取消注释:
    # choice = input("选择示例 (1-7) 或 'all': ")
    # if choice == 'all':
    #     for name, func in examples:
    #         try:
    #             func()
    #         except Exception as e:
    #             print(f"✗ {name} 失败: {e}")
    # else:
    #     idx = int(choice) - 1
    #     examples[idx][1]()


if __name__ == '__main__':
    main()
