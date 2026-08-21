"""
Deep Intrinsic 推理封装

用于在 pipeline 中替代原 Stage2 (Intrinsic) + Stage3 (投影) 的部分
直接从 pseudo-NIR 预测强度，同时输出 R/S 分解结果
"""
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DeepIntrinsicWrapper:
    """
    Deep Intrinsic + Intensity 模型的推理封装

    用法:
        wrapper = DeepIntrinsicWrapper(checkpoint_path, device='cuda')
        R, S, I_hat = wrapper.process_image(pseudo_nir)
    """

    def __init__(self,
                 checkpoint_path: str,
                 image_shape: Tuple[int, int] = (384, 1248),
                 device: str = 'cuda',
                 model_kwargs: dict = None):
        """
        Args:
            checkpoint_path: 模型 checkpoint 路径
            image_shape: 输入图像的 (H, W)
            device: 'cuda' 或 'cpu'
        """
        from src.models.deep_intrinsic_net import DeepIntrinsicIntensityNet

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_shape = image_shape

        # 加载模型
        # 支持通过 model_kwargs 传入与训练一致的网络参数（如 base_channels, intrinsic_net_type 等）
        if model_kwargs is None:
            # 向后兼容的默认参数（保持原行为）
            self.model = DeepIntrinsicIntensityNet(
                in_channels=1,
                base_channels=32,
                use_shading_for_intensity=False
            )
        else:
            # 确保 in_channels 默认为 1（除非显式指定）
            model_kwargs = dict(model_kwargs)
            model_kwargs.setdefault('in_channels', 1)
            self.model = DeepIntrinsicIntensityNet(**model_kwargs)

        # 加载权重
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 兼容不同格式
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            print(f"✓ 已加载 DeepIntrinsic 模型: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

    def decompose(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        兼容接口：分解图像为 Reflectance 和 Shading

        Args:
            image: numpy 数组 (H, W) 或 (H, W, 1), 值范围 [0, 255] uint8 或 [0, 1] float

        Returns:
            R: reflectance (H, W), [0, 255] uint8
            S: shading (H, W), [0, 255] uint8
        """
        # 确保是单通道
        if image.ndim == 3:
            if image.shape[2] == 1:
                image = image.squeeze(2)
            else:
                # 如果是3通道，取平均
                image = np.mean(image, axis=2).astype(image.dtype)

        R, S, _, _ = self.process_image(image, return_all=True)
        return R, S

    def process_image(self,
                      pseudo_nir: np.ndarray,
                      return_all: bool = True
                      ) -> Tuple[np.ndarray, ...]:
        """
        处理单张 pseudo-NIR 图像

        Args:
            pseudo_nir: numpy 数组 (H, W), 值范围 [0, 255] uint8 或 [0, 1] float
            return_all: 是否返回所有输出 (R, S, I_hat, I_recon)

        Returns:
            如果 return_all=True:
                R: reflectance (H, W), [0, 255] uint8
                S: shading (H, W), [0, 255] uint8 (归一化后)
                I_hat: 预测强度 (H, W), [0, 255] uint8
                I_recon: 重构图像 R*S (H, W), [0, 255] uint8
            否则:
                I_hat: 预测强度 (H, W), [0, 255] uint8
        """
        # 预处理
        if pseudo_nir.dtype == np.uint8:
            nir_tensor = torch.from_numpy(pseudo_nir).float() / 255.0
        else:
            nir_tensor = torch.from_numpy(pseudo_nir).float()

        # 调整维度: (H, W) -> (1, 1, H, W)
        nir_tensor = nir_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # 如果尺寸不匹配，进行缩放
        original_shape = nir_tensor.shape[2:]
        if original_shape != self.image_shape:
            nir_tensor = F.interpolate(
                nir_tensor,
                size=self.image_shape,
                mode='bilinear',
                align_corners=False
            )

        # 推理
        with torch.no_grad():
            R, S, I_hat, I_recon = self.model(nir_tensor)

        # 恢复原始尺寸
        if original_shape != self.image_shape:
            R = F.interpolate(R, size=original_shape, mode='bilinear', align_corners=False)
            S = F.interpolate(S, size=original_shape, mode='bilinear', align_corners=False)
            I_hat = F.interpolate(I_hat, size=original_shape, mode='bilinear', align_corners=False)
            I_recon = F.interpolate(I_recon, size=original_shape, mode='bilinear', align_corners=False)

        # 转为 numpy
        R_np = (R.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        I_hat_np = (I_hat.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        if return_all:
            # 归一化 S 用于显示
            S_np = S.squeeze().cpu().numpy()
            S_np = ((S_np / (S_np.max() + 1e-6)) * 255).clip(0, 255).astype(np.uint8)
            I_recon_np = (I_recon.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            return R_np, S_np, I_hat_np, I_recon_np
        else:
            return I_hat_np

    def process_batch(self,
                      pseudo_nirs: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量处理

        Args:
            pseudo_nirs: (B, H, W) numpy 数组

        Returns:
            R_batch: (B, H, W)
            S_batch: (B, H, W)
            I_hat_batch: (B, H, W)
        """
        results_R = []
        results_S = []
        results_I = []

        for i in range(pseudo_nirs.shape[0]):
            R, S, I_hat, _ = self.process_image(pseudo_nirs[i])
            results_R.append(R)
            results_S.append(S)
            results_I.append(I_hat)

        return np.stack(results_R), np.stack(results_S), np.stack(results_I)


def process_stage2_deep(config: dict, logger=None):
    """
    Deep 版本的 Stage2+3 处理函数

    可以作为 stage2_pseudo_nir_to_reflectance.py 中的替代路径
    """
    from src.utils.logger import setup_logger
    from src.utils.image_utils import load_image, save_image

    if logger is None:
        logger = setup_logger('stage2_deep')

    logger.info("=" * 60)
    logger.info("Stage 2+3 (Deep): Pseudo-NIR -> R/S/Intensity")
    logger.info("=" * 60)

    # 解析配置
    dataset_cfg = config['dataset']
    stage2_cfg = config.get('stage2', {})

    work_dir = dataset_cfg['work_dir']
    stage1_dir = work_dir / "stage1_pseudo_nir"

    # Deep 输出目录
    deep_output_dir = work_dir / "stage2_deep"
    deep_output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    checkpoint_path = stage2_cfg.get('deep_checkpoint')
    if checkpoint_path is None:
        logger.error("deep_checkpoint 未在配置中指定!")
        return 0

    device = 'cuda' if stage2_cfg.get('use_gpu', True) else 'cpu'

    logger.info(f"加载 Deep Intrinsic 模型: {checkpoint_path}")
    wrapper = DeepIntrinsicWrapper(
        checkpoint_path=checkpoint_path,
        image_shape=(384, 1248),
        device=device
    )

    # 处理每个 pseudo-NIR
    success_count = 0
    total_count = 0

    import glob
    nir_files = sorted(glob.glob(str(stage1_dir / "frame_*_cam*_nir.png")))

    for nir_path in nir_files:
        total_count += 1
        nir_path = Path(nir_path)

        # 解析文件名
        stem = nir_path.stem  # frame_000158_cam1_nir
        base_name = stem.rsplit('_', 1)[0]  # frame_000158_cam1

        # 输出路径
        output_r = deep_output_dir / f"{base_name}_r.png"
        output_s = deep_output_dir / f"{base_name}_s.png"
        output_i = deep_output_dir / f"{base_name}_intensity.png"

        # 跳过已存在
        skip_existing = stage2_cfg.get('skip_existing', True)
        if skip_existing and output_r.exists() and output_i.exists():
            logger.debug(f"跳过已存在: {base_name}")
            success_count += 1
            continue

        try:
            # 加载 pseudo-NIR
            nir_img = load_image(nir_path, color_mode='GRAY')

            # 推理
            R, S, I_hat, I_recon = wrapper.process_image(nir_img, return_all=True)

            # 保存
            save_image(R, output_r, color_mode='GRAY')
            save_image(S, output_s, color_mode='GRAY')
            save_image(I_hat, output_i, color_mode='GRAY')

            success_count += 1

            if total_count % 20 == 0:
                logger.info(f"已处理: {total_count} ({success_count} 成功)")

        except Exception as e:
            logger.error(f"处理失败 {base_name}: {e}")

    logger.info(f"Stage 2+3 (Deep) 完成: {success_count}/{total_count} 成功")
    logger.info(f"输出目录: {deep_output_dir}")

    return success_count


# ============ 测试代码 ============
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Deep Intrinsic checkpoint 路径')
    parser.add_argument('--input', type=str, required=True,
                        help='输入 pseudo-NIR 图像路径')
    parser.add_argument('--output_dir', type=str, default='./test_output',
                        help='输出目录')
    args = parser.parse_args()

    from PIL import Image

    # 加载模型
    wrapper = DeepIntrinsicWrapper(args.checkpoint, device='cuda')

    # 加载图像
    nir_img = np.array(Image.open(args.input).convert('L'))
    print(f"Input shape: {nir_img.shape}")

    # 推理
    R, S, I_hat, I_recon = wrapper.process_image(nir_img)

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(R).save(output_dir / 'R.png')
    Image.fromarray(S).save(output_dir / 'S.png')
    Image.fromarray(I_hat).save(output_dir / 'I_hat.png')
    Image.fromarray(I_recon).save(output_dir / 'I_recon.png')

    print(f"✓ 结果已保存到: {output_dir}")
