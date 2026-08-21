"""
STN 模型封装器
提供简单的加载和推理接口
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from .stn_net import STN, stn_init


class STNWrapper:
    """STN 模型的封装器,简化模型加载和推理"""

    def __init__(self, checkpoint_path, image_shape=(384, 1248), device='cuda'):
        """
        Args:
            checkpoint_path: 模型checkpoint路径
            image_shape: 输入图像的高度和宽度 (H, W)
            device: 'cuda' 或 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_shape = image_shape

        # 初始化模型
        self.model = STN(in_shape=image_shape, filt=True)
        self.model.apply(stn_init)

        # 加载权重
        # if os.path.exists(checkpoint_path):
        #     checkpoint = torch.load(checkpoint_path, map_location=self.device)
        #     if 'model_state_dict' in checkpoint:
        #         self.model.load_state_dict(checkpoint['model_state_dict'])
        #     else:
        #         self.model.load_state_dict(checkpoint)
        #     print(f"✓ 已加载STN模型: {checkpoint_path}")
        # else:
        #     raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 兼容多种checkpoint格式:
            # 1) 纯state_dict
            # 2) {'model_state_dict': ...}
            # 3) {'stnet': ..., 'dpnet': ..., ...} (cs-stereo风格)
            state_dict = None
            if isinstance(checkpoint, dict):
                if 'stnet' in checkpoint:
                    state_dict = checkpoint['stnet']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    # 有可能就是一个state_dict，但带了别的无关key
                    # 尝试直接当作state_dict加载
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[STNWrapper] Warning: missing keys when loading state_dict: {missing[:5]} ... (total {len(missing)})")
            if unexpected:
                print(f"[STNWrapper] Warning: unexpected keys in state_dict: {unexpected[:5]} ... (total {len(unexpected)})")

            print(f"✓ 已加载STN模型权重: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _pyramid(im, n_levels=4):
        """生成图像金字塔"""
        _, _, height, width = im.size()
        ims = [im]
        for i in range(1, n_levels):
            h = height // (2 ** i)
            w = width // (2 ** i)
            resized = F.interpolate(im, (h, w), mode='bilinear', align_corners=False)
            ims.append(resized)
        return ims

    def process_image(self, rgb_image, exp_ratio=0.5):
        """
        处理单张RGB图像,生成伪NIR图像

        Args:
            rgb_image: numpy数组 (H, W, 3), 值范围 [0, 255], uint8 或 float32
            exp_ratio: 曝光比率,控制输出亮度 (推荐 0.4-0.6)

        Returns:
            pseudo_nir: numpy数组 (H, W), 值范围 [0, 255], uint8
        """
        # 转换为 torch tensor
        if rgb_image.dtype == np.uint8:
            rgb_tensor = torch.from_numpy(rgb_image).float() / 255.0
        else:
            rgb_tensor = torch.from_numpy(rgb_image).float()

        # BGR to RGB (如果输入是OpenCV的BGR格式)
        # rgb_tensor = rgb_tensor[:, :, [2, 1, 0]]  # 根据需要解除注释

        # 调整为 (1, 3, H, W)
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 如果输入尺寸与模型不匹配,进行缩放
        _, _, h, w = rgb_tensor.shape
        if (h, w) != self.image_shape:
            rgb_tensor = F.interpolate(rgb_tensor, size=self.image_shape,
                                      mode='bilinear', align_corners=False)

        # 生成金字塔并推理
        with torch.no_grad():
            rgb_pyramid = self._pyramid(rgb_tensor, n_levels=4)
            nir_pyramid = self.model(rgb_pyramid, exp_ratio)
            pseudo_nir_tensor = nir_pyramid[0]  # 取最高分辨率

        # 如果进行了缩放,恢复原始尺寸
        if (h, w) != self.image_shape:
            pseudo_nir_tensor = F.interpolate(pseudo_nir_tensor, size=(h, w),
                                             mode='bilinear', align_corners=False)

        # 转换回 numpy
        pseudo_nir = pseudo_nir_tensor.squeeze().cpu().numpy()
        pseudo_nir = np.clip(pseudo_nir * 255, 0, 255).astype(np.uint8)

        return pseudo_nir

    def process_batch(self, rgb_images, exp_ratio=0.5):
        """
        批量处理RGB图像

        Args:
            rgb_images: numpy数组 (B, H, W, 3) 或列表 [array, ...]
            exp_ratio: 曝光比率

        Returns:
            pseudo_nirs: numpy数组 (B, H, W), uint8
        """
        if isinstance(rgb_images, list):
            return [self.process_image(img, exp_ratio) for img in rgb_images]
        else:
            batch_size = rgb_images.shape[0]
            results = []
            for i in range(batch_size):
                nir = self.process_image(rgb_images[i], exp_ratio)
                results.append(nir)
            return np.stack(results, axis=0)
