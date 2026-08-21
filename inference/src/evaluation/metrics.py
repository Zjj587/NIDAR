"""
强度评估指标计算模块
包含RMSE, MAE, PSNR, SSIM, LPIPS等指标
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings


def compute_rmse(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray = None) -> float:
    """计算RMSE (Root Mean Square Error)"""
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]
    return float(np.sqrt(np.mean((gt - pred) ** 2)))


def compute_mae(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray = None) -> float:
    """计算MAE (Mean Absolute Error)"""
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]
    return float(np.mean(np.abs(gt - pred)))


def compute_medae(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray = None) -> float:
    """计算MedAE (Median Absolute Error)"""
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]
    return float(np.median(np.abs(gt - pred)))


def compute_psnr(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray = None,
    max_val: float = 1.0,
) -> float:
    """计算PSNR (Peak Signal-to-Noise Ratio)"""
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]

    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(
    gt: np.ndarray,
    pred: np.ndarray,
    data_range: float = None,
) -> float:
    """
    计算SSIM (Structural Similarity Index)

    Args:
        gt: Ground truth图像
        pred: 预测图像
        data_range: 数据范围，如果为None则自动计算

    Returns:
        SSIM值
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        warnings.warn("skimage not available, SSIM will be NaN")
        return float('nan')

    if data_range is None:
        data_range = float(gt.max() - gt.min())

    if data_range < 1e-6:
        return 1.0  # 常值图像

    # 动态调整 win_size 以适应小图像
    min_dim = min(gt.shape[0], gt.shape[1])
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1  # 确保是奇数
    if win_size < 3:
        win_size = 3

    return float(structural_similarity(pred, gt, data_range=data_range, win_size=win_size))


def compute_lpips(
    gt_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    model=None,
    net: str = 'alex',
    device: str = 'cpu',
) -> float:
    """
    计算LPIPS (Learned Perceptual Image Patch Similarity)

    Args:
        gt_rgb: Ground truth RGB图像, (H, W, 3), [0, 1]
        pred_rgb: 预测RGB图像, (H, W, 3), [0, 1]
        model: 预加载的LPIPS模型（可选）
        net: 网络类型 'alex' 或 'vgg'
        device: 计算设备

    Returns:
        LPIPS值
    """
    try:
        import torch
        import lpips
    except ImportError:
        warnings.warn("torch or lpips not available, LPIPS will be NaN")
        return float('nan')

    try:
        # 使用预加载的模型或创建新模型
        if model is not None:
            loss_fn = model
        else:
            loss_fn = lpips.LPIPS(net=net).eval()
            if device != 'cpu':
                loss_fn = loss_fn.to(device)

        # 转换为tensor: (1, 3, H, W), 范围[-1, 1]
        gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).float()
        pred_tensor = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).float()

        gt_tensor = gt_tensor * 2.0 - 1.0
        pred_tensor = pred_tensor * 2.0 - 1.0

        if device != 'cpu':
            gt_tensor = gt_tensor.to(device)
            pred_tensor = pred_tensor.to(device)

        with torch.no_grad():
            lpips_val = loss_fn(gt_tensor, pred_tensor).item()

        return float(lpips_val)

    except Exception as e:
        warnings.warn(f"LPIPS computation failed: {e}")
        return float('nan')


def compute_intensity_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    min_intensity: float = 0.0,
    max_intensity: float = 1.0,
    lpips_model=None,
    gt_rgb: np.ndarray = None,
    pred_rgb: np.ndarray = None,
) -> Dict[str, float]:
    """
    计算强度图像的所有评估指标

    Args:
        gt: Ground truth灰度图像
        pred: 预测灰度图像
        min_intensity: 强度最小值
        max_intensity: 强度最大值
        lpips_model: 预加载的LPIPS模型（可选）
        gt_rgb: GT RGB图像用于LPIPS（可选）
        pred_rgb: Pred RGB图像用于LPIPS（可选）

    Returns:
        包含各项指标的字典
    """
    # 创建有效掩码（假设全部有效，如果需要可以扩展）
    valid_mask = np.ones_like(gt, dtype=bool)

    # 裁剪到有效范围
    gt_valid = np.clip(gt[valid_mask], min_intensity, max_intensity)
    pred_valid = np.clip(pred[valid_mask], min_intensity, max_intensity)

    # 基础指标
    rmse = float(np.sqrt(np.mean((gt_valid - pred_valid) ** 2)))
    mae = float(np.mean(np.abs(gt_valid - pred_valid)))
    medae = float(np.median(np.abs(gt_valid - pred_valid)))

    # PSNR
    mse = float(np.mean((pred_valid - gt_valid) ** 2))
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = float(10 * np.log10(max_intensity ** 2 / mse))

    # SSIM
    data_range = float(gt_valid.max() - gt_valid.min()) if gt_valid.size else 0.0
    if data_range < 1e-6:
        ssim_value = 1.0
    else:
        try:
            from skimage.metrics import structural_similarity
            # 根据图像大小动态调整窗口大小
            min_dim = min(gt.shape[0], gt.shape[1])
            if min_dim < 7:
                win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
                win_size = max(3, win_size)  # 最小窗口为3
            else:
                win_size = 7  # 默认窗口
            ssim_value = float(structural_similarity(pred, gt, data_range=data_range, win_size=win_size))
        except ImportError:
            ssim_value = float('nan')
        except Exception as e:
            warnings.warn(f"SSIM computation failed: {e}")
            ssim_value = float('nan')

    # LPIPS
    lpips_value = float('nan')
    if lpips_model is not None and gt_rgb is not None and pred_rgb is not None:
        try:
            import torch
            gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).float()
            pred_tensor = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).float()
            gt_tensor = gt_tensor * 2.0 - 1.0
            pred_tensor = pred_tensor * 2.0 - 1.0
            with torch.no_grad():
                lpips_value = float(lpips_model(gt_tensor, pred_tensor).item())
        except Exception as exc:
            warnings.warn(f"LPIPS computation failed: {exc}")

    return {
        'rmse': rmse,
        'mae': mae,
        'medae': medae,
        'psnr': psnr,
        'ssim': ssim_value,
        'lpips': lpips_value,
        'valid_pixels': int(valid_mask.sum()),
        'total_pixels': int(gt.size),
        'coverage': float(valid_mask.sum()) / float(gt.size),
    }


def compute_all_metrics(
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    compute_lpips: bool = True,
    lpips_net: str = 'alex',
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    计算所有评估指标的便捷函数

    Args:
        gt_img: Ground truth图像, BGR或灰度
        pred_img: 预测图像, BGR或灰度
        compute_lpips: 是否计算LPIPS
        lpips_net: LPIPS网络类型
        device: 计算设备

    Returns:
        指标字典
    """
    import cv2

    # 转换为灰度
    if gt_img.ndim == 3:
        gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        gt_gray = gt_img.astype(np.float32) / 255.0 if gt_img.max() > 1 else gt_img.astype(np.float32)

    if pred_img.ndim == 3:
        pred_gray = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        pred_gray = pred_img.astype(np.float32) / 255.0 if pred_img.max() > 1 else pred_img.astype(np.float32)

    # 计算基础指标
    metrics = compute_intensity_metrics(gt_gray, pred_gray)

    # 计算LPIPS（如果需要）
    if compute_lpips and gt_img.ndim == 3:
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        try:
            import torch
            import lpips as lpips_module
            lpips_model = lpips_module.LPIPS(net=lpips_net).eval()
            if device != 'cpu':
                lpips_model = lpips_model.to(device)

            gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).float()
            pred_tensor = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).float()
            gt_tensor = gt_tensor * 2.0 - 1.0
            pred_tensor = pred_tensor * 2.0 - 1.0

            if device != 'cpu':
                gt_tensor = gt_tensor.to(device)
                pred_tensor = pred_tensor.to(device)

            with torch.no_grad():
                metrics['lpips'] = float(lpips_model(gt_tensor, pred_tensor).item())
        except Exception:
            pass

    return metrics


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """格式化指标输出"""
    lines = []
    lines.append("=" * 40)
    lines.append("Intensity Evaluation Metrics")
    lines.append("=" * 40)

    key_order = ['rmse', 'mae', 'medae', 'psnr', 'ssim', 'lpips', 'coverage']
    key_names = {
        'rmse': 'RMSE',
        'mae': 'MAE',
        'medae': 'MedAE',
        'psnr': 'PSNR (dB)',
        'ssim': 'SSIM',
        'lpips': 'LPIPS',
        'coverage': 'Coverage',
    }

    for key in key_order:
        if key in metrics:
            value = metrics[key]
            name = key_names.get(key, key)
            if isinstance(value, float) and not np.isnan(value):
                lines.append(f"  {name:15s}: {value:.{precision}f}")
            else:
                lines.append(f"  {name:15s}: N/A")

    lines.append("=" * 40)
    return "\n".join(lines)
