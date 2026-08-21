"""
Deep Intrinsic + Intensity Network
保持 Intrinsic 的物理含义 (I ≈ R × S)，同时用 LiDAR 强度做监督

输入: pseudo-NIR (1, H, W)
输出: R (reflectance), S (shading), I_hat (intensity prediction)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """基础卷积块：Conv + BN + ELU"""
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """编码器块：下采样 + 两层卷积"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = ConvBlock(in_c, out_c, stride=2)  # 下采样
        self.conv2 = ConvBlock(out_c, out_c)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderBlock(nn.Module):
    """解码器块：上采样 + skip connection + 两层卷积"""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.conv1 = ConvBlock(in_c + skip_c, out_c)
        self.conv2 = ConvBlock(out_c, out_c)

    def forward(self, x, skip, target_size):
        # 上采样到目标尺寸
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        # 拼接 skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class IntrinsicUNet(nn.Module):
    """
    U-Net 风格的本征分解网络

    输入: (B, 1, H, W) pseudo-NIR
    输出: R (B, 1, H, W), S (B, 1, H, W)
    """
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        # 通道数配置 [32, 64, 128, 256, 512]
        nf = [base_channels * (2 ** i) for i in range(5)]

        # 初始卷积
        self.init_conv = ConvBlock(in_channels, nf[0])

        # 编码器
        self.enc1 = EncoderBlock(nf[0], nf[1])  # /2
        self.enc2 = EncoderBlock(nf[1], nf[2])  # /4
        self.enc3 = EncoderBlock(nf[2], nf[3])  # /8
        self.enc4 = EncoderBlock(nf[3], nf[4])  # /16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(nf[4], nf[4]),
            ConvBlock(nf[4], nf[4])
        )

        # 解码器
        self.dec4 = DecoderBlock(nf[4], nf[3], nf[3])
        self.dec3 = DecoderBlock(nf[3], nf[2], nf[2])
        self.dec2 = DecoderBlock(nf[2], nf[1], nf[1])
        self.dec1 = DecoderBlock(nf[1], nf[0], nf[0])

        # R head: reflectance 输出
        self.r_head = nn.Sequential(
            ConvBlock(nf[0], nf[0] // 2),
            nn.Conv2d(nf[0] // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # R ∈ [0, 1]
        )

        # S head: shading 输出
        self.s_head = nn.Sequential(
            ConvBlock(nf[0], nf[0] // 2),
            nn.Conv2d(nf[0] // 2, 1, kernel_size=3, padding=1),
            nn.Softplus()  # S > 0，用 softplus 保证正值
        )

    def forward(self, x):
        # 保存输入尺寸用于解码器
        sizes = [x.shape[2:]]  # [(H, W)]

        # 初始卷积
        x0 = self.init_conv(x)
        sizes.append(x0.shape[2:])

        # 编码
        x1 = self.enc1(x0)
        sizes.append(x1.shape[2:])
        x2 = self.enc2(x1)
        sizes.append(x2.shape[2:])
        x3 = self.enc3(x2)
        sizes.append(x3.shape[2:])
        x4 = self.enc4(x3)

        # Bottleneck
        x4 = self.bottleneck(x4)

        # 解码
        d4 = self.dec4(x4, x3, sizes[4])
        d3 = self.dec3(d4, x2, sizes[3])
        d2 = self.dec2(d3, x1, sizes[2])
        d1 = self.dec1(d2, x0, sizes[1])

        # 输出 R 和 S
        R = self.r_head(d1)
        S = self.s_head(d1)

        return R, S


class IntensityHead(nn.Module):
    """
    从 R (和可选的 S) 预测 LiDAR 强度

    这个 head 学习 reflectance -> intensity 的映射关系。
    可配置更大的 hidden_channels、更多层数、可选 BatchNorm 与更大卷积核。
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 3,
        kernel_size: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()

        assert num_layers >= 2, "IntensityHead: num_layers 至少为 2 (最后一层为输出 1 通道)"
        assert kernel_size % 2 == 1, "IntensityHead: kernel_size 需为奇数以保持空间尺寸"

        layers = []
        in_c = in_channels
        pad = kernel_size // 2

        # 前面 (num_layers-1) 个块：Conv(+BN)+ELU，保持空间分辨率
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(in_c, hidden_channels, kernel_size=kernel_size, padding=pad))
            if use_bn:
                layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ELU(inplace=True))
            in_c = hidden_channels

        # 最后一层映射到 1 通道 + Sigmoid 归一化到 [0,1]
        layers.append(nn.Conv2d(in_c, 1, kernel_size=1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, R, S=None):
        # 默认仅用 R；若需要，可拼接 S 一并作为输入（需保证 in_channels=2）
        if S is not None:
            x = torch.cat([R, S], dim=1)
        else:
            x = R
        return self.net(x)


class DeepIntrinsicIntensityNet(nn.Module):
    """
    完整的 Deep Intrinsic + Intensity 网络

    结合本征分解 (R, S) 和 LiDAR 强度预测
    保持物理约束: I ≈ R × S

    支持多种网络架构（通过 YAML 配置选择）：
    - intrinsic_net_type: 'unet' | 'deeplab'
    - intensity_head_type: 'conv' | 'unet' | 'deeplab' | None
      - 当设为 None 时，不使用独立的强度预测头，直接用 I_recon = R*S 作为强度预测
        所有强度损失都会作用在 intrinsic 网络上
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        use_shading_for_intensity: bool = False,
        # Intensity head 可选配置（保持向后兼容）
        intensity_hidden_channels: int = 32,
        intensity_depth: int = 3,
        intensity_kernel_size: int = 3,
        intensity_use_bn: bool = True,
        # 新增：网络类型选择
        intrinsic_net_type: str = 'unet',       # 'unet' | 'deeplab'
        intensity_head_type: str = 'conv',      # 'conv' | 'unet' | 'deeplab' | None
        # DeepLab 特定参数
        deeplab_base_channels: int = 32,
        deeplab_output_stride: int = 16,
    ):
        super().__init__()

        self.use_shading_for_intensity = use_shading_for_intensity
        self.intrinsic_net_type = intrinsic_net_type
        self.intensity_head_type = intensity_head_type

        # ========== 本征分解网络 ==========
        if intrinsic_net_type == 'unet':
            self.intrinsic_net = IntrinsicUNet(in_channels, base_channels)
        elif intrinsic_net_type == 'deeplab':
            from .network_variants import IntrinsicDeepLab
            self.intrinsic_net = IntrinsicDeepLab(
                in_channels=in_channels,
                base_channels=deeplab_base_channels,
                output_stride=deeplab_output_stride
            )
        else:
            raise ValueError(f"Unknown intrinsic_net_type: {intrinsic_net_type}. "
                           f"Choose from ['unet', 'deeplab']")

        # ========== 强度预测 head ==========
        # 当 intensity_head_type 为 None 时，不使用独立的强度预测头
        # 直接用 I_recon = R*S 作为强度预测，所有强度损失作用在 intrinsic 网络上
        if intensity_head_type is None:
            self.intensity_head = None
        else:
            intensity_in_c = 2 if use_shading_for_intensity else 1

            if intensity_head_type == 'conv':
                # 原始卷积堆叠方式
                self.intensity_head = IntensityHead(
                    in_channels=intensity_in_c,
                    hidden_channels=intensity_hidden_channels,
                    num_layers=intensity_depth,
                    kernel_size=intensity_kernel_size,
                    use_bn=intensity_use_bn,
                )
            elif intensity_head_type == 'unet':
                # 轻量 U-Net
                from .network_variants import IntensityUNet
                self.intensity_head = IntensityUNet(
                    in_channels=intensity_in_c,
                    base_channels=intensity_hidden_channels,
                    depth=intensity_depth
                )
            elif intensity_head_type == 'deeplab':
                # DeepLabV3+
                from .network_variants import IntensityDeepLab
                self.intensity_head = IntensityDeepLab(
                    in_channels=intensity_in_c,
                    base_channels=deeplab_base_channels,
                    output_stride=deeplab_output_stride
                )
            else:
                raise ValueError(f"Unknown intensity_head_type: {intensity_head_type}. "
                               f"Choose from ['conv', 'unet', 'deeplab', None]")

    def forward(self, x):
        """
        Args:
            x: pseudo-NIR 图像, shape (B, 1, H, W)

        Returns:
            R: reflectance, shape (B, 1, H, W)
            S: shading, shape (B, 1, H, W)
            I_hat: predicted intensity, shape (B, 1, H, W)
            I_recon: reconstructed image R*S, shape (B, 1, H, W)
        """
        # 本征分解
        R, S = self.intrinsic_net(x)

        # 重构
        I_recon = R * S

        # 强度预测
        if self.intensity_head is None:
            # 无独立强度头时，直接用 I_recon 作为 I_hat
            # 这样所有强度损失都会反传到 intrinsic 网络
            I_hat = I_recon
        elif self.use_shading_for_intensity:
            I_hat = self.intensity_head(R, S)
        else:
            I_hat = self.intensity_head(R)

        return R, S, I_hat, I_recon

    def get_intrinsic_only(self, x):
        """只获取本征分解结果，用于可视化"""
        R, S = self.intrinsic_net(x)
        I_recon = R * S
        return R, S, I_recon


# ============ 测试代码 ============
if __name__ == '__main__':

    def count_params(model):
        return sum(p.numel() for p in model.parameters()) / 1e6

    def test_model(model, x, name):
        R, S, I_hat, I_recon = model(x)
        print(f"\n{name}:")
        print(f"  Input: {x.shape} -> R: {R.shape}, S: {S.shape}, I_hat: {I_hat.shape}")
        print(f"  参数量: {count_params(model):.2f}M")
        return True

    # 模拟输入
    x = torch.randn(2, 1, 384, 1248)  # Waymo 尺寸

    print("=" * 60)
    print("测试不同网络架构组合")
    print("=" * 60)

    # 1. 默认配置: UNet + Conv
    model = DeepIntrinsicIntensityNet(
        intrinsic_net_type='unet',
        intensity_head_type='conv'
    )
    test_model(model, x, "1. intrinsic=UNet + intensity=Conv (默认)")

    # 2. UNet + UNet
    model = DeepIntrinsicIntensityNet(
        intrinsic_net_type='unet',
        intensity_head_type='unet',
        intensity_hidden_channels=16,
        intensity_depth=3
    )
    test_model(model, x, "2. intrinsic=UNet + intensity=UNet")

    # 3. UNet + DeepLab
    model = DeepIntrinsicIntensityNet(
        intrinsic_net_type='unet',
        intensity_head_type='deeplab',
        deeplab_base_channels=32
    )
    test_model(model, x, "3. intrinsic=UNet + intensity=DeepLab")

    # 4. DeepLab + Conv
    model = DeepIntrinsicIntensityNet(
        intrinsic_net_type='deeplab',
        intensity_head_type='conv',
        deeplab_base_channels=32
    )
    test_model(model, x, "4. intrinsic=DeepLab + intensity=Conv")

    # 5. DeepLab + UNet
    model = DeepIntrinsicIntensityNet(
        intrinsic_net_type='deeplab',
        intensity_head_type='unet',
        deeplab_base_channels=32,
        intensity_hidden_channels=16
    )
    test_model(model, x, "5. intrinsic=DeepLab + intensity=UNet")

    # 6. DeepLab + DeepLab
    model = DeepIntrinsicIntensityNet(
        intrinsic_net_type='deeplab',
        intensity_head_type='deeplab',
        deeplab_base_channels=32
    )
    test_model(model, x, "6. intrinsic=DeepLab + intensity=DeepLab (全 DeepLab)")

    # 7. 使用 S 作为强度输入
    model = DeepIntrinsicIntensityNet(
        intrinsic_net_type='unet',
        intensity_head_type='unet',
        use_shading_for_intensity=True,
        intensity_hidden_channels=16
    )
    test_model(model, x, "7. 使用 R+S 作为强度输入 (in_channels=2)")

    print("\n" + "=" * 60)
    print("✓ 所有网络架构组合测试通过!")
    print("=" * 60)
