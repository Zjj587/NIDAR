"""
网络变体模块

提供多种可选的网络架构：
1. IntensityUNet: 轻量 U-Net 用于强度预测
2. DeepLabV3PlusWrapper: DeepLabV3+ 封装，支持单通道输入
3. IntrinsicDeepLab: 基于 DeepLabV3+ 的本征分解网络

可通过配置文件选择使用哪种架构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ===================== 基础模块（复用自 deep_intrinsic_net.py）=====================

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
        self.conv1 = ConvBlock(in_c, out_c, stride=2)
        self.conv2 = ConvBlock(out_c, out_c)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class DecoderBlock(nn.Module):
    """解码器块：上采样 + skip connection + 两层卷积"""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.conv1 = ConvBlock(in_c + skip_c, out_c)
        self.conv2 = ConvBlock(out_c, out_c)

    def forward(self, x, skip, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


# ===================== IntensityUNet =====================

class IntensityUNet(nn.Module):
    """
    轻量 U-Net 用于强度预测

    类似 IntrinsicUNet，但输出单通道强度图。
    可配置深度和通道数，默认比 IntrinsicUNet 更轻量。

    输入: R (B, 1, H, W) 或 R||S (B, 2, H, W)
    输出: I_hat (B, 1, H, W)
    """
    def __init__(self, in_channels=1, base_channels=16, depth=3):
        """
        Args:
            in_channels: 输入通道数（1=仅R，2=R+S）
            base_channels: 基础通道数（默认16，比 IntrinsicUNet 的32更轻量）
            depth: 编码器/解码器层数（2-4）
        """
        super().__init__()

        self.depth = depth

        # 通道数配置
        nf = [base_channels * (2 ** i) for i in range(depth + 1)]

        # 初始卷积
        self.init_conv = ConvBlock(in_channels, nf[0])

        # 编码器
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(EncoderBlock(nf[i], nf[i + 1]))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(nf[depth], nf[depth]),
            ConvBlock(nf[depth], nf[depth])
        )

        # 解码器
        self.decoders = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.decoders.append(DecoderBlock(nf[i], nf[i - 1], nf[i - 1]))

        # 输出头
        self.out_head = nn.Sequential(
            ConvBlock(nf[0], nf[0] // 2),
            nn.Conv2d(nf[0] // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, R, S=None):
        # 处理输入
        if S is not None:
            x = torch.cat([R, S], dim=1)
        else:
            x = R

        input_size = x.shape[2:]

        # 编码
        features = []
        x = self.init_conv(x)
        features.append(x)

        for enc in self.encoders:
            x = enc(x)
            features.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # 解码
        for i, dec in enumerate(self.decoders):
            skip_idx = self.depth - 1 - i
            target_size = features[skip_idx].shape[2:]
            x = dec(x, features[skip_idx], target_size)

        # 确保输出尺寸与输入一致
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return self.out_head(x)


# ===================== ASPP 模块（DeepLabV3+ 核心）=====================

class ASPPConv(nn.Sequential):
    """ASPP 卷积分支"""
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Module):
    """ASPP 全局池化分支"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling

    多尺度空洞卷积，捕获不同感受野的上下文信息。
    """
    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()

        modules = []
        # 1x1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # 空洞卷积分支
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 全局池化分支
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 融合
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# ===================== 轻量 ResNet Backbone（适合单通道输入）=====================

class BasicBlock(nn.Module):
    """ResNet BasicBlock"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """ResNet Bottleneck"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class LightResNetBackbone(nn.Module):
    """
    轻量 ResNet Backbone，支持单通道输入

    用于 DeepLabV3+，提供 low_level 和 high_level 特征。
    比标准 ResNet 更轻量（channels 减半），适合单通道灰度图任务。
    """
    def __init__(self, in_channels=1, base_channels=32, layers=(2, 2, 2, 2),
                 block=BasicBlock, output_stride=16):
        super().__init__()

        self.inplanes = base_channels

        # 根据 output_stride 设置 dilation
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]

        # Stem
        self.conv1 = nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=strides[3], dilation=dilations[3])

        # 记录输出通道数
        self.low_level_channels = base_channels * block.expansion  # layer1 输出
        self.high_level_channels = base_channels * 8 * block.expansion  # layer4 输出

        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        low_level = self.layer1(x)  # 用于 DeepLabV3+ 的 low-level 特征
        x = self.layer2(low_level)
        x = self.layer3(x)
        high_level = self.layer4(x)  # 高级特征

        return {'low_level': low_level, 'out': high_level}


# ===================== DeepLabV3+ Head =====================

class DeepLabV3PlusHead(nn.Module):
    """
    DeepLabV3+ 解码器头

    结合 ASPP 的高级特征和 low-level 特征，输出指定通道数。
    """
    def __init__(self, in_channels, low_level_channels, num_classes,
                 aspp_out_channels=256, low_level_out_channels=48,
                 aspp_dilate=(6, 12, 18)):
        super().__init__()

        # Low-level 特征投影
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_out_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_out_channels),
            nn.ReLU(inplace=True),
        )

        # ASPP
        self.aspp = ASPP(in_channels, aspp_out_channels, aspp_dilate)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Conv2d(aspp_out_channels + low_level_out_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        low_level = self.project(features['low_level'])
        high_level = self.aspp(features['out'])

        # 上采样高级特征到 low-level 尺寸
        high_level = F.interpolate(high_level, size=low_level.shape[2:],
                                   mode='bilinear', align_corners=False)

        # 拼接并分类
        x = torch.cat([low_level, high_level], dim=1)
        return self.classifier(x)


# ===================== DeepLabV3+ 完整模型 =====================

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ 完整模型

    支持单通道输入，可用于：
    - 本征分解（输出 R、S）
    - 强度预测（输出 I_hat）
    """
    def __init__(self, in_channels=1, num_classes=1, base_channels=32,
                 output_stride=16, backbone_layers=(2, 2, 2, 2)):
        super().__init__()

        self.backbone = LightResNetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            layers=backbone_layers,
            output_stride=output_stride
        )

        self.head = DeepLabV3PlusHead(
            in_channels=self.backbone.high_level_channels,
            low_level_channels=self.backbone.low_level_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        out = self.head(features)
        # 上采样到输入尺寸
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out


# ===================== 用于本征分解的 DeepLabV3+ =====================

class IntrinsicDeepLab(nn.Module):
    """
    基于 DeepLabV3+ 的本征分解网络

    输入: pseudo-NIR (B, 1, H, W)
    输出: R (B, 1, H, W), S (B, 1, H, W)

    使用共享的 backbone + 两个独立的 head 分别预测 R 和 S。
    """
    def __init__(self, in_channels=1, base_channels=32, output_stride=16):
        super().__init__()

        # 共享 backbone
        self.backbone = LightResNetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            layers=(2, 2, 2, 2),
            output_stride=output_stride
        )

        # R head
        self.r_head = DeepLabV3PlusHead(
            in_channels=self.backbone.high_level_channels,
            low_level_channels=self.backbone.low_level_channels,
            num_classes=1,
        )

        # S head
        self.s_head = DeepLabV3PlusHead(
            in_channels=self.backbone.high_level_channels,
            low_level_channels=self.backbone.low_level_channels,
            num_classes=1,
        )

    def forward(self, x):
        input_shape = x.shape[-2:]

        # 共享特征提取
        features = self.backbone(x)

        # R 预测
        R = self.r_head(features)
        R = F.interpolate(R, size=input_shape, mode='bilinear', align_corners=False)
        R = torch.sigmoid(R)  # R ∈ [0, 1]

        # S 预测
        S = self.s_head(features)
        S = F.interpolate(S, size=input_shape, mode='bilinear', align_corners=False)
        S = F.softplus(S)  # S > 0

        return R, S


# ===================== 用于强度预测的 DeepLabV3+ =====================

class IntensityDeepLab(nn.Module):
    """
    基于 DeepLabV3+ 的强度预测网络

    输入: R (B, 1, H, W) 或 R||S (B, 2, H, W)
    输出: I_hat (B, 1, H, W)
    """
    def __init__(self, in_channels=1, base_channels=32, output_stride=16):
        super().__init__()

        self.net = DeepLabV3Plus(
            in_channels=in_channels,
            num_classes=1,
            base_channels=base_channels,
            output_stride=output_stride
        )

    def forward(self, R, S=None):
        if S is not None:
            x = torch.cat([R, S], dim=1)
        else:
            x = R

        out = self.net(x)
        return torch.sigmoid(out)  # 输出 [0, 1]


# ===================== 测试代码 =====================

if __name__ == '__main__':
    print("测试网络变体模块...")

    B, H, W = 2, 384, 1248

    # 测试 IntensityUNet
    print("\n1. 测试 IntensityUNet:")
    model = IntensityUNet(in_channels=1, base_channels=16, depth=3)
    R = torch.randn(B, 1, H, W)
    out = model(R)
    print(f"   输入 R: {R.shape} -> 输出 I_hat: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {params / 1e6:.2f}M")

    # 测试 IntrinsicDeepLab
    print("\n2. 测试 IntrinsicDeepLab:")
    model = IntrinsicDeepLab(in_channels=1, base_channels=32)
    x = torch.randn(B, 1, H, W)
    R, S = model(x)
    print(f"   输入: {x.shape} -> R: {R.shape}, S: {S.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {params / 1e6:.2f}M")

    # 测试 IntensityDeepLab
    print("\n3. 测试 IntensityDeepLab:")
    model = IntensityDeepLab(in_channels=1, base_channels=32)
    R = torch.randn(B, 1, H, W)
    out = model(R)
    print(f"   输入 R: {R.shape} -> 输出 I_hat: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {params / 1e6:.2f}M")

    # 测试带 S 输入的 IntensityDeepLab
    print("\n4. 测试 IntensityDeepLab (R+S 输入):")
    model = IntensityDeepLab(in_channels=2, base_channels=32)
    R = torch.randn(B, 1, H, W)
    S = torch.randn(B, 1, H, W)
    out = model(R, S)
    print(f"   输入 R+S: {R.shape}+{S.shape} -> 输出 I_hat: {out.shape}")

    print("\n✓ 所有测试通过!")
