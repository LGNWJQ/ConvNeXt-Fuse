# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2022 04 12 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Conv_Next_Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + x
        return x


# 基本卷积模块
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect')
        self.relu = nn.ReLU(True)
        self.is_last = is_last

    def forward(self, x):
        out = self.conv2d(x)
        if self.is_last is False:
            out = self.relu(out)
        return out


# 基本卷积增加通道模块
class Increase_Channels_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Increase_Channels_Block, self).__init__()
        padding = kernel_size // 2
        self.increase = nn.Sequential(LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                                      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                                padding_mode='reflect'))

    def forward(self, x):
        out = self.increase(x)
        return out


class Conv_Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Conv_Encoder, self).__init__()
        self.conv1 = ConvLayer(in_channels, 32, 3, 1)
        self.CNext_Block2 = Conv_Next_Block(32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.CNext_Block2(x)
        return x


class CNN_Decoder(nn.Module):
    def __init__(self, output_nc=1, kernel_size=3, stride=1):
        super(CNN_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            ConvLayer(32, 32, kernel_size, stride),
            ConvLayer(32, 16, kernel_size, stride),
            ConvLayer(16, output_nc, kernel_size, stride, is_last=True)
        )

    def forward(self, encoder_output):
        return self.decoder(encoder_output)


class Train_Module(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, kernel_size=3, stride=1):
        super(Train_Module, self).__init__()
        self.encoder = Conv_Encoder(input_nc)
        self.decoder = CNN_Decoder(output_nc=output_nc, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        encoder_feature = self.encoder(x)
        out = self.decoder(encoder_feature)
        return out


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # net = Train_Module(input_nc=3, output_nc=3)
    net = Conv_Next_Block(96)
    # train_net = Train_Module(input_nc=3, output_nc=3)
    print("ConvNext-Fusion have {} paramerters in total".format(sum(x.numel() for x in net.parameters())))
    writer = SummaryWriter('./logs/C_Next/Train_Module2')
    writer.add_graph(net, torch.randn((7, 96, 256, 256)))
    writer.close()
    # RGB: ConvNext-Fusion have 10048 paramerters in total
    # GRAY:
