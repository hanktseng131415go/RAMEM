#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
from torch import nn
import torch.nn.functional as F

from typing import Sequence, Tuple
import math

# reference: https://github.com/Megvii-BaseDetection/YOLOX

class SigmoidGeometricMean(torch.autograd.Function):
    """Forward and backward function of geometric mean of two sigmoid
    functions.
    This implementation with analytical gradient function substitutes
    the autograd function of (x.sigmoid() * y.sigmoid()).sqrt(). The
    original implementation incurs none during gradient backprapagation
    if both x and y are very small values.
    """

    @staticmethod
    def forward(ctx, x, y):
        x_sigmoid = x.sigmoid()
        y_sigmoid = y.sigmoid()
        z = (x_sigmoid * y_sigmoid).sqrt()
        ctx.save_for_backward(x_sigmoid, y_sigmoid, z)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x_sigmoid, y_sigmoid, z = ctx.saved_tensors
        grad_x = grad_output * z * (1 - x_sigmoid) / 2
        grad_y = grad_output * z * (1 - y_sigmoid) / 2
        return grad_x, grad_y
    
sigmoid_geometric_mean = SigmoidGeometricMean.apply

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.conv(x)

class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        # self.dconv = BaseConv(
        #     in_channels, in_channels, ksize=ksize,
        #     stride=stride, groups=in_channels, act=act
        # )
        # self.pconv = BaseConv(
        #     in_channels, out_channels, ksize=1,
        #     stride=1, groups=1, act=act
        # )

        self.dconv = BaseConv(
            in_channels, in_channels, ksize=ksize,
            stride=stride, groups=in_channels, act=act
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1,
            stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv.fuseforward(x)
        return self.pconv(x)
    
class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        # self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.conv1 = DWConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        # self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)
        self.conv2 = DWConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        # self.layer1 = BaseConv(in_channels, mid_channels, ksize=1, stride=1, act="lrelu")
        # self.layer2 = BaseConv(mid_channels, in_channels, ksize=3, stride=1, act="lrelu")

        self.layer1 = BaseConv(in_channels, mid_channels, ksize=1, stride=1, act="silu")
        self.layer2_d = BaseConv(mid_channels, mid_channels, ksize=5, stride=1, groups=mid_channels, act="silu")
        self.layer2_p = BaseConv(mid_channels, in_channels, ksize=1, stride=1, act="silu")


    def forward(self, x):
        
        # out = self.layer2(self.layer1(x))
        
        out = self.layer1(x)
        out = self.layer2_p(self.layer2_d(out))
        
        return x + out


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=2, act="silu"):
        super().__init__()
        # self.conv1 = DWConv(3, out_channels // 2, 3, 2, act=act)
        # self.conv2 = DWConv(out_channels // 2, out_channels // 2, 3, 1, act=act)
        # self.conv3 = DWConv(out_channels // 2, out_channels, 3, 1, act=act)

        self.conv1 = BaseConv(3, out_channels // 2, 3, 2, act=act)
        self.conv2 = BaseConv(out_channels // 2, out_channels // 2, 3, 1, act=act)
        self.conv3 = BaseConv(out_channels // 2, out_channels, 3, 1, act=act)

#        self.up = nn.Upsample(scale_factor=2)
        
    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
#        x = self.up(x)
        # patch_top_left = x[..., ::2, ::2]
        # patch_top_right = x[..., ::2, 1::2]
        # patch_bot_left = x[..., 1::2, ::2]
        # patch_bot_right = x[..., 1::2, 1::2]
        # x = torch.cat(
        #     (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,
        # )
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
            self, in_channels, out_channels, shortcut=True,
            expansion=0.5, depthwise=False, act="silu"
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        
        self.conv1 = BaseConv(in_channels, hidden_channels, 3, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 5, stride=1, act=act)

        # self.conv1 = Conv(in_channels, hidden_channels, 3, stride=1, act=act)
        # self.conv2 = BaseConv(hidden_channels, out_channels, 5, stride=1, act=act)
        
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class ChannelAttention(nn.Module):
    """Channel attention Module.
    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, channels, init_cfg = None):
        super().__init__()
        
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self, in_channels, out_channels, n=1,
            shortcut=True, expansion=0.5, depthwise=False, act="silu"
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)
        self.attn = ChannelAttention(2 * hidden_channels)

    def forward(self, x):
        
        x_2 = self.conv2(x)
        
        x_1 = self.conv1(x)     
        x_1 = self.m(x_1)
        
        x = torch.cat((x_1, x_2), dim=1)
        
        x = self.attn(x)
        x = self.conv3(x)
        
        return x


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(self, depth, in_channels=3, stem_out_channels=32, out_indices=(3, 4, 5)):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        out_features = out_indices
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[0], stride=2))
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[1], stride=2))
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[2], stride=2))
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)]
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu"
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = []
        for idx, layer in enumerate([self.stem, self.dark2, self.dark3, self.dark4, self.dark5]):
            x = layer(x)

            if idx + 1 in self.out_features:
                outputs.append(x)
        return outputs


class CSPDarknet(nn.Module):

    def __init__(self, dep_mul=1.33, wid_mul=1.25, out_indices=(2, 3, 4, 5), depthwise=True, act="silu"):
        super().__init__()
        # out_indices=(2, 3, 4)
        self.depthwise = depthwise
        self.out_features = out_indices
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2, base_channels * 2,
                n=base_depth * 1, depthwise=self.depthwise, act=act
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4, base_channels * 4,
                n=base_depth * 2, depthwise=self.depthwise, act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8, base_channels * 8,
                n=base_depth * 2, depthwise=self.depthwise, act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16, base_channels * 16, n=base_depth * 1,
                shortcut=False, depthwise=self.depthwise, act=act,
            ),
        )
            
#        self.linear = nn.Linear(1792, 10)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x):
        outputs = []
#        print(x.shape)
        for idx, layer in enumerate([self.stem, self.dark2, self.dark3, self.dark4, self.dark5]):
            x = layer(x)
            # print(idx+1)
            # print(x.shape)

            if idx + 1 in self.out_features:
                # out = F.avg_pool2d(x, x.size()[2:])
                # out = out.view(out.size(0), -1)
                out = x
                # print(x.shape)
                outputs.append(out)
                
#        out = torch.cat(outputs, 1)
#        out = self.linear(out)
        
        return outputs

class PAFPN(nn.Module):
    """Path Aggregation Network with CSPNeXt blocks.
    Args:
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer.
            Defaults to 3.
        use_depthwise (bool): Whether to use depthwise separable convolution in
            blocks. Defaults to False.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """
    # in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4
    def __init__(self,
                   # in_channels=[96, 192, 384],
                  in_channels=[320, 640, 1280],
                  # out_channels=96,
                  out_channels=256,
                 num_csp_blocks=4,
                 use_depthwise=True,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='silu'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        Conv = DWConv if use_depthwise else BaseConv
        # Conv = BaseConv
        
        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                Conv(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    n=num_csp_blocks,
                    depthwise=use_depthwise,
                    shortcut=False))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                Conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    n=num_csp_blocks,
                    depthwise=use_depthwise,
                    shortcut=False))        
            
        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                Conv(
                    in_channels[i],
                    out_channels,
                    3))

        # for i in range(2):
        self.downsamples.append(
            nn.Sequential(Conv(in_channels[-1],
                               out_channels * 1,
                               3,
                               stride=2),
                          CSPLayer(out_channels * 1,
                                   out_channels * 1,
                                   n=num_csp_blocks,
                                   depthwise=use_depthwise)))
        self.downsamples.append(
            nn.Sequential(Conv(out_channels * 1,
                               out_channels * 1,
                               3,
                               stride=2),
                          CSPLayer(out_channels * 1,
                                   out_channels * 1,
                                   n=num_csp_blocks,
                                   depthwise=use_depthwise)))
        
        self.out_convs.append(
            Conv(
                out_channels,
                out_channels,
                3))
        self.out_convs.append(
            Conv(
                out_channels,
                out_channels,
                3))
            
    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        # assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            # print('out: ', out.shape)
            outs.append(out)
        
        outs.append(self.downsamples[-2](outs[-1]))
        outs.append(self.downsamples[-1](outs[-1]))
        
        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return outs

class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class Coord(nn.Module):

    def __init__(self, in_channels, out_channels, coord=True, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        
        self.coord = coord
        
        in_size = in_channels+2
        
        if self.coord == False:
            in_size = in_channels
            
        if with_r:
            in_size += 1
        # self.conv = nn.Conv2d(in_size, out_channels, kernel_size=9, stride=1, padding=4, dilation=1)
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size=1, dilation=1)
        
        # self.conv = upa_block(in_size, out_channels, stride=1, l=1, k=1, same=False, bn=False, cpa=True, act=False)

    def forward(self, x):
        
        if self.coord == True:
            ret = self.addcoords(x)
        else:
            ret = x
        
#        print('ret: ', ret.shape)
        ret = self.conv(ret)
        # ret = self.mlp(ret.permute(0,2,3,1)).permute(0,3,1,2)
        
        return ret
    
class MaskFeatModule(nn.Module):
    """Mask feature head used in RTMDet-Ins.
    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels of the mask feature
             map branch.
        num_levels (int): The starting feature map level from RPN that
             will be used to predict the mask feature map.
        num_prototypes (int): Number of output channel of the mask feature
             map branch. This is the channel count of the mask
             feature map that to be dynamically convolved with the predicted
             kernel.
        stacked_convs (int): Number of convs in mask feature branch.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True)
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    def __init__(
        self,
        in_channels,
        feat_channels = 256,
        stacked_convs = 4,
        num_levels = 5,
        num_prototypes = 8,
        act_cfg = dict(type='ReLU', inplace=True),
        norm_cfg = dict(type='BN')
    ):
        super().__init__()
        self.num_levels = num_levels
        self.fusion_conv = nn.Conv2d(num_levels * in_channels, in_channels, 1)
        Conv = BaseConv
        convs = []
        for i in range(stacked_convs):
            in_c = in_channels if i == 0 else feat_channels
            convs.append(
                Conv(
                    in_c,
                    feat_channels,
                    3,
                    ))
        self.stacked_convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(
            feat_channels, num_prototypes, kernel_size=1)

    def forward(self, features):
        # multi-level feature fusion
        fusion_feats = [features[0]]
        size = features[0].shape[-2:]
        for i in range(1, self.num_levels):
            # print('features: ', torch.tensor(features).shape)
            # print('i: ', i)
            f = F.interpolate(features[i], size=size, mode='bilinear')
            # print('f: ', f.shape)
            fusion_feats.append(f)
        fusion_feats = torch.cat(fusion_feats, dim=1)
        fusion_feats = self.fusion_conv(fusion_feats)
        # pred mask feats
        mask_features = self.stacked_convs(fusion_feats)
        mask_features = self.projection(mask_features)
        # print('mask_features: ', mask_features.shape)
        return mask_features    

class RTMDetInsSepBNHead(nn.Module):
    """Detection Head of RTMDet-Ins with sep-bn layers.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
        norm_cfg (:obj:`ConfigDict` or dict)): Config dict for normalization
            layer. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict)): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        pred_kernel_size (int): Kernel size of prediction layer. Defaults to 1.
    """

    def __init__(self,
                 num_classes = 20,
                 in_channels = 320,
                 share_conv = True,
                 with_objectness = False, 
                 pred_kernel_size = 1,
                 use_depthwise=True,
                 ) :
       
        super().__init__()
        
        self.share_conv = share_conv
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_kernel = nn.ModuleList()
        self.rtm_obj = nn.ModuleList()
        
        self.num_dyconvs = 3
        self.num_prototypes = 20
        self.dyconv_channels = 8
        self.pred_kernel_size = 1
        self.strides = [8, 16, 32, 64, 128]
        self.stacked_convs = 4
        self.in_channels = in_channels
        self.feat_channels = 256
        self.num_base_priors = 1
        self.cls_out_channels = 21
        self.with_objectness = with_objectness
        
        # calculate num dynamic parameters
        # weight_nums, bias_nums = [], []
        # for i in range(self.num_dyconvs):
        #     if i == 0:
        #         weight_nums.append(
        #             (self.num_prototypes + 2) * self.dyconv_channels)
        #         bias_nums.append(self.dyconv_channels)
        #     elif i == self.num_dyconvs - 1:
        #         weight_nums.append(self.dyconv_channels)
        #         bias_nums.append(1)
        #     else:
        #         weight_nums.append(self.dyconv_channels * self.dyconv_channels)
        #         bias_nums.append(self.dyconv_channels)
        # self.weight_nums = weight_nums
        # self.bias_nums = bias_nums
        # self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.num_gen_params = self.num_prototypes
        pred_pad_size = self.pred_kernel_size // 2
        
        Conv = DWConv if use_depthwise else BaseConv
        
        for n in range(len(self.strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            kernel_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    Conv(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        ))
                reg_convs.append(
                    Conv(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        ))
                kernel_convs.append(
                    Conv(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        ))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(cls_convs)
            self.kernel_convs.append(kernel_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    self.pred_kernel_size,
                    padding=pred_pad_size))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=pred_pad_size))
            self.rtm_kernel.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_gen_params,
                    self.pred_kernel_size,
                    padding=pred_pad_size))
            if self.with_objectness:
                self.rtm_obj.append(
                    nn.Conv2d(
                        self.feat_channels,
                        1,
                        self.pred_kernel_size,
                        padding=pred_pad_size))

        if self.share_conv:
            for n in range(len(self.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i] = self.cls_convs[0][i]
                    self.reg_convs[n][i] = self.reg_convs[0][i]
        
        # self.num_prototypes = 13
        self.mask_head = MaskFeatModule(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            stacked_convs=4,
            num_levels=len(self.strides),
            num_prototypes=self.num_prototypes)
        
        # self.addcoords = AddCoords()
        
        self.coord = Coord(self.num_prototypes, self.num_prototypes)

    def parse_dynamic_params(self, flatten_kernels):
        """split kernel head prediction to conv weight and bias."""
        n_inst = flatten_kernels.size(0)
        # print('weight_nums: ', self.weight_nums.shape)
        # print('bias_nums: ', self.bias_nums.shape)
        n_layers = len(self.weight_nums)
        # print('n_inst: ', n_inst)
        # print('n_layers: ', n_layers)
        params_splits = list(
            torch.split_with_sizes(
                flatten_kernels, self.weight_nums + self.bias_nums, dim=1))
        weight_splits = params_splits[:n_layers]
        bias_splits = params_splits[n_layers:]
        for i in range(n_layers):
            if i < n_layers - 1:
                weight_splits[i] = weight_splits[i].reshape(
                    n_inst * self.dyconv_channels, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_inst *
                                                        self.dyconv_channels)
            else:
                weight_splits[i] = weight_splits[i].reshape(n_inst, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_inst)

        return weight_splits, bias_splits
    
    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - kernel_preds (list[Tensor]): Dynamic conv kernels for all scale
              levels, each is a 4D-tensor, the channels number is
              num_gen_params.
            - mask_feat (Tensor): Output feature of the mask head. Each is a
              4D-tensor, the channels number is num_prototypes.
        """
        mask_feat = self.mask_head(feats)

        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        for idx, (x, stride) in enumerate(
                zip(feats, self.strides)):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for kernel_layer in self.kernel_convs[idx]:
                kernel_feat = kernel_layer(kernel_feat)
            kernel_pred = torch.tanh(self.rtm_kernel[idx](kernel_feat))

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))
            
            # print('F.relu(self.rtm_reg[idx](reg_feat)) : ', F.relu(self.rtm_reg[idx](reg_feat)).shape)
            reg_dist = self.rtm_reg[idx](reg_feat)#) * stride#[0]
            # print('cls_score: ', cls_score.shape)
            # print('reg_dist: ', reg_dist.shape)
            # print('kernel_pred: ', kernel_pred.shape)
            
            cls_scores.append(cls_score.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.cls_out_channels))
            bbox_preds.append(reg_dist.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4))
            kernel_preds.append(kernel_pred.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_gen_params))
            # print('mask_feat: ', mask_feat.shape)
            
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        kernel_preds = torch.cat(kernel_preds, dim=1)

        # print('cls_score: ', cls_scores.shape)
        # print('bbox_preds: ', bbox_preds.shape)
        # print('kernel_preds: ', kernel_preds.shape)
        # h, w = mask_feat.shape[-2:]
        # print('pre mask_feat: ', mask_feat.shape)
        # mask_feat = self.addcoords(mask_feat)#.permute(0, 2, 3, 1)
        mask_feat = self.coord(mask_feat).permute(0, 2, 3, 1)
        # mask_feat = mask_feat.unsqueeze(1).repeat(1, 6175, 1, 1, 1)
        # print('post mask_feat: ', mask_feat.shape)
        # batch_pos_mask_logits = []
        # for i in range(mask_feat.shape[0]):
        #     weights, biases = self.parse_dynamic_params(kernel_preds[i])
        #     n_layers = len(weights)
        #     x = mask_feat[i].reshape(1, -1, h, w)
        #     for i, (weight, bias) in enumerate(zip(weights, biases)):
        #         # print('x: ', x.shape)
        #         # print('weight: ', weight.shape)
        #         # print('bias: ', bias.shape)
        #         x = F.conv2d(
        #             x, weight, bias=bias, stride=1, padding=0, groups=6175)
        #         if i < n_layers - 1:
        #             x = F.relu(x)
        #         # print('x: ', x.shape)
        #     x = x.permute(0, 2, 3, 1)
        #     # print('reshape x: ', x.shape)
        #     batch_pos_mask_logits.append(x)
        #     # print('============')
        # mask_feat = torch.cat(batch_pos_mask_logits, dim=0)
        # # print('mask_feat: ', mask_feat.shape)
        
        return cls_scores, bbox_preds, kernel_preds, mask_feat
#%%
# import time   
# net = net.cuda()
# x = torch.randn(1, 3, 416, 416).cuda()
# #%%
# start = time.time()
# y = net(x)
# end = time.time() -start
# fps = 1/end
## %%
#if __name__ == "__main__":
#    from thop import profile
#
#    # self.depth = 0.33
#    # self.width = 0.50
#    depth = 0.33
#    width = 0.375
#    m = CSPDarknet(dep_mul=depth, wid_mul=width, out_indices=(3, 4, 5))
#    m.init_weights()
#    m.eval()
#
#    inputs = torch.rand(1, 3, 640, 640)
#    # total_ops, total_params = profile(m, (inputs,))
#    # print("total_ops {}G, total_params {}M".format(total_ops/1e9, total_params/1e6))
#    level_outputs = m(inputs)
#    for level_out in level_outputs:
#        print(tuple(level_out.shape))
