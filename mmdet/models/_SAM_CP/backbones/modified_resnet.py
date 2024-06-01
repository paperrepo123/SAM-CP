# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseModel, BaseModule
from torch import nn

from mmdet.registry import MODELS


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                OrderedDict([('-1', nn.AvgPool2d(stride)),
                             ('0',
                              nn.Conv2d(
                                  inplanes,
                                  planes * self.expansion,
                                  1,
                                  stride=1,
                                  bias=False)),
                             ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3]).permute(2, 0,
                                                       1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)

        return x[0]


@MODELS.register_module()
class ModifiedResNet(BaseModule):
    """A modified ResNet contains the following changes:

    - Apply deep stem with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is
      prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """ # noqa

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int = 50,
                 base_channels: int = 64,
                 input_size: int = 224,
                 num_attn_heads: int = 32,
                 output_dim: int = 1024,
                 output_levels=[1,2,3,4],
                 frozen_param=False,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        self.input_size = input_size
        self.block, stage_blocks = self.arch_settings[depth]
        self.output_levels=output_levels
        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3,
            base_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels // 2)
        self.conv2 = nn.Conv2d(
            base_channels // 2,
            base_channels // 2,
            kernel_size=3,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels // 2)
        self.conv3 = nn.Conv2d(
            base_channels // 2,
            base_channels,
            kernel_size=3,
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        # this is a *mutable* variable used during construction
        self._inplanes = base_channels
        self.layer1 = self._make_layer(base_channels, stage_blocks[0])
        self.layer2 = self._make_layer(
            base_channels * 2, stage_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            base_channels * 4, stage_blocks[2], stride=2)
        self.layer4 = self._make_layer(
            base_channels * 8, stage_blocks[3], stride=2)

        embed_dim = base_channels * 32
        self.attnpool = AttentionPool2d(input_size // 32, embed_dim,
                                        num_attn_heads, output_dim)
        self.frozen_param=frozen_param
        self._freeze_stages()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_param >= 0:
            for i in range(1, 4):
                m = getattr(self, f'bn{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
            for i in range(1, 4):
                m = getattr(self, f'conv{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

            for i in range(1, 5):
                m = getattr(self, f'layer{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                             (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        outs = []
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        if 1 in self.output_levels:
            outs.append(x.detach() if self.frozen_param else x)
        x = self.layer2(x)
        if 2 in self.output_levels:
            outs.append(x.detach() if self.frozen_param else x)
        x = self.layer3(x)
        if 3 in self.output_levels:
            outs.append(x.detach() if self.frozen_param else x)
        x = self.layer4(x)
        if 4 in self.output_levels:
            outs.append(x.detach() if self.frozen_param else x)
        # x = self.attnpool(x)
        # outs = []
        # for i, layer_name in enumerate(self.res_layers):
        #     res_layer = getattr(self, layer_name)
        #     x = res_layer(x)
        #     if i in self.out_indices:
        #         outs.append(x)
        return outs
