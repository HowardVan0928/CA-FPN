# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from einops import rearrange
from mmdet.models.layers import AdaptiveCrossingConv
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType

class ChannelBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc0 = nn.Sequential(
            ConvModule(in_channels=1, out_channels=1, kernel_size=(1,7), stride=(1,1),padding=(0,3),
                       bias=False,act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=1, out_channels=1, kernel_size=(2,1), stride=(1,1),padding=(0,0),
                       bias=False,act_cfg=None),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        attn = torch.cat([avg_out,max_out],dim=3)
        attn = rearrange(attn,'b c h w -> b h w c')
        attn = self.fc0(attn)
        attn = rearrange(attn,'b h w c -> b c h w')
        attn = self.sigmoid(attn)
        return x * attn
    
class DeepCrossingAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_0=(7,3), kernel_size_1=(9,4)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial = AdaptiveCrossingConv(in_channels, kernel_size_0=kernel_size_0, kernel_size_1=kernel_size_1, stride=1)
        self.channel = ChannelBlock()
        self.conv = ConvModule(in_channels, out_channels, kernel_size=3, stride=1, padding=1,act_cfg=None,norm_cfg=None)
        
    def forward(self, x):
        attn0 = self.spatial(x)
        attn1 = self.channel(x)
        attn = attn0 + attn1
        out = self.conv(attn)
        return out

class FeatureAlignModule(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.conv0 = ConvModule(2*in_channels, 2*in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2,norm_cfg=dict(type="BN"),act_cfg=dict(type="ReLU"))
        self.conv1 = ConvModule(2*in_channels, 2*in_channels, kernel_size=1, stride=1, padding=0,norm_cfg=None,act_cfg=dict(type="Sigmoid"))
        self.conv2 = ConvModule(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2,norm_cfg=None,act_cfg=None)
        

    def forward(self,x0,x1):
        x = torch.cat([x0,x1],dim=1)
        x = self.conv0(x)
        x = self.conv1(x)
        attn0, attn1 = torch.split(x, [x0.shape[1], x1.shape[1]], dim=1)
        attn = attn0 * x0 + attn1 * x1
        attn = self.conv2(attn)
        return attn

  
@MODELS.register_module()
class CrossingAttentionFPN(BaseModule):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.align_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = DeepCrossingAttention(in_channels[i],out_channels)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add align convs (bottom-up)
        used_backbone_levels = len(self.lateral_convs)
        
        for i in range(used_backbone_levels - 1, 0, -1):
            align_conv = FeatureAlignModule(out_channels,kernel_size=5)
            self.align_convs.append(align_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = self.align_convs[i-1](laterals[i - 1], F.interpolate(
                    laterals[i], **self.upsample_cfg))
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = self.align_convs[i-1](laterals[i - 1], F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg))

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

@MODELS.register_module()
class CrossingAttentionPAFPN(CrossingAttentionFPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CrossingAttentionPAFPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.convert_align_convs = nn.ModuleList()

        used_backbone_levels = len(self.lateral_convs)
        for i in range(0, used_backbone_levels - 1):
            align_conv = FeatureAlignModule(out_channels,kernel_size=5)
            self.convert_align_convs.append(align_conv)
        
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = self.align_convs[i-1](laterals[i - 1], F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'))

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] = self.convert_align_convs[i](inter_outs[i + 1] ,
                                self.downsample_convs[i](inter_outs[i]))

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)



if __name__=="__main__":
    x0 = torch.randn(1,256,400,400).cuda()
    x1 = torch.randn(1,512,200,200).cuda()
    x2 = torch.randn(1,1024,100,100).cuda()
    x3 = torch.randn(1,2048,50,50).cuda()
    model = CrossingAttentionPAFPN(in_channels=[256,512,1024,2048],out_channels=256,num_outs=6).cuda()
    y = model((x0,x1,x2,x3))
    print(len(y))
    print(y[-1].shape)
    print(y[-2].shape)
    print(y[-3].shape)
    print(y[-4].shape)
    print(y[-5].shape)
    print(y[-6].shape)