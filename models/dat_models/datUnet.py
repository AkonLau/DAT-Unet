# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------
# --------------------------------------------------------
# DAT-UNet
# Modified by Kangjun Liu
# 2024-10-23
# --------------------------------------------------------

import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import DropPath, to_2tuple

from .dat_blocks import *
from .nat import NeighborhoodAttention2D


class LayerScale(nn.Module):

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)

class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio,
                 heads, heads_q, stride,
                 offset_range_factor,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, 
                 use_dwc_mlp, ksize, nat_ksize,
                 k_qna, nq_qna, qna_activation, 
                 layer_scale_value, use_lpu, log_cpb):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed) for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] != 'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP

        self.mlps = nn.ModuleList(
            [ 
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity() 
                for _ in range(2 * depths)
            ]
        )
        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )

        for i in range(depths):

            if stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, ksize, log_cpb)
                )

            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(dim_embed, nat_ksize, heads, attn_drop, proj_drop)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):

        x = self.proj(x)

        for d in range(self.depths):
            
            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x)
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x


class TransformerStage_up(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups,
                 use_pe, sr_ratio,
                 heads, heads_q, stride,
                 offset_range_factor,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate,
                 use_dwc_mlp, ksize, nat_ksize,
                 k_qna, nq_qna, qna_activation,
                 layer_scale_value, use_lpu, log_cpb):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed) for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] != 'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP

        self.mlps = nn.ModuleList(
            [
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity()
                for _ in range(2 * depths)
            ]
        )
        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1,
                          groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )

        for i in range(depths):

            if stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads,
                                       hc, n_groups, attn_drop, proj_drop,
                                       stride, offset_range_factor, use_pe, dwc_pe,
                                       no_off, fixed_pe, ksize, log_cpb)
                )
            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(dim_embed, nat_ksize, heads, attn_drop, proj_drop)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):

        x = self.proj(x)

        for d in range(self.depths):

            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x)
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale*dim_scale*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class DAT_UNET(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, out_chans=1, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 heads=[3, 6, 12, 24], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 strides=[-1,-1,-1,-1],
                 offset_range_factor=[1, 2, 3, 4],
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 nat_ksizes=[3,3,3,3],
                 layer_scale_values=[-1,-1,-1,-1],
                 use_lpus=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 final_upsample="expand_first",
                 **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.out_channels = out_chans
        self.final_upsample = final_upsample
        self.patch_size = patch_size
        self.img_size = img_size

        self.patch_proj = nn.Sequential(
            nn.Conv2d(in_chans, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(in_chans, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        )

        mid_img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build encoder and bottleneck layers
        self.stages = nn.ModuleList()
        for i in range(self.num_layers):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(
                    mid_img_size, window_sizes[i], ns_per_pts[i],
                    dim1, dim2, depths[i],
                    stage_spec[i], groups[i], use_pes[i],
                    sr_ratios[i], heads[i], heads_q[i], strides[i],
                    offset_range_factor[i],
                    dwc_pes[i], no_offs[i], fixed_pes[i],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths[:i]):sum(depths[:i + 1])], use_dwc_mlps[i],
                    ksizes[i], nat_ksizes[i], ksize_qnas[i], nqs[i],qna_activation,
                    layer_scale_values[i], use_lpus[i], log_cpb[i]
                )
            )
            mid_img_size = mid_img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        self.norm_down = LayerNormProxy(dims[-1]) 

        # build decoder layers
        self.stages_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

########################
        for i_layer in range(self.num_layers):
            mid_img_size = mid_img_size * 2
            i = self.num_layers - 1 - i_layer # reverse order 3,2,1,0

            concat_linear = nn.Conv2d(2*int(dim_stem*2**i),
            int(dim_stem*2**i), 1, 1, 0) if i_layer > 0 else nn.Identity()

            self.concat_back_dim.append(concat_linear)

            dim1 = dims[i]
            dim2 = dim_stem if i == 0 else dims[i - 1] * 2
#             if i_layer ==0 :
#                 self.stages_up.append(nn.Identity())
#             else:
            self.stages_up.append(
                    TransformerStage_up(
                        mid_img_size, window_sizes[i], ns_per_pts[i],
                        dim1, dim2, depths[i],
                        stage_spec[i], groups[i], use_pes[i],
                        sr_ratios[i], heads[i], heads_q[i], strides[i],
                        offset_range_factor[i],
                        dwc_pes[i], no_offs[i], fixed_pes[i],
                        attn_drop_rate, drop_rate, expansion, drop_rate,
                        dpr[sum(depths[:i]):sum(depths[:i + 1])], use_dwc_mlps[i],
                        ksizes[i], nat_ksizes[i], ksize_qnas[i], nqs[i],qna_activation,
                        layer_scale_values[i], use_lpus[i], log_cpb[i]
                    )
                )

        self.up_projs = nn.ModuleList()
        for i_layer in range(self.num_layers-1):
            i = self.num_layers - 1 - i_layer # reverse order 3,2,1
            self.up_projs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dims[i], dims[i - 1], 6, 2, 2, bias=False),
                    LayerNormProxy(dims[i - 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.ConvTranspose2d(dims[i], dims[i - 1], 4, 2, 1, bias=False),
                    LayerNormProxy(dims[i - 1])
                )
            )
########################

        self.norm_up = LayerNormProxy(dim_stem)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=dim_stem)
            self.output = nn.Conv2d(in_channels=dim_stem, out_channels=self.out_channels, kernel_size=1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def load_pretrained(self, state_dict, lookup_22k):

        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l_side = int(math.sqrt(n))
                    assert n == l_side ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l_side, l_side, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
                if 'cls_head' in keys:
                    new_state_dict[state_key] = state_value[lookup_22k]

        msg = self.load_state_dict(new_state_dict, strict=False)
        return msg

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_proj(x)

        x_downsample = []
        for i in range(self.num_layers):
            x_downsample.append(x)
            x = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)

        x = self.norm_down(x)

        return x, x_downsample

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for i in range(self.num_layers):
            if i > 0:
                x = torch.cat([x,x_downsample[self.num_layers-1-i]],1)
                x = self.concat_back_dim[i](x)
                x = self.stages_up[i](x)
            else:
                x = self.stages_up[i](x)
                
            if i < 3:
                x = self.up_projs[i](x)
        x = self.norm_up(x)
        return x

    def up_x4(self, x):
        H, W = self.img_size // self.patch_size, self.img_size // self.patch_size
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = self.up(x)
            x = x.view(B,self.patch_size*H,self.patch_size*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.output(x)

        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x,x_downsample)
        x = self.up_x4(x)
        x = torch.sigmoid(x)

        return x


if __name__ == '__main__':

    model = DAT_UNET(
        img_size=256,
        patch_size=4,
        in_chans=3,
        out_chans=1,
        expansion=4,
        dim_stem=64,
        dims=[64, 128, 256, 512],
        depths=[4, 4, 4, 4],
        stage_spec=[
            ['D', 'D', 'D', 'D'],
            ['D', 'D', 'D', 'D'],
            ['D', 'D', 'D', 'D'],
            ['D', 'D', 'D', 'D'],
        ],
        heads=[2, 4, 8, 16],
        window_sizes=[7, 7, 7, 7],
        groups=[1, 2, 4, 8],
        use_pes=[True, True, True, True],
        dwc_pes=[False, False, False, False],
        strides=[8, 4, 2, 1],
        offset_range_factor=[-1, -1, -1, -1],
        no_offs=[False, False, False, False],
        fixed_pes=[False, False, False, False],
        use_dwc_mlps=[True, True, True, True],
        use_lpus=[True, True, True, True],
        use_conv_patches=True,
        ksizes=[9, 7, 5, 3],
        nat_ksizes=[7, 7, 7, 7],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2
    )

    print('**** Setup ****')
    print('Total params: %.2fK' % (sum(p.numel() for p in model.parameters()) * 10 ** -3))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) * 10 ** -6))
    print('************')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input = torch.randn(1, 3, 256, 256)
    input = input.to(device)
    model = model.to(device)

    out = model(input)[0]
    print(out.shape)

    # # 使用 ptflops 统计模型的 FLOPs 和参数数量
    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
    #
    # print(f"FLOPs: {flops}")
    # print(f"Params: {params}")
