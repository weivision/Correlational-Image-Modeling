# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial, reduce
from operator import mul

import copy
from math import cos, pi

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from .nn_util import PatchEmbed, Attention, Block, CrossBlock
from .nn_util import CorrelationBlockV0, CorrelationBlockV1, concat_all_gather
from util.pos_embed import get_2d_sincos_pos_embed


class CorrelationalAutoencoderViT(nn.Module):
    """ Correlational Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, context_size=176, template_size=64,
                 base_momentum=0.99, end_momentum=1.0, accum_iter=1, 
                 T=0.2, sigma_cont=1.0, sigma_corr=1.0, stop_grad_conv=True):
        super().__init__()

        # --------------------------------------------------------------------------
        # cim encoder specifics
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.stop_grad_conv = stop_grad_conv
        self.embed_dim = embed_dim
        
        self.img_size = img_size
        self.context_size = context_size
        self.template_size = template_size
        self.patch_size = patch_size
        
        self.num_patches = (img_size // patch_size)**2

        self.num_context_patches = (context_size // patch_size)**2
        self.num_temp_patches = (template_size // patch_size)**2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # cim decoder specifics
        self.sigma_cont = sigma_cont
        self.sigma_corr = sigma_corr

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_g_embed = self.build_mlp(3, embed_dim, 4096, 256)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_context_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_corr_block = CorrelationBlockV1(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred_mask = nn.Sequential(
            nn.Conv2d(decoder_embed_dim, 2, kernel_size=1, bias=True),
            nn.Upsample((self.context_size, self.context_size), mode='bilinear', align_corners=True)
        ) # decoder to mask

        self.decoder_predictor = self.build_mlp(2, 256, 4096, 256)

        self.mask_xent_loss = nn.CrossEntropyLoss()
        self.mask_mse_loss = nn.MSELoss(reduction="none")

        # --------------------------------------------------------------------------
        self.initialize_weights()

        # momentum
        # --------------------------------------------------------------------------
        self.patch_embed_c = copy.deepcopy(self.patch_embed)
        self.blocks_c = copy.deepcopy(self.blocks)
        self.norm_c = copy.deepcopy(self.norm)
        self.decoder_g_embed_c = copy.deepcopy(self.decoder_g_embed)

        for param_o, param_c in zip(self.patch_embed.parameters(), self.patch_embed_c.parameters()):
            param_c.data.copy_(param_o.data)
            param_c.requires_grad = False

        for param_o, param_c in zip(self.blocks.parameters(), self.blocks_c.parameters()):
            param_c.data.copy_(param_o.data)
            param_c.requires_grad = False

        for param_o, param_c in zip(self.norm.parameters(), self.norm_c.parameters()):
            param_c.data.copy_(param_o.data)
            param_c.requires_grad = False

        for param_o, param_c in zip(self.decoder_g_embed.parameters(), self.decoder_g_embed_c.parameters()):
            param_c.data.copy_(param_o.data)
            param_c.requires_grad = False
        
        self.T = T
        self.base_momentum = base_momentum
        self.end_momentum = end_momentum
        self.momentum = base_momentum
        self.accum_iter = accum_iter
        self.epoch = 0
        self.iteration = 0
        self.max_iter = 0
        self.iter_per_epoch = 0

    def build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_ln=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                # mlp.append(nn.LayerNorm(dim2))
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_ln:
                # mlp.append(nn.LayerNorm(dim2))
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
    
    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the target network."""
        for param_o, param_c in zip(self.patch_embed.parameters(), self.patch_embed_c.parameters()):
            param_c.data = param_c.data * self.momentum + param_o.data * (1.0 - self.momentum)

        for param_o, param_c in zip(self.blocks.parameters(), self.blocks_c.parameters()):
            param_c.data = param_c.data * self.momentum + param_o.data * (1.0 - self.momentum)

        for param_o, param_c in zip(self.norm.parameters(), self.norm_c.parameters()):
            param_c.data = param_c.data * self.momentum + param_o.data * (1.0 - self.momentum)

        for param_o, param_c in zip(self.decoder_g_embed.parameters(), self.decoder_g_embed_c.parameters()):
            param_c.data = param_c.data * self.momentum + param_o.data * (1.0 - self.momentum)
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # pos_temp_embed = get_2d_sincos_pos_embed(self.pos_temp_embed.shape[-1], int(self.num_temp_patches**.5), cls_token=True)
        # self.pos_temp_embed.copy_(torch.from_numpy(pos_temp_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_context_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        if self.stop_grad_conv:
            if isinstance(self.patch_embed, PatchEmbed):
                # xavier_uniform initialization
                val = math.sqrt(6. / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dim))
                nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
                nn.init.zeros_(self.patch_embed.proj.bias)

                if self.stop_grad_conv:
                    self.patch_embed.proj.weight.requires_grad = False
                    self.patch_embed.proj.bias.requires_grad = False
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def unpatchify_feat(self, x):
        """
        x: (N, L, C)
        x_: (N, C, h, w)
        """
        N, L, C = x.shape 
        h = w = int(L**.5)
        
        x = x.reshape(shape=(N, h, w, C))
        x = x.permute(0, 3, 1, 2).contiguous() # fix Warning: Grad strides do not match

        return x
    
    def forward_encoder(self, x, ts):
        # embed patches
        x_t = []
        for t in ts:
            B, _, w, h = t.shape
            t = self.patch_embed(t)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            t = torch.cat((cls_tokens, t), dim=1)
            t = t + self.interpolate_pos_encoding(t, w, h)

            x_t.append(t)
        
        with torch.no_grad():
            B, _, w, h = x.shape
            x = self.patch_embed_c(x)
            
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # add positional encoding to each token
            x = x + self.interpolate_pos_encoding(x, w, h)
        
        # apply Transformer blocks
        for blk in self.blocks:
            t_s = []
            for t in x_t:
                t = blk(t)
                t_s.append(t)
            x_t = t_s
        
        t_s = []
        for t in x_t:
            t = self.norm(t)
            t_s.append(t)
        
        with torch.no_grad():
            for blk in self.blocks_c:
                x = blk(x)
            x = self.norm_c(x)

        return x, t_s

    def forward_mask_decoder(self, x, ts):
        # embed tokens
        x = self.decoder_embed(x)
        
        # add pos embed
        x = x + self.decoder_pos_embed

        t_s = []
        for t in ts:
            t = self.decoder_embed(t)
            t_s.append(t)

        # apply Correlation Transformer blocks
        preds = []
        for t in t_s:
            x_t, t = self.decoder_corr_block(x.float(), t.float())
            x_t = self.decoder_norm(x_t)
            # remove cls token & predictor projection
            m = self.unpatchify_feat(x_t[:, 1:, :])
            m = self.decoder_pred_mask(m)
            preds.append(m)

        return preds

    def mse_loss(self, preds, targets, alpha: float = 0.75, gamma: float = 2):
        
        N, H, W = targets.shape
        targets = targets.unsqueeze(1)
        probs = preds.sigmoid()
        mse_loss = self.mask_mse_loss(preds, targets.float())
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = mse_loss * ((1 - p_t) ** gamma)
        
        loss = loss.mean(1).sum() / (N*H*W)
        return loss
    
    def forward_mask_loss(self, preds, masks):
        loss = 0.0
        for i, (pred, mask) in enumerate(zip(preds, masks)):
            loss += self.mask_xent_loss(pred, mask)

        return loss
    
    def forward_head(self, x, ts):

        with torch.no_grad():
            x = self.decoder_g_embed_c(x[:, 0])
            targets = x
        
        preds = []
        for t in ts:
            t = self.decoder_g_embed(t[:, 0])
            t = self.decoder_predictor(t)
            preds.append(t)

        return preds, targets

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward_global_loss(self, preds, targets):
        loss = 0.0
        for i, pred in enumerate(preds):
            loss += self.contrastive_loss(pred, targets)
        return loss

    def forward(self, imgs, temps, masks):
        x, ts = self.forward_encoder(imgs, temps)
        
        loss = 0.0
        if self.sigma_cont > 0.0:
            global_preds, targets = self.forward_head(x, ts)
            global_loss = self.forward_global_loss(global_preds, targets)
            loss += self.sigma_cont*global_loss
        
        if self.sigma_corr > 0.0:
            mask_preds = self.forward_mask_decoder(x, ts)
            mask_loss = self.forward_mask_loss(mask_preds, masks)
            loss += self.sigma_corr*mask_loss
        else:
            mask_preds = None
        
        return loss, mask_preds

    def train_update(self, epochs, iter_per_epoch, cur_epoch):
        self.iter_per_epoch = iter_per_epoch
        self.max_iter = epochs * iter_per_epoch
        self.epoch = cur_epoch
        self.iteration = cur_epoch * iter_per_epoch

    def iter_update(self):

        if (self.iteration + 1) % self.accum_iter == 0:
            self.momentum = (
                self.end_momentum
                - (self.end_momentum - self.base_momentum)
                * (cos(pi * self.iteration / float(self.max_iter)) + 1)
                / 2
            )
            self.momentum_update()

        self.iteration += 1


def cim_vit_tiny_patch16_dec512d(**kwargs):
    model = CorrelationalAutoencoderViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def cim_vit_small_patch16_dec512d(**kwargs):
    model = CorrelationalAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def cim_vit_base_patch16_dec512d(**kwargs):
    model = CorrelationalAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def cim_vit_large_patch16_dec512d(**kwargs):
    model = CorrelationalAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def cim_vit_huge_patch14_dec512d(**kwargs):
    model = CorrelationalAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
cim_vit_tiny_patch16 = cim_vit_tiny_patch16_dec512d  # decoder: 512 dim
cim_vit_small_patch16 = cim_vit_small_patch16_dec512d  # decoder: 512 dim
cim_vit_base_patch16 = cim_vit_base_patch16_dec512d  # decoder: 512 dim
cim_vit_large_patch16 = cim_vit_large_patch16_dec512d  # decoder: 512 dim
cim_vit_huge_patch14 = cim_vit_huge_patch14_dec512d  # decoder: 512 dim
