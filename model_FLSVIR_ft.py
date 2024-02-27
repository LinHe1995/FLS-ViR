# --------------------------------------------------------
# References:
# Reformer: https://github.com/google/trax/tree/master/trax/models/reformer
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from reversible import ReversibleSequence
from timm.models.vision_transformer import PatchEmbed

TOKEN_SELF_ATTN_VALUE = -5e4

class PreNorm(nn.Module):
    def __init__(self, drop_path, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn
        self.drop_path = DropPath(drop_path)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.drop_path(self.fn(x, **kwargs))

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, **kwargs):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, **kwargs):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, seq_len, patch_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        kw_dim = seq_len 
        self.fc = nn.Linear(patch_dim, kw_dim)
        self.fc2 = nn.Linear(kw_dim, patch_dim)

        self.norm = nn.LayerNorm(kw_dim)

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.fc(attn)
        attn = self.norm(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn).transpose(-2, -1)

        x = (attn @ v).transpose(-2, -1)
        x = self.fc2(x).transpose(-2, -1).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x

class FLSVIR(nn.Module):
    def __init__(self, emb_dim, mlp_dim, seq_len, num_layers, num_patches, num_heads,mlp_ratio,
                 dropout_rate,attn_dropout_rate, layer_dropout = 0.0, reverse_thres = 0.0, use_scale_norm = False, use_rezero = False):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        
        mlp_hidden_dim = int(emb_dim * mlp_ratio)
        get_mlp = Mlp(in_features=emb_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.7)
        get_attn = Attention(emb_dim, seq_len, patch_dim=num_patches+1, num_heads=num_heads, qkv_bias=False, attn_drop=0.7, proj_drop=0.7)
        norm_type = nn.LayerNorm
        residual_fn_wrapper = partial(PreNorm, 0.7, norm_type, emb_dim)

        blocks = []
        for ind in range(num_layers):

            f = residual_fn_wrapper(get_attn)
            g = residual_fn_wrapper(get_mlp)
            blocks.append(nn.ModuleList([f, g]))

        self.layers = ReversibleSequence(nn.ModuleList(blocks), layer_dropout = layer_dropout, reverse_thres = reverse_thres, send_signal = True)


    def forward(self, x, **kwargs):

        x = torch.cat([x, x], dim = -1)
        x = self.layers(x, **kwargs)
        x = torch.stack(x.chunk(2, dim=-1)).mean(dim=0)

        return x


class FLSReLM(nn.Module):
    def __init__(self, emb_dim, mlp_dim, seq_len, num_layers, num_heads = 8, mlp_ratio= 4.0, num_classes = 27, image_size=(224,224), patch_size=(17,17),
                 dropout_rate = 0.1, attn_dropout_rate = 0.0, layer_dropout = 0.0, reverse_thres = 0.0,
                 use_scale_norm = False, use_rezero = False):
        super().__init__()
        emb_dim = emb_dim
        h, w = image_size
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw

        self.patch_embed = PatchEmbed(w, fw, 3, emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.flsre = FLSVIR(emb_dim, mlp_dim, seq_len, num_layers, num_patches, num_heads, mlp_ratio,
                           dropout_rate = dropout_rate, attn_dropout_rate = attn_dropout_rate, layer_dropout = layer_dropout, reverse_thres = reverse_thres,
                           use_scale_norm = use_scale_norm, use_rezero = use_rezero)
        
        self.new_fc_i = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim // 4, emb_dim, bias=False),
            nn.ReLU(inplace=True)
        )

        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x, **kwargs):


        emb = self.patch_embed(x)
        emb = emb + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(emb.shape[0], -1, -1)
        emb = torch.cat((cls_tokens, emb), dim=1)

        x = self.flsre(emb, **kwargs)
        x = self.new_fc_i(x)
        x = self.norm(x)
        logits = self.classifier(x[:, 0])
        return logits
