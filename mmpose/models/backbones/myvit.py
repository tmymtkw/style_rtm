import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

class PatchEmbedLayer(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 emb_dim:int=384,
                 patch_size:int=16,
                 img_size=(192, 256),
                 has_token:bool=True):
        super(PatchEmbedLayer, self).__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.has_token = has_token
        
        self.num_patches = (self.img_size[0]//self.patch_size) * (self.img_size[1]//self.patch_size)
        if self.has_token:
            self.num_patches += 1

        self.patch_emb = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.emb_dim,
                                   kernel_size=self.patch_size,
                                   stride=self.patch_size)

        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

        if not self.has_token:
            return
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

    def forward(self, x:torch.Tensor):
        # patch embedding (B, C, H, W) -> (B, emb_dim, H/patch_size, W/patch_size)
        x = self.patch_emb(x)
        # flatten (B, emb_dim, H/patch_size, W/patch_size) -> (B, emb_dim, num_patches) -> (B, num_patches, emb_dim)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        # cls_token
        if self.has_token:
            x = torch.cat([self.cls_token.repeat(repeats=(x.size(0),1,1)), x], dim=1)
        # pos embed
        x = x + self.pos_emb
        return x
    
class ScaledEmbedLayer(nn.Module):
    def __init__(self,
                 in_channels:int,
                 emb_dim:int,
                 patch_size:int,
                 img_size=(192, 256),
                 num_token:int=17):
        super(ScaledEmbedLayer, self).__init__()
        # var
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_token = num_token
        self.num_patches = (self.img_size[0]//self.patch_size) * (self.img_size[1]//self.patch_size)

        # module
        self.patch_emb = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches + self.num_token, emb_dim))
        self.kpt_token = nn.Parameter(torch.randn(1, self.num_token, emb_dim))

    def forward(self, x:torch.Tensor):
        x = self.patch_emb(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = torch.cat([self.kpt_token.repeat(repeats=(x.size(0),1,1)), x], dim=1)
        x = x + self.pos_emb
        return x
    
class ScaledAttentionLayer(nn.Module):
    def __init__(self,
                 emb_dim:int=384,
                 out_dim:int=384,
                 head:int=12,
                 drop_rate:float=0.,
                 is_bias:bool=True):
        super(ScaledAttentionLayer, self).__init__()
        self.head = head
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.head_dim = self.emb_dim // head
        self.sqrt = self.head_dim ** 0.5
        self.is_bias = is_bias

        self.Wq = nn.Linear(emb_dim, out_dim, bias=is_bias)
        self.Wk = nn.Linear(emb_dim, out_dim, bias=is_bias)
        self.Wv = nn.Linear(emb_dim, out_dim, bias=is_bias)

        self.attn_drop = nn.Dropout(drop_rate)

        self.w_o = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x:torch.Tensor):
        batch_size, num_patch, _ = x.size()

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        # split (B, N, D) -> (B, N, head, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)
        # (B, N, head, head_dim) -> (B, h, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_T = k.transpose(2, 3)
        dot = (q @ k_T) / self.sqrt
        attn = F.softmax(dot, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, num_patch, self.out_dim)
        out = self.w_o(out)
        return out
    
class NormalEncoderBlock(nn.Module):
    def __init__(self,
                 emb_dim:int=384,
                 head:int=12,
                 hidden_dim:int=192*4,
                 drop_rate:float=0.):
        super(NormalEncoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(emb_dim)

        self.attn = ScaledAttentionLayer(emb_dim, emb_dim, head, drop_rate)

        self.ln2 = nn.LayerNorm(emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x:torch.Tensor):
        out = self.attn(self.ln1(x)) + x
        out = self.mlp(self.ln2(out)) + out
        return out

@MODELS.register_module()
class MyVit(BaseBackbone):
    def __init__(self,
                 in_channels:int=3,
                 emb_dim:int=384,
                 patch_size:int=16,
                 img_size=(192, 256),
                 num_blocks:int=12,
                 head:int=12,
                 hidden_dim:int=384*4,
                 drop_rate:float=0.,
                 num_token:int=17,
                 output_type:str="featmap"):
        super(MyVit, self).__init__()
        self.output_type = output_type
        self.emb_dim = emb_dim
        self.num_token = num_token
        self.out_size = (img_size[0]//patch_size, img_size[1]//patch_size)

        self.emb_layer = ScaledEmbedLayer(in_channels,
                                         emb_dim,
                                         patch_size,
                                         img_size,
                                         num_token)
        
        self.encoder = nn.Sequential(
            *[NormalEncoderBlock(emb_dim, head, hidden_dim, drop_rate) for _ in range(num_blocks)]
        )

    def forward(self, x:torch.Tensor):
        x = self.emb_layer(x)
        x = self.encoder(x)
        out = self.format_output(x)
        # print(len(out), out[0].shape)
        return tuple(out)
    
    def format_output(self, x:torch.Tensor):
        batch = x.shape[0]
        if self.output_type == "featmap":
            out = x[:, self.num_token:]
            out = [out.reshape(batch, self.out_size[0], self.out_size[1], self.emb_dim).permute(0, 3, 1, 2)]
        elif self.output_type == "token":
            out = [x[:, :self.num_token]]
        elif self.output_type == "both":
            feat_kpt = x[:, :self.num_token]

            feat_img = x[:, self.num_token:]
            feat_img = feat_img.reshape(batch, self.out_size[0], self.out_size[1], self.emb_dim).permute(0, 3, 1, 2)

            out = [feat_kpt, feat_img]
        else:
            return x
        return out