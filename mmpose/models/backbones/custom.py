import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

class custom_Patching(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph = patch_size, pw = patch_size)

    def forward(self, x):
        x = self.net(x)
        return x

class custom_LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        super().__init__()
        self.lp = nn.Linear(patch_dim, dim)

    def forward(self, x):
        x = self.lp(x)
        return x

class custom_Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches+1, dim))

    def forward(self, x):
        batch_size, _, __ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embed
        return x    # (1, n_patches+1, dim)

class custom_MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.4):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.GeLU(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class custom_MHSA(nn.Module):
    def __init__(self, dim, num_head):
        super().__init__()
        self.n_heads = dim // num_head
        self.w_q = nn.Linear(dim, dim, bias=True)
        self.w_k = nn.Linear(dim, dim, bias=True)
        self.w_v = nn.Linear(dim, dim, bias=True)
        self.split2heads = Rearrange("b n (h d) -> b h n d", h=self.n_heads)
        self.concat = Rearrange("b h n d -> b n (h d)", h=self.n_heads)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x) # 全てxを代入するので、self attentionになる？
        q = self.split2heads(q)
        k = self.split2heads(k)
        v = self.split2heads(v)

        logit = torch.matmul(q, k.transpose(-1, -2) * (self.dim_heads ** -0.5))
        attn_weight = nn.softmax(logit, dim=-1)
        output = torch.matmul(attn_weight, v)
        output = self.concat(output)
        return output

class custom_TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, depth):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mhsa = custom_MHSA(dim=dim, num_head=n_heads)
        self.mlp = custom_MLP(dim=dim, hidden_dim=mlp_dim)
        self.depth = depth

    def forward(self, x):
        for _ in range(depth):
            x = self.mhsa(self.norm(x)) + x
            x = self.mlp(self.norm(x)) + x
        return x

class custom_ViT(nn.Module):
    def __init__(self, img_size=(256,192), out_size=(64,48), patch_size=16, lp_dim=384, in_channels=3, n_heads=4, depth=4, mlp_dim=384*4):
        super().__init__()
        n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) # 16 * 12  384 = 16 * 12 * 2
        # n_patches = out_size[0] * out_size[1]
        patch_dim = in_channels * (patch_size ** 2)
        dim = lp_dim # = embed_dim
        self.depth = depth

        self.patching = custom_Patching(patch_size)
        self.linear_projection = custom_LinearProjection(patch_dim=patch_dim, dim=dim)
        self.embedding = custom_Embedding(dim=dim, n_patches=n_patches)
        self.transformerblock = custom_TransformerBlock(dim=dim, n_heads=n_heads, mlp_dim=mlp_dim, depth=depth)

    def forward(self, x):
        x = self.patching(x)
        x = self.linear_projection(x)
        x = self.embedding(x)
        x = self.tranformerblock(x)
        return x



@MODELS.register_module()
class CustomBackbone(BaseBackbone):
    def __init__():
        super().__init__()
        self.vit = custom_ViT(img_size=(256,192), out_size=(64,48), patch_size=16, lp_dim=384, in_channels=2, n_heads=4, depth=4, mlp_dim=384*4)
        self.vit2 = custom_ViT(img_size=(256,192), out_size=(64,48), patch_size=16, lp_dim=384, in_channels=1, n_heads=4, depth=4, mlp_dim=384*4)
        
    def forward(self, x):
        xl = x[0, :, :]
        xab = x[1:, :, :]
        xl = self.vit2(xl)
        xab = self.vit(xab)
        ourput = xl + xab
        return output