import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from .myvit import PatchEmbedLayer, ScaledEmbedLayer, ScaledAttentionLayer, NormalEncoderBlock

class MultiHeadAtentionLayer(nn.Module):
    def __init__(self,
                 emb_dim:int,
                 head:int,
                 drop_rate:float,
                 is_bias:bool = True):
        super().__init__()
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = self.emb_dim // head
        self.sqrt = self.head_dim ** 0.5
        self.is_bias = is_bias

        self.Wq = nn.Linear(emb_dim, emb_dim, is_bias)
        self.Wk = nn.Linear(emb_dim, emb_dim, is_bias)
        self.Wv = nn.Linear(emb_dim, emb_dim, is_bias)

        self.attn_drop = nn.Dropout(drop_rate)

        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, q:torch.Tensor, kv:torch.Tensor):
        batch_size, num_patch, _ = q.size()

        q = self.Wq(q)
        k = self.Wk(kv)
        v = self.Wv(kv)
        # split (B, N, D) -> (B, N, head, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, k.shape[1], self.head, self.head_dim)
        v = v.view(batch_size, v.shape[1], self.head, self.head_dim)
        # (B, N, head, head_dim) -> (B, h, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_T = k.transpose(2, 3)
        dot = (q @ k_T) / self.sqrt
        attn = F.softmax(dot, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, num_patch, self.emb_dim)
        out = self.w_o(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self,
                 emb_dim:int,
                 head:int,
                 hidden_dim:int,
                 drop_rate:float = 0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)

        self.mhsa = ScaledAttentionLayer(emb_dim,
                                         emb_dim,
                                         head,
                                         drop_rate)
        
        self.ln2 = nn.LayerNorm(emb_dim)
        
        self.mha = MultiHeadAtentionLayer(emb_dim,
                                          head,
                                          drop_rate)
        
        self.ln3 = nn.LayerNorm(emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, feat_kpt:torch.Tensor, feat_img:torch.Tensor):
        feat_kpt = self.mhsa(self.ln1(feat_kpt)) + feat_kpt
        feat_kpt = self.mha(self.ln2(feat_kpt), feat_img) + feat_kpt
        feat_kpt = self.mlp(self.ln3(feat_kpt)) + feat_kpt
        return feat_kpt

@MODELS.register_module()
class MyVit2(BaseBackbone):
    def __init__(self,
                 in_channels:int,
                 emb_dim:int,
                 patch_size:int,
                 img_size = (192,256),
                 num_enc:int = 12,
                 num_dec:int = 12,
                 head:int = 12,
                 hidden_dim:int = 384,
                 drop_rate:float = 0.,
                 has_token:bool = False,
                 num_kpt:int = 17
                 ):
        super().__init__()
        self.num_enc = num_enc
        self.kpt_token = torch.Tensor([i for i in range(num_kpt)]).to(device="cuda", dtype=torch.int32)

        self.emb_img = PatchEmbedLayer(in_channels,
                                          emb_dim,
                                          patch_size,
                                          img_size,
                                          has_token)
        
        self.emb_kpt = nn.Embedding(num_kpt, emb_dim)
        
        self.norm = nn.LayerNorm(emb_dim)
        
        self.encoder = nn.ModuleList(
            [NormalEncoderBlock(emb_dim, head, hidden_dim, drop_rate) for _ in range(num_enc)]
        )

        self.decoder = nn.ModuleList()
        for _ in range(num_dec):
            self.decoder.append(DecoderBlock(emb_dim, head, hidden_dim, drop_rate))

        self.is_train = True

    def forward(self, x:torch.Tensor):
        # print(self.training)
        batch_size = x.shape[0]
        x = self.emb_img(x)
        img_feats = []
        for i in range(self.num_enc):
            x = self.encoder[i](x)
            if not self.training and i < self.num_enc - 1:
                continue
            img_feats.append(x)
        # print(len(img_feats))
        feat_kpt = self.emb_kpt(self.kpt_token)
        feat_kpt = feat_kpt.repeat(repeats=(batch_size, 1, 1))
        out = []
        for img_feat in img_feats:
            for layer in self.decoder:
                feat_kpt = layer(feat_kpt, img_feat)
            out.append(feat_kpt)
        # print(len(img_feats))
        return tuple(out)