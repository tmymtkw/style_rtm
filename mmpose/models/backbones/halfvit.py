from typing import List
import math

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

class EmbedBlock(BaseModule):
    def __init__(self, c:int, pSize:int, pNum:int, projSize:int, hasToken:bool=True):
        super().__init__()
        # (b, c, h, w) -> (b, h/p * w/p = pNum, (p * p * c) = projSize)
        self.c = c
        self.pSize = pSize
        self.pNum = pNum
        self.hasToken = hasToken
        if self.hasToken:
            self.pNum += 1
        self.projSize = projSize
        self.proj = nn.Conv2d(in_channels=self.c, 
                              out_channels=self.projSize, 
                              kernel_size=self.pSize, 
                              stride=self.pSize)
        self.posEmbed = nn.Parameter(torch.zeros(1, self.pNum, self.projSize))
        if not hasToken:
            return
        self.clsToken = nn.Parameter(torch.zeros(1, 1, self.projSize))

    def forward(self, x:torch.Tensor):
        batch = x.shape[0]
        
        x = self.proj(x) # (b, c, h, w) -> (b, projSize, h/p, w/p)
        x = x.flatten(2)              # -> (b, projSize, h/p * w/p)
        x = x.transpose(1, 2)         # -> (b, h/p * w/p, projsize)
        if self.hasToken:
            clsTokens = self.clsToken.expand(batch, -1, -1)
            x = torch.cat((clsTokens, x), dim=1)
        x = x + self.posEmbed
        # TODO dropout
        return x

class MlpBlock(BaseModule):
    def __init__(self, dim:int, hiddenDim:int, dropRatio:int, isBias:bool=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hiddenDim, isBias),
            nn.GELU(),
            nn.Dropout(p=dropRatio),
            nn.Linear(hiddenDim, dim, isBias),
            nn.Dropout(p=dropRatio)
        )

    def forward(self, x:torch.Tensor):
        x = self.mlp(self.norm(x)) + x
        return x
    
class ScaleAttentionBlock(BaseModule):
    def __init__(self, 
                 inDim:int, 
                 outDim:int, 
                 headNum:int, 
                 pNum:int, 
                 dropRatio:float=0.,
                 isBias:bool=True):
        assert outDim % headNum == 0, "outDim cant split"
        super().__init__()
        self.Wq = nn.Linear(inDim, outDim, isBias)
        self.Wk = nn.Linear(inDim, outDim, isBias)
        self.Wv = nn.Linear(inDim, outDim, isBias)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(outDim, outDim),
            nn.Dropout(dropRatio)
        )
        # (b, p, (h * d)) -> (b, p, h, d) -> (b, h, p, d)
        self.outDim = outDim
        self.pNum = pNum
        self.d = outDim // headNum
        self.h = headNum

    def split(self, t:torch.Tensor):
        batch = t.shape[0]
        t = t.view(batch, -1, self.d, self.h)
        t = t.permute(0, 2, 1, 3)
        return t
    
    def cat(self, t:torch.Tensor):
        batch = t.shape[0]
        t = t.view(batch, -1, self.outDim)
        return t

    def forward(self, x:torch.Tensor):
        # get q, k, v
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        # print(q.size())
        # split into multi heads
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)
        # attention
        attnWeight = torch.matmul(q, k.transpose(-1, -2))
        attnWeight = attnWeight / math.sqrt(self.d)
        attnWeight = self.softmax(attnWeight)
        # TODO dropout
        # output
        out = torch.matmul(attnWeight, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = self.cat(out)
        out = self.mlp(out)
        return out
    
class ScaleTransBlock(BaseModule):
    def __init__(self, inDim:int, outDim:int, heads:int, pNum:int, hiddenDim:int, dropRatio:float, attnBias:bool=True, mlpBias:bool=True):
        super().__init__()
        self.norm = nn.LayerNorm(inDim)
        self.attn = ScaleAttentionBlock(inDim=inDim, outDim=outDim, headNum=heads, pNum=pNum, isBias=attnBias)
        self.mlp = MlpBlock(dim=outDim, hiddenDim=hiddenDim, dropRatio=dropRatio, isBias=mlpBias)

    def forward(self, x):
        x = self.norm(x)
        x = self.attn(x)
        x = self.mlp(x)
        return x
    
class TransBlock(BaseModule):
    def __init__(self, inDim:int, heads:int, pNum:int, hiddenDim:int, dropRatio:int, attnBias:bool=True, mlpBias:bool=True):
        super().__init__()
        self.norm0 = nn.LayerNorm(inDim)
        self.attn = ScaleAttentionBlock(inDim=inDim, outDim=inDim, headNum=heads, pNum=pNum, isBias=attnBias)
        self.mlp = MlpBlock(dim=inDim, hiddenDim=hiddenDim, dropRatio=dropRatio, isBias=mlpBias)

    def forward(self, x:torch.Tensor):
        x = x + self.attn(self.norm0(x))
        x = self.mlp(x)
        return x
    
@MODELS.register_module()
class NormalVit(BaseBackbone):
    def __init__(self,
                 inChannel:int,
                 imgSize:tuple,
                 patchSize:int,
                 projSize:int,
                 heads:int, 
                 dropRatio:float=0.,
                 attnBias:bool=True, 
                 mlpBias:bool=True, 
                 hasToken:bool=True,
                 init_cfg: dict | List[dict] | None = None):
        super(NormalVit, self).__init__()
        self.emb_dim = projSize
        self.patchNum = (imgSize[0]//patchSize) * (imgSize[1]//patchSize)
        self.hasToken = hasToken
        self.out_size = (imgSize[0]//patchSize, imgSize[1]//patchSize)

        self.emb = EmbedBlock(inChannel, patchSize, self.patchNum, projSize, hasToken)

        self.encoder = nn.Sequential(
            *[TransBlock(projSize, heads, self.patchNum, projSize*4, dropRatio, attnBias, mlpBias) for _ in range(5)]
        )

    def forward(self, x:torch.Tensor):
        batch = x.shape[0]

        x = self.emb(x)
        x = self.encoder(x)
        out = x[:, self.hasToken:]
        out = out.reshape(batch, self.out_size[0], self.out_size[1], self.emb_dim).permute(0, 3, 1, 2)
        return tuple([out])

@MODELS.register_module()
class HalfVit(BaseBackbone):
    def __init__(self, 
                 inChannel:int, 
                 imgSize:tuple,
                 patchSize:int,
                 projSize:int,
                 heads:int, 
                 dropRatio:float=0.,
                 attnBias:bool=True, 
                 mlpBias:bool=True, 
                 hasToken:bool=True,
                 init_cfg: dict | List[dict] | None = None):
        super().__init__(init_cfg)
        assert len(imgSize) == 2, "uncorrect img size"
        self.imgSize = imgSize
        self.c = inChannel
        self.pSize = patchSize
        self.gridSize = [imgSize[0]//self.pSize, imgSize[1]//self.pSize]
        self.pNum = (imgSize[0] // self.pSize) * (imgSize[1] // self.pSize)
        self.d = projSize
        self.hasToken = hasToken
        # stage 1 (b, c, h, w) -> (b, pNum, d)
        self.embed = EmbedBlock(self.c, self.pSize, self.pNum, self.d, self.hasToken)
        # stage 2.0 (b, pNum, d) -> (b, pNum, d/2)
        self.downTrans0 = ScaleTransBlock(inDim=self.d, 
                                          outDim=self.d//2, 
                                          heads=heads, 
                                          pNum=self.pNum,
                                          hiddenDim=self.d*2*3,
                                          dropRatio=dropRatio,
                                          attnBias=attnBias, 
                                          mlpBias=mlpBias)
        self.down()
        # stage 2.1 (b, pNum, d/2) -> (b, pNum, d/4)
        self.downTrans1 = ScaleTransBlock(inDim=self.d, 
                                          outDim=self.d//2, 
                                          heads=heads, 
                                          pNum=self.pNum, 
                                          hiddenDim=self.d*2*3,
                                          dropRatio=dropRatio,
                                          attnBias=attnBias, 
                                          mlpBias=mlpBias)
        self.down()
        # stage 2.2 (b, pNum, d/4) -> (b, pNum, d/8)
        self.downTrans2 = ScaleTransBlock(inDim=self.d, 
                                          outDim=self.d//2, 
                                          heads=heads, 
                                          pNum=self.pNum, 
                                          hiddenDim=self.d*2*3,
                                          dropRatio=dropRatio,
                                          attnBias=attnBias, 
                                          mlpBias=mlpBias)
        self.down()
        # stage 3
        self.transBlock = TransBlock(inDim=self.d, 
                                     heads=heads, 
                                     pNum=self.pNum, 
                                     hiddenDim=self.d*3, 
                                     dropRatio=dropRatio,
                                     attnBias=attnBias, 
                                     mlpBias=mlpBias)
        # stage 4.2
        self.upNorm2 = nn.LayerNorm(self.d)
        self.upAttn2 = ScaleAttentionBlock(inDim=self.d, outDim=self.d*2, headNum=heads, pNum=self.pNum, isBias=attnBias)
        self.upMlp2 = MlpBlock(dim=self.d*2, hiddenDim=self.d*2*3, dropRatio=dropRatio, isBias=mlpBias)
        self.up()
        # stage 4.1
        self.upNorm1 = nn.LayerNorm(self.d)
        self.upAttn1 = ScaleAttentionBlock(inDim=self.d, outDim=self.d*2, headNum=heads, pNum=self.pNum, isBias=attnBias)
        self.upMlp1 = MlpBlock(dim=self.d*2, hiddenDim=self.d*2*3, dropRatio=dropRatio, isBias=mlpBias)
        self.up()
        # stage 4.0
        self.upNorm0 = nn.LayerNorm(self.d)
        self.upAttn0 = ScaleAttentionBlock(inDim=self.d, outDim=self.d*2, headNum=heads, pNum=self.pNum, isBias=attnBias)
        self.upMlp0 = MlpBlock(dim=self.d*2, hiddenDim=self.d*2*3, dropRatio=dropRatio, isBias=mlpBias)

    def down(self):
        self.d //= 2

    def up(self):
        self.d *= 2

    def forward(self, x:torch.Tensor):
        # 1. patch embedding
        x = self.embed(x)
        # print(x.shape)
        # 2. down encoding
        prev0 = x
        x = self.downTrans0(x)

        prev1 = x
        x = self.downTrans1(x)

        prev2 = x
        x = self.downTrans2(x)

        # 3. normal transformer
        x = self.transBlock(x)

        # 4. up encoding
        x = self.upNorm2(x)
        x = self.upAttn2(x) + prev2
        x = self.upMlp2(x)

        x = self.upNorm1(x)
        x = self.upAttn1(x) + prev1
        x = self.upMlp1(x)

        x = self.upNorm0(x)
        x = self.upAttn0(x) + prev0
        x = self.upMlp0(x)

        out = []
        out.append(self.formOut(x))

        return tuple(out)
    
    # TODO 調整
    def formOut(self, x:torch.Tensor):
        batch = x.shape[0]
        imgToken = x[:, self.hasToken:]
        return imgToken.reshape(batch, *self.gridSize, -1).permute(0, 3, 1, 2)
    
@MODELS.register_module()
class HalfCatVit(BaseBackbone):
    def __init__(self, 
                 inChannel:int, 
                 imgSize:tuple,
                 patchSize:int,
                 projSize:int,
                 heads:int, 
                 dropRatio:float=0.,
                 attnBias:bool=True, 
                 mlpBias:bool=True, 
                 hasToken:bool=True,
                 init_cfg: dict | List[dict] | None = None):
        super().__init__(init_cfg)
        assert len(imgSize) == 2, "uncorrect img size"
        self.imgSize = imgSize
        self.c = inChannel
        self.pSize = patchSize
        self.gridSize = [imgSize[0]//self.pSize, imgSize[1]//self.pSize]
        self.pNum = (imgSize[0] // self.pSize) * (imgSize[1] // self.pSize)
        self.d = projSize
        self.hasToken = hasToken
        # stage 1 (b, c, h, w) -> (b, pNum, d)
        self.embed = EmbedBlock(self.c, self.pSize, self.pNum, self.d, self.hasToken)
        # stage 2.0 (b, pNum, d) -> (b, pNum, d/2)
        self.downTrans0 = ScaleTransBlock(inDim=self.d, 
                                          outDim=self.d//2, 
                                          heads=heads, 
                                          pNum=self.pNum,
                                          hiddenDim=self.d*2*3,
                                          dropRatio=dropRatio,
                                          attnBias=attnBias, 
                                          mlpBias=mlpBias)
        self.down()
        # stage 2.1 (b, pNum, d/2) -> (b, pNum, d/4)
        self.downTrans1 = ScaleTransBlock(inDim=self.d, 
                                          outDim=self.d//2, 
                                          heads=heads, 
                                          pNum=self.pNum, 
                                          hiddenDim=self.d*2*3,
                                          dropRatio=dropRatio,
                                          attnBias=attnBias, 
                                          mlpBias=mlpBias)
        self.down()
        # stage 2.2 (b, pNum, d/4) -> (b, pNum, d/8)
        self.downTrans2 = ScaleTransBlock(inDim=self.d, 
                                          outDim=self.d//2, 
                                          heads=heads, 
                                          pNum=self.pNum, 
                                          hiddenDim=self.d*2*3,
                                          dropRatio=dropRatio,
                                          attnBias=attnBias, 
                                          mlpBias=mlpBias)
        self.down()
        # stage 3
        self.transBlock = TransBlock(inDim=self.d, 
                                     heads=heads, 
                                     pNum=self.pNum, 
                                     hiddenDim=self.d*3, 
                                     dropRatio=dropRatio,
                                     attnBias=attnBias, 
                                     mlpBias=mlpBias)
        # stage 4.2
        self.upNorm2 = nn.LayerNorm(self.d)
        self.upAttn2 = ScaleAttentionBlock(inDim=self.d, outDim=self.d, headNum=heads, pNum=self.pNum, isBias=attnBias)
        self.upMlp2 = MlpBlock(dim=self.d*2, hiddenDim=self.d*2*3, dropRatio=dropRatio, isBias=mlpBias)
        self.up()
        # stage 4.1
        self.upNorm1 = nn.LayerNorm(self.d)
        self.upAttn1 = ScaleAttentionBlock(inDim=self.d, outDim=self.d, headNum=heads, pNum=self.pNum, isBias=attnBias)
        self.upMlp1 = MlpBlock(dim=self.d*2, hiddenDim=self.d*2*3, dropRatio=dropRatio, isBias=mlpBias)
        self.up()
        # stage 4.0
        self.upNorm0 = nn.LayerNorm(self.d)
        self.upAttn0 = ScaleAttentionBlock(inDim=self.d, outDim=self.d, headNum=heads, pNum=self.pNum, isBias=attnBias)
        self.upMlp0 = MlpBlock(dim=self.d*2, hiddenDim=self.d*2*3, dropRatio=dropRatio, isBias=mlpBias)

    def down(self):
        self.d //= 2

    def up(self, scale:int=2):
        self.d *= scale

    def forward(self, x:torch.Tensor):
        # 1. patch embedding
        x = self.embed(x)

        # 2. down encoding
        x = self.downTrans0(x)
        prev0 = x

        x = self.downTrans1(x)
        prev1 = x

        x = self.downTrans2(x)
        prev2 = x

        # 3. normal transformer
        x = self.transBlock(x)

        # 4. up encoding
        x = self.upAttn2(self.upNorm2(x)) + x
        x = torch.cat((prev2, x), dim=2)
        x = self.upMlp2(x)

        x = self.upAttn1(self.upNorm1(x)) + x
        x = torch.cat((prev1, x), dim=2)
        x = self.upMlp1(x)

        x = self.upAttn0(self.upNorm0(x)) + x
        x = torch.cat((prev0, x), dim=2)
        x = self.upMlp0(x)

        out = []
        out.append(self.formOut(x))

        return tuple(out)
    
    # TODO 調整
    def formOut(self, x:torch.Tensor):
        batch = x.shape[0]
        imgToken = x[:, self.hasToken:]
        return imgToken.reshape(batch, *self.gridSize, -1).permute(0, 3, 1, 2)
