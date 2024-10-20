import torch
import torch.nn as nn
import torch.nn.functional as F

class IIRConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1
                 ):
        super().__init__()
        self.out_channels = out_channels
        # self.w_a = [nn.Parameter(torch.zeros(1, in_channels, kernel_size, kernel_size)) for _ in range(self.out_channels)]
        self.conv_a = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        self.w_b = nn.Parameter(torch.Tensor([[[[1, 1, 0],
                                                [1, 0, 0],
                                                [0, 0, 0]]]]).expand(out_channels, 1, 3, 3), requires_grad=False)

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x:torch.Tensor):

        x = self.conv_a(x)
        out = F.conv2d(x, self.w_b, self.bias, stride=1, padding=1, groups=self.out_channels)
        out = x + out

        return out
    
    def freeze(self):
        self.w_b.requires_grad_(False)

class DepthWiseIIRConv(nn.Module):
    def __init__(self,
                 out_channels,
                 kernel_size):
        super().__init__()
        self.out_channels = out_channels
        self.w_a = nn.Parameter(torch.zeros(1, out_channels, kernel_size, kernel_size).cuda())

        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1).cuda())

    def forward(self, x:torch.Tensor):
        B, C, H, W = x.shape
        # print(self.w_a.device)
        x = F.pad(x, (1, 1, 1, 1), 'constant', 0)
        out = torch.Tensor(torch.zeros(B, self.out_channels, H+2, W+2)).cuda()

        for h in range(1, H+1):
            for w in range(1, W+1):
                ax = torch.sum(torch.mul(x[:, :, h-1:h+2, w-1:w+2], self.w_a[:, :, :, :]), dim=(2, 3), keepdim=True)
                out[:, :, h:h+1, w:w*1] = \
                    out[:, :, h-1:h, w-1:w] + out[:, :, h-1:h, w:w+1] + out[:, :, h:h+1, w-1:w] + ax
        # print("calculated")
        out = out[:, :, 1:-1, 1:-1]
        out = out + self.bias

        return out
    
    
if __name__ == '__main__':
    iirconv = DepthWiseIIRConv(100, 3)
    x = torch.Tensor(10, 100, 64, 48)

    out = iirconv(x)
    print(out.shape)
