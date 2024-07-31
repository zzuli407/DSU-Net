import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
import math
import torchvision.ops
from einops import rearrange
import numbers

def PhiTPhi_fun(x, PhiW):
    temp = F.conv2d(x, PhiW, padding=0, stride=32, bias=None)
    temp = F.conv_transpose2d(temp, PhiW, stride=32)
    return temp

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class NBLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(NBLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class BLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = NBLayerNorm(dim)
        else:
            self.body = BLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class DGB1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DGB1, self).__init__()
        self.norm1 = LayerNorm(in_channels, 'WithBias')
        self.Deconv1 = nn.Sequential(
            DeformableConv2d(in_channels=in_channels, out_channels=in_channels), nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=bias), nn.GELU()
        )
        self.Deconv2 = nn.Sequential(
            DeformableConv2d(in_channels=in_channels, out_channels=out_channels), nn.GELU()
        )

    def forward(self, x1, x2):
        x_in = self.norm1(x1)
        x = self.Deconv1(x_in)
        x = self.conv2(x + x2)
        out = self.Deconv2(x) + x1
        return out

class DGB2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DGB2, self).__init__()
        self.norm1 = LayerNorm(in_channels, 'WithBias')
        self.Deconv1 = nn.Sequential(
            DeformableConv2d(in_channels=in_channels, out_channels=in_channels), nn.GELU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=bias), nn.GELU()
        )
        self.Deconv2 = nn.Sequential(
            DeformableConv2d(in_channels=in_channels, out_channels=in_channels), nn.GELU()
        )
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x1, x2):
        x_in = self.norm1(x1)
        x = self.Deconv1(x_in)
        x = self.conv1(x * x2)
        out = self.conv2(self.Deconv2(x) + x1)
        return out

# Define Cross Attention Block
class CAtten(torch.nn.Module):
    def __init__(self, channels):
        super(CAtten, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        
        self.norm_x = LayerNorm(1, 'WithBias')
        self.norm_z = LayerNorm(31, 'WithBias')

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.p = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        self.v = nn.Conv2d(in_channels=self.channels+1, out_channels=self.channels+1, kernel_size=1, stride=1, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
        )

    def forward(self, x, z):
        
        x0 = self.norm_x(x)
        z0 = self.norm_z(z)
        
        z1 = self.t(z0)
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1) # k,v: b, c, hw
        x1 = self.p(x0) # q: b, c, hw
        x1 = x1.view(b, c, -1)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        x1 = torch.nn.functional.normalize(x1, dim=-1)
        x_t = x1.permute(0, 2, 1) # q: b, hw, c
        att = torch.matmul(z1, x_t) # k*q
        att = self.softmax(att) # b, c, c
        
        z2 = self.g(z0)
        z_v = z2.view(b, c, -1) # v: b, c, hw
        out_x = torch.matmul(att, z_v) # qk*v
        out_x = out_x.view(b, c, h, w)
        out_x = self.w(out_x) + self.pos_emb(z2) + z
        y = self.v(torch.cat([x, out_x], 1))

        return y


# Define ISCA block
class Atten(torch.nn.Module):
    def __init__(self, channels):
        super(Atten, self).__init__()
               
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        self.conv_q = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
        )
        self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        
    def forward(self, pre, cur):

        b, c, h, w = pre.shape
        pre_ln = self.norm1(pre)
        cur_ln = self.norm2(cur)
        q = self.conv_q(cur_ln)
        q = q.view(b, c, -1)
        k, v = self.conv_kv(pre_ln).chunk(2, dim=1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        att = torch.matmul(q, k.permute(0, 2, 1))
        att = self.softmax(att)
        out = torch.matmul(att, v).view(b, c, h, w)
        out = self.conv_out(out) + cur
        
        return out


class StageDenoiser(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(StageDenoiser, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.GELU()
        )
        self.convd3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=bias),
            nn.Sigmoid()
        )
        self.conv33 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.GELU()
        )
        self.convd33 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=bias),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(in_channels=in_channels*2, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x1 = self.conv3(x)
        x2 = self.convd3(x)+x
        x_ = x1*x2
        x3 = self.convd33(x_)+x_
        x33 = self.conv33(x_)
        out = self.conv1(torch.cat([x3, x33], 1))
        return out

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        )

    def forward(self, x):
        out = self.deconv(x)
        return out

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        # init weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_channels, 3 * kernel_size * kernel_size, kernel_size=kernel_size,
                                          stride=stride, padding=self.padding, bias=True)
        # init
        self.reset_parameters()
        self._init_weight()

    def reset_parameters(self):
        n = self.in_channels * (self.kernel_size**2)
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def _init_weight(self):
        # init offset_mask conv
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = torchvision.ops.deform_conv2d(input=x, offset=offset,  weight=self.weight, bias=self.bias,
                                          padding=self.padding, mask=mask, stride=self.stride)
        return x

    def flops(self, H, W):
        flops = 0
        # offset
        flops += H * W * 3 * self.kernel_size * self.kernel_size * self.in_channels * self.kernel_size * self.kernel_size
        # df
        flops += H * W * self.out_channels * self.in_channels

        return flops


class Stage(torch.nn.Module):
    def __init__(self):
        super(Stage, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.lam_step = nn.Parameter(torch.Tensor([0.5]))
        self.DGB1 = DGB1(31, 31)
        self.DGB2 = DGB2(31, 32)
        self.atten = Atten(32)
        self.catten = CAtten(channels=31)
        self.conv1 = nn.Conv2d(31, 32, 1, 1, padding=0)
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        self.forward = StageDenoiser(in_channels=32, out_channels=32)
        self.backward = StageDenoiser(in_channels=32, out_channels=32)
        
    def forward(self, x, x_pre, x_cur, x_Weight, x_init, map_bata1, map_bata2):

        # GDB
        lam_step = torch.div(self.lam_step, self.lam_step * self.softmax(x) + 1)
        x = x - lam_step * PhiTPhi_fun(x, x_Weight)
        x_input = x + lam_step * x_init

        z = self.DGB1(x_pre, x_cur) #31

        x_input = self.DGB2(map_bata1 * x_cur + z, x_input)  #32

        # PMM
        z = self.conv1(z) #32
        x = self.norm1(z)
        atten = self.atten(x, x)
        x_f = self.forward(atten) + z  #32

        x = self.norm2(x_input)  #32
        catten = self.catten(atten[:, :1, :, :], x[:, 1:, :, :])
        x_b = self.backward(map_bata2 * x_f + catten) + x_input

        x_pred = x_f + x_b

        return x_pred


class DSU(torch.nn.Module):
    def __init__(self, LayerN, sr):
        super(DSU, self).__init__()
        onelayer = []
        self.LayerN = LayerN
        self.patch_size = 32
        self.n_input = int(sr * 1024)
        for i in range(LayerN):
            onelayer.append(Stage())
        self.x_weight = nn.Parameter(init.xavier_normal_(torch.Tensor(self.n_input, 1, self.patch_size, self.patch_size)))
        self.fs = nn.ModuleList(onelayer)
        self.f1 = nn.Conv2d(1, 31, 3, padding=1, bias=True)
        self.f2 = nn.Conv2d(1, 31, 3, padding=1, bias=True)
        self.map_bata1 = nn.Parameter(torch.zeros((1, 31, 1, 1)) + 1e-2, requires_grad=True)
        self.map_bata2 = nn.Parameter(torch.zeros((1, 32, 1, 1)) + 1e-2, requires_grad=True)

    def forward(self, x):

        x_init = F.conv2d(x, self.x_weight, stride=self.patch_size, padding=0, bias=None)
        x_init = F.conv_transpose2d(x_init, self.x_weight, stride=self.patch_size)
        x = x_init
        x_pre = self.f1(x)
        x_cur = self.f2(x)

        for i in range(self.LayerN):
            x_dual = self.fs[i](x, x_pre, x_cur, self.x_weight, x_init, self.map_bata1, self.map_bata2)
            x = x_dual[:, :1, :, :]
            x_pre = x_cur
            x_cur = x_dual[:, 1:, :, :]

        x_final = x

        return x_final
