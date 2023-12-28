import torch.nn as nn
import torch.nn.functional as F
import torch
import kornia as K
import collections

import math
from itertools import repeat

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

def quant_max(tensor):
    """
    Returns the max value for symmetric quantization.
    """
    return torch.abs(tensor.detach()).max() + 1e-8

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

_pair = _ntuple(2)

class quant_weight(nn.Module):
    """
    Quantization function for quantize weight with maximum.
    """

    def __init__(self, k_bits):
        super(quant_weight, self).__init__()
        self.k_bits = k_bits
        self.qmax = 2. ** (k_bits - 1) - 1.
        self.round = TorchRound()

    def forward(self, input):
        # no learning
        max_val = quant_max(input)
        weight = input * self.qmax / max_val
        q_weight = self.round(weight)
        q_weight = q_weight * max_val / self.qmax
        return q_weight


def TorchRound():
    """
    Apply STE to clamp function.
    """

    class identity_quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            out = torch.round(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return identity_quant().apply


def quant_conv3x3(in_channels, out_channels, kernel_size=3, padding=1, stride=1, k_bits=32, bias=False, groups=1):
    return QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, k_bits=k_bits, bias=bias, groups=groups)


def quant_conv1x1(in_channels, out_channels, kernel_size=1, padding=1, stride=1, k_bits=32, bias=False, groups=1):
    return QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, k_bits=k_bits, bias=bias, groups=groups)


class QuantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, k_bits=32, ):
        super(QuantConv2d, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))


        else:
            self.register_parameter('bias', None)
        self.k_bits = k_bits
        self.quant_weight = quant_weight(k_bits=k_bits)
        self.output = None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameter(self):
        stdv = 1.0 / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias_flag:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input, bits=None,epoch=None):
        if bits is not None:
            if input.size(0) != 1:
                if bits is not None: #train
                    if epoch < 150:
                        return nn.functional.conv2d(input, self.quant_weight(self.weight), self.bias, self.stride, self.padding,self.dilation, self.groups)
            
                for i in range(input.size(0)):
                    if bits[i] == 32:
                        out = nn.functional.conv2d(input[i].unsqueeze(0), self.weight, self.bias, self.stride,
                                                   self.padding, self.dilation, self.groups)
                    else:
                        self.quant_weight = quant_weight(k_bits=bits[i])
                        weight_q = self.quant_weight(self.weight)
                        out = nn.functional.conv2d(input[i].unsqueeze(0), weight_q, self.bias, self.stride,
                                                   self.padding, self.dilation, self.groups)

                    if i == 0:
                        out_stacked = out
                    else:
                        out_stacked = torch.cat([out_stacked, out], dim=0)
                return out_stacked
            else:
                self.quant_weight = quant_weight(k_bits=bits)


        return nn.functional.conv2d(input, self.quant_weight(self.weight), self.bias, self.stride, self.padding,
                                    self.dilation, self.groups)


#############################

class BitSelectorw(nn.Module):
    def __init__(self, n_feats, bias=False, ema_epoch=1, search_space=[8,16], linq=False):

        super(BitSelectorw, self).__init__()


        self.search_space = search_space

        self.net_small = nn.Sequential(
            nn.Linear(n_feats + 8, len(search_space))
        )


        nn.init.ones_(self.net_small[0].weight)
        nn.init.zeros_(self.net_small[0].bias)
        nn.init.ones_(self.net_small[0].bias[-1])


    def forward(self, x):
        weighted_bits = x[3]
        bits = x[2]
        grad = x[0]
        x = x[1]

        layer_std_s = torch.std(x, (2, 3)).detach()
        x_embed = torch.cat([grad, layer_std_s], dim=1)  # [B, C+2]

        bit_type = self.net_small(x_embed)

        flag = torch.argmax(bit_type, dim=1)
        p = F.softmax(bit_type, dim=1)

        if len(self.search_space) == 4:
            p1 = p[:, 0]
            p2 = p[:, 1]
            p3 = p[:, 2]
            p4 = p[:, 3]

            bits_hard = (flag == 0) * self.search_space[0] + (flag == 1) * self.search_space[1] + (flag == 2) * \
                        self.search_space[2] + (flag == 3) * self.search_space[3]
            bits_soft = p1 * self.search_space[0] + p2 * self.search_space[1] + p3 * self.search_space[2] + p4 * \
                        self.search_space[3]
            bits_out = bits_hard.detach() - bits_soft.detach() + bits_soft
            bits += bits_out
            weighted_bits += bits_out / (
                    self.search_space[0] * p1.detach() + self.search_space[1] * p2.detach() + self.search_space[
                2] * p3.detach() + self.search_space[3] * p4.detach())


        elif len(self.search_space) == 3:
            p1 = p[:, 0]
            p2 = p[:, 1]
            p3 = p[:, 2]
            bits_hard = (flag == 0) * self.search_space[0] + (flag == 1) * self.search_space[1] + (flag == 2) * \
                        self.search_space[2]
            bits_soft = p1 * self.search_space[0] + p2 * self.search_space[1] + p3 * self.search_space[2]
            bits_out = bits_hard.detach() - bits_soft.detach() + bits_soft
            bits += bits_out
            weighted_bits += bits_out / (
                    self.search_space[0] * p1.detach() + self.search_space[1] * p2.detach() + self.search_space[
                2] * p3.detach())


        elif len(self.search_space) == 2:
            p1 = p[:, 0]
            p2 = p[:, 1]
            bits_hard = (flag == 0) * self.search_space[0] + (flag == 1) * self.search_space[1]
            bits_soft = p1 * self.search_space[0] + p2 * self.search_space[1]
            bits_out = bits_hard.detach() - bits_soft.detach() + bits_soft
            bits += bits_out
            weighted_bits += bits_out / (self.search_space[0] * p1.detach() + self.search_space[1] * p2.detach())

        return [grad, bits_out, bits, weighted_bits]


class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout_rate=0.2, k_bits=32):
        super(InitConv, self).__init__()

        self.conv = quant_conv3x3(in_channels, out_channels, k_bits=k_bits, kernel_size=3, padding=1)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout2d(y, self.dropout_rate)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='bn', k_bits=32):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)  
        self.relu1 = nn.ReLU(inplace=True) 
        self.conv1 = quant_conv3x3(in_channels, in_channels, kernel_size=3, padding=1)


        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = quant_conv3x3(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, bits=None,epoch=None):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1, bits,epoch)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y, bits,epoch)
        y = y + x

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels, k_bits=32):
        super(EnDown, self).__init__()
        self.conv = quant_conv3x3(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, bits=None,epoch=None):
        y = self.conv(x, bits,epoch)

        return y


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)


    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


class DeBlock(nn.Module):
    def __init__(self, in_channels, norm='bn'):
        super(DeBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm) 
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)


        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EndConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EndConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        y = self.conv(x)
        return y


class Unet(nn.Module):
    def __init__(self, in_channels=4, base_channels=16, num_classes=4):
        super(Unet, self).__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout_rate=0.2)
        self.select1 = BitSelectorw(base_channels, bias=False, ema_epoch=1)  #
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels * 2)

        self.select2 = BitSelectorw(base_channels * 2, bias=False, ema_epoch=1)
        self.EnBlock2_1 = EnBlock(in_channels=base_channels * 2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels * 2)
        self.EnDown2 = EnDown(in_channels=base_channels * 2, out_channels=base_channels * 4)

        self.select3 = BitSelectorw(base_channels * 4, bias=False, ema_epoch=1)
        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(in_channels=base_channels * 4, out_channels=base_channels * 8)

        self.select4 = BitSelectorw(base_channels * 8, bias=False, ema_epoch=1)
        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

        self.DeUp3 = DeUp_Cat(in_channels=base_channels * 8, out_channels=base_channels * 4)
        self.DeBlock3 = DeBlock(in_channels=base_channels * 4)

        self.DeUp2 = DeUp_Cat(in_channels=base_channels * 4, out_channels=base_channels * 2)
        self.DeBlock2 = DeBlock(in_channels=base_channels * 2)

        self.DeUp1 = DeUp_Cat(in_channels=base_channels * 2, out_channels=base_channels)
        self.DeBlock1 = DeBlock(in_channels=base_channels * 1)

        self.EndConv = EndConv(in_channels=base_channels, out_channels=num_classes)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x,epoch=0):
        image = x
        grads: torch.Tensor = K.filters.spatial_gradient(image, order=1)

        grad = torch.mean(torch.abs(grads.squeeze(1)), (3, 4))  

        grad = grad.reshape(grad.size(0), -1)

        f = None;
        weighted_bits = 0;
        bits = 0

        x = self.InitConv(x)  

        # feature
        grad, bits_out, bits, weighted_bits = self.select1([grad, x, bits, weighted_bits])
        bits_outs = bits_out.unsqueeze(-1)


        x1_1 = self.EnBlock1(x, bits_out,epoch)
        x1_2 = self.EnDown1(x1_1, bits_out,epoch)  # (1, 32, 64, 64, 64)


        grad, bits_out, bits, weighted_bits = self.select2([grad, x1_2, bits, weighted_bits])
        bits_outs = torch.cat((bits_outs,bits_out.unsqueeze(-1)),dim=-1)

        x2_1 = self.EnBlock2_1(x1_2, bits_out,epoch)
        x2_1 = self.EnBlock2_2(x2_1, bits_out,epoch)
        x2_2 = self.EnDown2(x2_1, bits_out,epoch)  # (1, 64, 32, 32, 32)


        grad, bits_out, bits, weighted_bits = self.select3([grad, x2_2, bits, weighted_bits])
        bits_outs = torch.cat((bits_outs,bits_out.unsqueeze(-1)),dim=-1)

        x3_1 = self.EnBlock3_1(x2_2, bits_out,epoch)
        x3_1 = self.EnBlock3_2(x3_1, bits_out,epoch)
        x3_2 = self.EnDown3(x3_1, bits_out,epoch)  # (1, 128, 16, 16, 16)


        grad, bits_out, bits, weighted_bits = self.select4([grad, x3_2, bits, weighted_bits])
        bits_outs = torch.cat((bits_outs,bits_out.unsqueeze(-1)),dim=-1)


        x4_1 = self.EnBlock4_1(x3_2, bits_out,epoch)
        x4_2 = self.EnBlock4_2(x4_1, bits_out,epoch)
        x4_3 = self.EnBlock4_3(x4_2, bits_out,epoch)
        x4 = self.EnBlock4_4(x4_3, bits_out,epoch)  # (1, 128, 16, 16, 16)


        y3_1 = self.DeUp3(x4, x3_1)  # (1, 64, 32, 32, 32)
        y3_2 = self.DeBlock3(y3_1)

        y2_1 = self.DeUp2(y3_2, x2_1)  # (1, 32, 64, 64, 64)
        y2_2 = self.DeBlock2(y2_1)

        y1_1 = self.DeUp1(y2_2, x1_1)  # (1, 16, 128, 128, 128)

        return y1_1, bits, weighted_bits, bits_outs


if __name__ == '__main__':
    with torch.no_grad():
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')

        x = torch.rand((1, 4, 128, 128), device=cuda0)

        model = Unet(in_channels=4, base_channels=16, num_classes=4)
        model.cuda()

        from thop import profile

        flops, params = profile(model, (x,))
        print('flops: ', flops, 'params: ', params)

