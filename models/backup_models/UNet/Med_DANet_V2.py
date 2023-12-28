import torch
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
from models.backup_models.UNet.quantify_UNet import Unet
from models.Decision_Networks.Shuffle_netV2 import ShuffleNetV2
from torch import nn
import torch.nn.functional as F
import numpy as np
import random

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

def inverse_gumbel_cdf(y, mu, beta):
    return mu - beta * np.log(-np.log(y))


def gumbel_softmax_sampling(h, mu=0, beta=1, tau=1):
    """
    h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
    """
    shape_h = h.shape
    p = F.softmax(h, dim=1)
    y = torch.rand(shape_h) + 1e-10  # ensure all y is positive.
    g = inverse_gumbel_cdf(y, mu, beta).to(h.device)
    x = torch.log(p) + g  # samples follow Gumbel distribution.
    x = x / tau
    x = F.softmax(x, dim=1)  # now, the x approximates a one_hot vector.
    return x

def get_patches_frame(input_frames, actions, patch_size, image_size):
    theta = torch.zeros(input_frames.size(0), 2, 3).cuda()
    patch_coordinate = (actions * (image_size - patch_size))
    x1, x2, y1, y2 = patch_coordinate[:, 1], patch_coordinate[:, 1] + patch_size, \
                     patch_coordinate[:, 0], patch_coordinate[:, 0] + patch_size

    theta[:, 0, 0], theta[:, 1, 1] = patch_size / image_size, patch_size / image_size
    theta[:, 0, 2], theta[:, 1, 2] = -1 + (x1 + x2) / image_size, -1 + (y1 + y2) / image_size
    grid = F.affine_grid(theta.float(), torch.Size((input_frames.size(0), 4, patch_size, patch_size)),align_corners=False)
    patches = F.grid_sample(input_frames, grid)  # [NT,C,H1,W1]
    return patches

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=1, stride=1):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride,
                                                  padding=(kernel_size - stride) // 2, groups=inplanes, bias=False),
                                        nn.BatchNorm2d(outplanes))
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.conv_block(x)+x)

class lastdecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.DeBlock1 = DeBlock(in_channels=in_channels)
        self.EndConv = EndConv(in_channels=in_channels, out_channels=out_channels)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y1_2 = self.DeBlock1(x)
        y = self.EndConv(y1_2)
        y = self.Softmax(y)
        return y       

class UNet_crop(nn.Module):
    def __init__(self, input_channels, num_classes, mode_train=True, decision_net_predict=False, **kwargs):
        super().__init__()

        self.mode_train = mode_train
        self.decision_net_predict = decision_net_predict

        self.decision_network = ShuffleNetV2(channels=4, input_size=128, num_class=3, model_size='0.5x')
        self.actions_network = ShuffleNetV2(channels=4, input_size=128, num_class=2, model_size='0.5x')
        self.CNN = Unet(in_channels=input_channels, base_channels=16, num_classes=num_classes)

        self.conv1 = lastdecoder(in_channels=16, out_channels=num_classes)
        self.conv2 = lastdecoder(in_channels=16, out_channels=num_classes)

        self.sig = nn.Sigmoid()
        self.crop_size = 96

    def forward(self, x, crop=True, random_train=False, decide = False, quantify_loss = False,epoch=None):

        if self.mode_train:

            if crop == False:

                output, wholeoutbit, weighted_bits, bits_outs = self.CNN(x,epoch)
                output = self.conv2(output) 
                return output
            
            if crop:
                if random_train:
                    crop_index = random.randint(0,1)
                    if crop_index == 0:
                        crop_size = self.crop_size
                        actions = self.actions_network(x)
                        actions = self.sig(actions)

                        crop_input = get_patches_frame(x, actions, crop_size, 128).detach()
                        crop, wholeoutbit, weighted_bits, bits_outs = self.CNN(crop_input,epoch)
                        # b, c, h, w = x.shape
                        b, _, h, w = x.shape
                        c = crop.size(1)
                        crop_output = torch.zeros((b, c, h, w), dtype=x.dtype, device=x.device)
                        crop_output[:, :, 0:crop_size, 0:crop_size] = crop
                        theta = torch.zeros(b, 2, 3)
                        identity = torch.eye(2).reshape(1, 2, 2).repeat(b, 1, 1)
                        theta[:, 0:2, 0:2] = identity
                        actions = actions.reshape(x.size(0), 2, 1)
                        offset = actions * (h - crop_size)
                        theta[:, 0:2, 2:3] = - 2 * offset / h
                        grid = F.affine_grid(theta.float(), torch.Size((b, c, h, w)), align_corners=False).cuda()
                        crop_output = F.grid_sample(crop_output, grid)
                        crop_output = self.conv1(crop_output)
                        return crop_output
                    if crop_index == 1:
                        output, wholeoutbit, weighted_bits, bits_outs = self.CNN(x,epoch)
                        output = self.conv2(output) 
                        return output

                if  not random_train:
                    crop_size = self.crop_size
                    actions = self.actions_network(x)    
                    actions = self.sig(actions)
                    crop1_input = get_patches_frame(x, actions, crop_size, 128).detach()
                    crop1, crop1_wholeoutbit, crop1_weighted_bits, crop1_bits_outs = self.CNN(crop1_input,epoch)
                    whole_output, whole_wholeoutbit, whole_weighted_bits, whole_bits_outs = self.CNN(x,epoch)
                    b, _, h, w = x.shape
                    c = crop1.size(1)

                    # b, c, h, w = x.shape
                    crop1_output = torch.zeros((b, c, h, w), dtype=x.dtype, device=x.device)
                    crop1_output[:, :, 0:crop_size, 0:crop_size] = crop1
                    theta1 = torch.zeros(b, 2, 3)
                    identity = torch.eye(2).reshape(1, 2, 2).repeat(b, 1, 1)
                    theta1[:, 0:2, 0:2] = identity
                    actions = actions.reshape(x.size(0), 2, 1)
                    offset1 = actions * (h - crop_size)
                    theta1[:, 0:2, 2:3] = - 2 * offset1 / h
                    grid1 = F.affine_grid(theta1.float(), torch.Size((b, c, h, w)), align_corners=False).cuda()
                    crop1_output = F.grid_sample(crop1_output, grid1)
                    crop1_output = self.conv1(crop1_output)
                    whole_output = self.conv2(whole_output)
                    if decide == False:
                        if quantify_loss == True:

                            bits_weights = torch.ones((b,10), dtype=x.dtype, device=x.device)
                            bits_weights[:, 0:4] = crop1_bits_outs / 32
                            bits_weights[:, 5:9] = whole_bits_outs / 32
                            GFLOPs = torch.FloatTensor([53.085,95.551,95.553,169.871,185.94,94.371,169.87,169.868,314.644,319.247]).reshape(10, 1).to(x.device)
                            GFLOPs_output = torch.mm(bits_weights, GFLOPs).reshape(b,1) / 1000
                            return crop1_output, whole_output, GFLOPs_output
                        if quantify_loss == False:
                            return crop1_output, whole_output
                    else:
                        choice_max = self.decision_network(x)
                        choice = gumbel_softmax_sampling(choice_max)
                        shape = choice.size()
                        _, ind = choice.max(dim=-1)
                        choice_hard = torch.zeros_like(choice).view(-1, shape[-1])
                        choice_hard.scatter_(1, ind.view(-1, 1), 1)
                        choice_hard = choice_hard.view(*shape)

                        choice_hard = (choice_hard - choice).detach() + choice
                        bits_weights = torch.ones((b,10), dtype=x.dtype, device=x.device)
                        bits_weights[:, 0:4] = crop1_bits_outs / 32
                        bits_weights[:, 5:9] = whole_bits_outs / 32
                        GFLOPs = torch.FloatTensor([53.085,95.551,95.553,169.871,185.94,94.371,169.87,169.868,314.644,319.247]).reshape(10, 1).to(x.device)
                        crop1_GFLOPs = torch.mm(bits_weights[:, 0:5], GFLOPs[0:5, :]).reshape(b,1) / 1000
                        whole_GFLOPs = torch.mm(bits_weights[:, 5:10], GFLOPs[5:10, :]).reshape(b,1) / 1000
                        GFLOPS = torch.zeros((b,1), dtype=x.dtype, device=x.device)
                        GFLOPs = torch.cat((GFLOPS, crop1_GFLOPs, whole_GFLOPs),dim=1)
                        GFLOPs_output = (choice_hard * GFLOPs).sum(-1).reshape(x.size(0),1)
                        decision_output = torch.zeros((b, 4, 128, 128), dtype=x.dtype, device=crop1_output.device)
                        decision_output = torch.stack((decision_output, crop1_output, whole_output), 0).permute(1,0,2,3,4)
                        choice_hard = choice_hard.reshape(b,3,1,1,1).repeat(1,1,4,h,w)
                        decision_output = torch.mul(choice_hard, decision_output).sum(1)

                        return crop1_output, whole_output, decision_output, GFLOPs_output, choice_max.argmax(1)



        else:
            if self.decision_net_predict:
                    choice = self.decision_network(x).argmax(1)
                    if choice == 0:
                        output = torch.zeros((x.size(0), 4, 128, 128), dtype=x.dtype)
                        GFLOPs_output = torch.zeros(1,1).sum()
                    elif choice == 2:
                        output, wholeoutbit, weighted_bits, bits_outs = self.CNN(x,epoch)
                        output = self.conv2(output)

                        bits_weights = torch.ones((1,5), dtype=x.dtype, device=x.device)
                        bits_weights[:, 0:4] = bits_outs / 32
                        whole_GFLOPs = torch.FloatTensor([94.371,169.87,169.868,314.644,319.247]).reshape(5, 1).to(x.device)
                        GFLOPs_output = torch.mm(bits_weights, whole_GFLOPs).sum() / 1000
                    else:
                        crop_size = self.crop_size

                        actions = self.actions_network(x)
                        actions = self.sig(actions)
                        b, _, h, w = x.shape
                        x = get_patches_frame(x, actions, crop_size, 128)
                        output_crop, wholeoutbit, weighted_bits, bits_outs = self.CNN(x,epoch)
                        c = output_crop.size(1)
                        output = torch.zeros((b, c, h, w), dtype=output_crop.dtype, device=output_crop.device)
                        output[:, :, 0:crop_size, 0:crop_size] = output_crop
                        theta = torch.zeros(1, 2, 3)
                        identity = torch.eye(2).reshape(1, 2, 2)
                        theta[:, 0:2, 0:2] = identity
                        theta[:, 0:2, 2:3] = - (actions.reshape(1, 2, 1) * (h - crop_size)) * 2 / h
                        grid = F.affine_grid(theta.float(), torch.Size((1, c, h, w)), align_corners=False).cuda()
                        output = F.grid_sample(output, grid)
                        output = self.conv1(output)

                        bits_weights = torch.ones((1,5), dtype=x.dtype, device=x.device)
                        bits_weights[:, 0:4] = bits_outs / 32
                        crop1_GFLOPs = torch.FloatTensor([53.085,95.551,95.553,169.871,185.94]).reshape(5, 1).to(x.device)

                        GFLOPs_output = torch.mm(bits_weights, crop1_GFLOPs).sum() / 1000
            return output, GFLOPs_output


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128), device=cuda0)
        model = UNet_crop(input_channels=4, num_classes=4, mode_train=True)
        model.cuda()
        output = model(x)
        print('output:', output.shape)
        flop = FlopCountAnalysis(model, x)
        print(flop_count_table(flop, max_depth=4))
        print(flop_count_str(flop))
        print(flop.total())



