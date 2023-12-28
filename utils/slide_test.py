import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
up_kwargs = {'mode': 'bilinear', 'align_corners': True}
H, W, D = 240, 240, 155

def slide_window_2D_only_output(ori_img, crop_size, model):

    stride_rate = 1.0/3.0
    stride = int(crop_size * stride_rate)  # default = 85
    batch, classes, origin_h, origin_w = ori_img.size()

    with torch.cuda.device_of(ori_img):
        outputs = ori_img.new().resize_(batch, classes, origin_h, origin_w).zero_().cuda()
        count_norm = ori_img.new().resize_(batch, 1, origin_h, origin_w).zero_().cuda()

    h_grids = int(math.ceil(1.0 * (origin_h - crop_size) / stride)) + 1
    w_grids = int(math.ceil(1.0 * (origin_w - crop_size) / stride)) + 1

    for idh in range(h_grids):  # 3
        for idw in range(w_grids):
            h0 = idh * stride
            w0 = idw * stride
            h1 = min(h0 + crop_size, origin_h)
            w1 = min(w0 + crop_size, origin_w)

            #adjustment
            if h1 == origin_h:
                h0 = h1 - crop_size
            if w1 == origin_w:
                w0 = w1 - crop_size

            crop_img = crop_image_2D(ori_img, h0, h1, w0, w1).cuda()
            output = model_inference_only_output(model, crop_img).cuda()
            outputs[:, :, h0:h1, w0:w1] += crop_image_2D(output, 0, h1 - h0, 0, w1 - w0)
            count_norm[:, :, h0:h1, w0:w1] += 1
    assert ((count_norm == 0).sum() == 0)
    outputs = outputs / count_norm
    outputs = outputs[:, :, :origin_h, :origin_w]
    outputs = F.softmax(outputs, 1)
    return outputs


def slide_window_2D_out_gflops(ori_img, crop_size, model):

    stride_rate = 1.0/3.0
    stride = int(crop_size * stride_rate)  # default = 85
    batch, classes, origin_h, origin_w = ori_img.size()

    with torch.cuda.device_of(ori_img):
        outputs = ori_img.new().resize_(batch, classes, origin_h, origin_w).zero_().cuda()
        count_norm = ori_img.new().resize_(batch, 1, origin_h, origin_w).zero_().cuda()

    h_grids = int(math.ceil(1.0 * (origin_h - crop_size) / stride)) + 1
    w_grids = int(math.ceil(1.0 * (origin_w - crop_size) / stride)) + 1
    gflops_slice = 0

    for idh in range(h_grids):  # 3
        for idw in range(w_grids):
            h0 = idh * stride
            w0 = idw * stride
            h1 = min(h0 + crop_size, origin_h)
            w1 = min(w0 + crop_size, origin_w)

            #adjustment
            if h1 == origin_h:
                h0 = h1 - crop_size
            if w1 == origin_w:
                w0 = w1 - crop_size

            crop_img = crop_image_2D(ori_img, h0, h1, w0, w1).cuda()

            output, gflops = model_inference_out_gflops(model, crop_img)
            output, gflops = output.cuda(), gflops.cuda()
            outputs[:, :, h0:h1, w0:w1] += crop_image_2D(output, 0, h1 - h0, 0, w1 - w0)
            count_norm[:, :, h0:h1, w0:w1] += 1
            gflops_slice += gflops
    assert ((count_norm == 0).sum() == 0)
    outputs = outputs / count_norm
    outputs = outputs[:, :, :origin_h, :origin_w]
    outputs = F.softmax(outputs, 1)
    return outputs, gflops_slice


def model_inference_out_gflops(model, image):
    output, gflops = model(image, crop = True)
    return output, gflops

def model_inference_only_output(model, image):
    output, gflops = model(image, crop=True)
    return output

def resize_image(img, H, W, D, **up_kwargs):
    return F.interpolate(img, (H, W, D), **up_kwargs)

def crop_image(img, h0, h1, w0, w1, d0, d1):
    return img[:, :, h0:h1, w0:w1, d0:d1]

def crop_image_2D(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]
