import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import nibabel as nib
import imageio
import scipy.misc
import SimpleITK as sitk
from utils.slide_test import slide_window_2D_only_output, slide_window_2D_out_gflops

cudnn.benchmark = True


def one_hot(ori, classes):

    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in range(classes):
        index_list = (ori == j).nonzero()
        for i in range(len(index_list)):
            batch, height, width, depth = index_list[i]
            new_gd[batch, j, height, width, depth] = 1

    return new_gd.float()


def tailor_and_concat(x, model):

    temp = []

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    y = x.clone()

    for i in range(len(temp)):
        # temp[i] = model(temp[i])
        temp[i] = model(temp[i])
    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return y[..., :155]


def dice_score(o, t, eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den


def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==4)))
    return mIOU_score


def softmax_output_dice(output, target):
    ret = []

    # whole
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # active
    o = (output == 3);t = (target == 4)
    ret += dice_score(o, t),

    return ret


keys = 'whole', 'core', 'enhancing', 'loss'


def validate_softmax(
        valid_loader,
        model,
        heatmap_use=True,
        heatmap_dir='',
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly!
        verbose=False,
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        visual='',  # the path to save visualization
        postprocess=False,  # Default False, when use postprocess, the score of dice_ET would be changed.
        valid_in_train=False,  # if you are valid when train
        ):

    H, W, T = 240, 240, 160
    model.eval()
    WT_LIST, TC_LIST, ET_LIST, flops_sample_list = [], [], [], []
    runtimes = []
    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        if valid_in_train:
            target_cpu = data[1][0, :H, :W, :T].numpy()
            data = [t.cuda(non_blocking=True) for t in data]
            x, target = data[:2]
        else:
            x = data
            x.cuda()
        flops_sample = 0

        torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
        start_time = time.time()

        x = x[..., :155]
        output = x.clone().cpu().detach().numpy()
        print('start to predict segmentation!!')

        for s in range(155):

            x_s = x[..., s].cuda()
            x_origin = x_s
            logit, gflops_slice = slide_window_2D_out_gflops(x_s, crop_size=128, model=model)  # no flip
            logit += slide_window_2D_only_output(x_s.flip(dims=(2,)), crop_size=128, model=model).flip(dims=(2,))  # flip H
            logit += slide_window_2D_only_output(x_s.flip(dims=(3,)), crop_size=128, model=model).flip(dims=(3,))  # flip W
            logit += slide_window_2D_only_output(x_s.flip(dims=(2, 3)), crop_size=128, model=model).flip(dims=(2, 3))  # flip H, W
            output1 = logit / 4.0
            output1 = output1.cpu().numpy()
            flops_sample += gflops_slice

            output[..., s] = output1


        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
        runtimes.append(elapsed_time)

        print(f'flops_sample{i} is: {flops_sample}')
        flops_sample_list.append(flops_sample)
        print('-------------------------------------------------------------------------------')

        output = output[0, :, :H, :W, :T]
        output = output.argmax(0)

        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        print(msg)

        if savepath:
            # .npy for further model ensemble
            # .nii for directly model submission
            assert save_format in ['npy', 'nii']
            if save_format == 'npy':
                np.save(os.path.join(savepath, name + '_preds'), output)
            if save_format == 'nii':
                # raise NotImplementedError
                oname = os.path.join(savepath, name + '.nii.gz')
                seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)

                seg_img[np.where(output == 1)] = 1
                seg_img[np.where(output == 2)] = 2
                seg_img[np.where(output == 3)] = 4

                if verbose:
                    print('1:', np.sum(seg_img == 1), ' | 2:', np.sum(seg_img == 2), ' | 4:', np.sum(seg_img == 4))
                    print('WT:', np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4)), ' | TC:',
                          np.sum((seg_img == 1) | (seg_img == 4)), ' | ET:', np.sum(seg_img == 4))

                # #BraTS2019 evaluation
                nib.save(nib.Nifti1Image(seg_img, None), oname)
                print('Successfully save {}'.format(oname))

                ##BraTS2020 evaluation
                #Read an submission file generated by the open_brats for the necessary reference information
                # c0 = sitk.ReadImage('./dataset/BraTS2020/BraTS20_Validation_002.nii.gz')
                # Direction = c0.GetDirection()
                # Origin = c0.GetOrigin()
                # Spacing = c0.GetSpacing()
                # seg_img = sitk.GetImageFromArray(seg_img.transpose(2, 1, 0))
                # seg_img.SetOrigin(Origin)
                # seg_img.SetSpacing(Spacing)
                # seg_img.SetDirection(Direction)
                # sitk.WriteImage(seg_img, f"{oname}.nii.gz")
                # print('Successfully save {}'.format(oname))

                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
                    Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
                    Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
                    Snapshot_img[:, :, 2, :][np.where(output == 3)] = 255

                    for frame in range(T):
                        if not os.path.exists(os.path.join(visual, name)):
                            os.makedirs(os.path.join(visual, name))
                        imageio.imwrite(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
        if heatmap_use:
            if not os.path.exists(os.path.join(heatmap_dir, name)):
                os.makedirs(os.path.join(heatmap_dir, name))

            image = x.cpu().numpy().squeeze()[0, ...]   # (H, W, D)
            print('image:', image.shape)
            image = image * 256 / image.max()

            attention = torch.argmax(attention, dim=1)  # (B, H, W, D)
            attention = one_hot(attention, classes=4).float()  # (B, Cl, H, W, D)
            attention = torch.nn.functional.interpolate(attention, scale_factor=8, mode='trilinear', align_corners=False)
            attention = attention.cpu().detach().numpy().squeeze()[1, ...]
            heatmap = attention / np.max(attention)       # (H, W, D)
            heatmap = np.uint8(heatmap * 256)
            print('heatmap:', heatmap.shape)

            for slice in range(0, heatmap.shape[-1]):
                image_slice = image[..., slice]
                image_slice = np.expand_dims(image_slice, axis=2)
                image_slice = image_slice.repeat(3, axis=2)

                heatmap_slice = heatmap[..., slice]       # (H, W)
                heatmap_slice = cv2.applyColorMap(heatmap_slice, cv2.COLORMAP_JET)
                heatmap_mix = heatmap_slice * 0.9 + image_slice
                heatmap_tri = np.hstack((heatmap_slice, heatmap_mix, image_slice))

                cv2.imwrite(os.path.join(heatmap_dir, name, str(slice) + '.png'), heatmap_tri)
   
    print('-------------------------------------------------------------------------------')
    print(f'Med-DANetV2 over all flops for all samples: {sum(flops_sample_list)} GFLOPs')
    print(f'Med-DANetV2 mean flops per sample:  {sum(flops_sample_list)/len(flops_sample_list)} GFLOPs')
    print('-------------------------------------------------------------------------------')
    print('runtimes:', sum(runtimes)/len(runtimes))
    print('-------------------------------------------------------------------------------')
    
def test_flip(x, model, only_output=True):
    x = x[..., :155]
    inter_output = x.clone()
    inter_output = inter_output[:, :, :240, :240, :155]
    for s in range(155):
        x_s = x[..., s].cuda()
        sw1 = slide_window_2D_only_output(x_s, crop_size=128, model=model)
        sw1 = sw1[0, :, :240, :240]  # (B, C, H, W) -> (C, H, W)
        inter_output[0, :, :240, :240, s] = sw1
    return inter_output