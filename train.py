
import argparse
import os
from pickle import FALSE
import random
import logging
from sre_parse import FLAGS
import numpy as np
import time
import math
import setproctitle

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.backup_models.UNet.Med_DANet_V2 import UNet_crop
import torch.distributed as dist
from models import criterions

from data.BraTS import BraTS
# from data.BraTS_2020 import BraTS
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from tensorboardX import SummaryWriter
from torch import nn

import warnings

warnings.filterwarnings('ignore')

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='shr', type=str)

parser.add_argument('--experiment',
                    default='cnn300_50crop',
                    type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='cnn300_50crop'
                            'training on train_0.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='./dataset/BraTS2019/', type=str)
# parser.add_argument('--root', default='./dataset/BraTS2020/', type=str)
parser.add_argument('--train_dir', default='Train', type=str)

parser.add_argument('--valid_dir', default='Train', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--heatmap_dir', default='heatmap', type=str)

parser.add_argument('--test_date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train.txt', type=str)

parser.add_argument('--valid_file', default='valid.txt', type=str)

parser.add_argument('--dataset', default='brats', type=str)

parser.add_argument('--model_name',
                    default='cnn300_50crop',
                    type=str)

parser.add_argument('--input_C', default=4, type=int)

parser.add_argument('--input_H', default=240, type=int)

parser.add_argument('--input_W', default=240, type=int)

parser.add_argument('--input_D', default=160, type=int)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--output_D', default=155, type=int)

# Training Information
parser.add_argument('--lr', default=0.0001, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--seed', default=1085, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='4,5,6,7', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=64, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=350, type=int)

parser.add_argument('--save_freq', default=50, type=int)

parser.add_argument('--load_file', default='', type=str)


parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()


def dice_score(o, t, eps=1e-8):
    num = 2 * (o * t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num / den


def softmax_output_dice(output, target):
    ret = []

    # WT
    o = output > 0
    t = target > 0  # ce
    ret += dice_score(o, t),
    # TC
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # ET
    o = (output == 3)
    t = (target == 4)
    ret += dice_score(o, t),

    return ret


def main_worker():
    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.distributed.init_process_group('nccl')  # 初始化GPU通信方式NCCL, PyTorch实现分布式运算是通过NCCL进行显卡通信�?
    torch.cuda.set_device(args.local_rank)  # 为这个进程指定GPU

    model = UNet_crop(input_channels=4, num_classes=4, mode_train=True)

    model.cuda(args.local_rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)

    if args.start_epoch > 0:     
        load_file = args.load_file
        if os.path.exists(load_file):
            checkpoint = torch.load(load_file, map_location=lambda storage, loc: storage)

            model.load_state_dict(checkpoint['state_dict'])
            print('Successfully loading checkpoint of epoch: {} and training from epoch: {}'
                        .format(checkpoint['epoch'], args.start_epoch))           
        else:
            print('There is no checkpoint file to load!')

    model.train()
    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters() if "decision_network" not in n and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "decision_network" in n and p.requires_grad],
            "lr": args.lr * 10,
        }
    ]
    
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    criterion = getattr(criterions, args.criterion)

    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                      args.experiment + args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        train_label_for_DN = f"./traininglabels_DN/{args.experiment + args.date}"
        train_label_for_DN_HGG = f"{train_label_for_DN}/HGG"
        train_label_for_DN_LGG = f"{train_label_for_DN}/LGG"
        if not os.path.exists(train_label_for_DN):
            os.makedirs(train_label_for_DN)
        if not os.path.exists(train_label_for_DN_HGG):
            os.makedirs(train_label_for_DN_HGG)
        if not os.path.exists(train_label_for_DN_LGG):
            os.makedirs(train_label_for_DN_LGG)

    if args.local_rank == 0:
        writer = SummaryWriter()
        
    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)

    train_set = BraTS(train_list, train_root, args.mode)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    logging.info('Samples for train = {}'.format(len(train_set)))

    num_gpu = (len(args.gpu) + 1) // 2

    num_iter_perepoch = len(train_set) // args.batch_size
    num_iter_perepoch = num_iter_perepoch * int(128)

    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)

    start_time_training = time.time()

    torch.set_grad_enabled(True)

    fix = 0
    train_epoch = [150, 300]
    scale_lr = 10
    for epoch in range(args.start_epoch, args.end_epoch):
        train_sampler.set_epoch(epoch)  # shuffle
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        start_epoch = time.time()

        for i, data in enumerate(train_loader):

            x, target = data
            x = x.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)

            # shuffle batchsize & slice dimension
            max_number = (args.batch_size // num_gpu) * 128
            index = torch.randperm(max_number)
            B, D = x.size(0), x.size(-1)
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(B * D, 4, 128, 128)
            target = target.permute(0, 3, 1, 2).contiguous().view(B * D, 128, 128)
            x = x[index]
            target = target[index]

            if epoch < train_epoch[0]:

                for s in range(128):
                    current_iter = epoch * num_iter_perepoch + i * int(128) + (s + 1)
                    warm_up_learning_rate_adjust_iter(args.lr, current_iter, num_iter_perepoch,
                                                      args.end_epoch * num_iter_perepoch, optimizer, power=0.9)

                    x_s = x[s * (args.batch_size // num_gpu):(s + 1) * (args.batch_size // num_gpu) - 1, ...]

                    crop1_output, whole_output = model(x_s, crop=True, decide=False, quantify_loss = False,epoch= epoch)

                    loss_crop1, loss1_crop1, loss2_crop1, loss3_crop1 = criterion(crop1_output, target[s * (
                                args.batch_size // num_gpu):(s + 1) * (args.batch_size // num_gpu) - 1, ...])
                    loss_whole, loss1_whole, loss2_whole, loss3_whole = criterion(whole_output, target[
                                    s * (args.batch_size // num_gpu):(s + 1) * (args.batch_size // num_gpu) - 1,...])

                    loss = (loss_crop1 + loss_whole)/2
                    loss1 = (loss1_crop1 + loss1_whole)/2
                    loss2 = (loss2_crop1 + loss2_whole)/2
                    loss3 = (loss3_crop1 + loss3_whole)/2

                    reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
                    reduce_loss1 = all_reduce_tensor(loss1, world_size=num_gpu).data.cpu().numpy()
                    reduce_loss2 = all_reduce_tensor(loss2, world_size=num_gpu).data.cpu().numpy()
                    reduce_loss3 = all_reduce_tensor(loss3, world_size=num_gpu).data.cpu().numpy()

                    if args.local_rank == 0:
                        logging.info(
                            'Epoch: {}_Iter:{}_Slice:{}  loss: {:.5f} || 1:{:.4f} | 2:{:.4f} | 3:{:.4f} ||'
                            .format(epoch, i, s, reduce_loss, reduce_loss1, reduce_loss2, reduce_loss3))
                        logging.info(
                            'crop1 predicition: NET:{:.4f} | ED:{:.4f} | ET:{:.4f}'.format(loss1_crop1, loss2_crop1,
                                                                                           loss3_crop1))
                        logging.info(
                            'whole predicition: NET:{:.4f} | ED:{:.4f} | ET:{:.4f}'.format(loss1_whole, loss2_whole,
                                                                                           loss3_whole))

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

            if epoch < train_epoch[1] and epoch >= train_epoch[0]:

                for s in range(128):
                    current_iter = epoch * num_iter_perepoch + i * int(128) + (s + 1)
                    warm_up_learning_rate_adjust_iter(args.lr, current_iter, num_iter_perepoch,
                                                      args.end_epoch * num_iter_perepoch, optimizer, power=0.9)


                    x_s = x[s * (args.batch_size // num_gpu):(s + 1) * (args.batch_size // num_gpu) - 1, ...]


                    crop1_output, whole_output, GFLOPs_output = model(x_s, crop=True, decide=False, quantify_loss = True,epoch= epoch)

                    loss_crop1, loss1_crop1, loss2_crop1, loss3_crop1 = criterion(crop1_output, target[s * (
                                args.batch_size // num_gpu):(s + 1) * (args.batch_size // num_gpu) - 1, ...])
                    loss_whole, loss1_whole, loss2_whole, loss3_whole = criterion(whole_output, target[
                                    s * (args.batch_size // num_gpu):(s + 1) * (args.batch_size // num_gpu) - 1,...])

                    loss = (loss_crop1 + loss_whole)/2 + 0.00015 * GFLOPs_output.sum() 
                    loss1 = (loss1_crop1 + loss1_whole)/2
                    loss2 = (loss2_crop1 + loss2_whole)/2
                    loss3 = (loss3_crop1 + loss3_whole)/2

                    reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
                    reduce_loss1 = all_reduce_tensor(loss1, world_size=num_gpu).data.cpu().numpy()
                    reduce_loss2 = all_reduce_tensor(loss2, world_size=num_gpu).data.cpu().numpy()
                    reduce_loss3 = all_reduce_tensor(loss3, world_size=num_gpu).data.cpu().numpy()

                    if args.local_rank == 0:
                        logging.info(
                            'Epoch: {}_Iter:{}_Slice:{}  loss: {:.5f} || 1:{:.4f} | 2:{:.4f} | 3:{:.4f} ||'
                            .format(epoch, i, s, reduce_loss, reduce_loss1, reduce_loss2, reduce_loss3))
                        logging.info(
                            'crop1 predicition: NET:{:.4f} | ED:{:.4f} | ET:{:.4f}'.format(loss1_crop1, loss2_crop1,
                                                                                           loss3_crop1))
                        logging.info(
                            'whole predicition: NET:{:.4f} | ED:{:.4f} | ET:{:.4f}'.format(loss1_whole, loss2_whole,
                                                                                           loss3_whole))

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()
            if epoch >= train_epoch[1]:
                if fix == 0:
                    for name, parameter in model.named_parameters():
                        if "decision_network" not in name:
                            parameter.requires_grad = False

                    fix = 1

                epoch_start_iter = train_epoch[1] * num_iter_perepoch
                for s in range(128):
                    current_iter = epoch * num_iter_perepoch + i * int(128) + (s + 1)
                    warm_up_learning_rate_adjust_iter2(args.lr, current_iter, num_iter_perepoch,
                                                      args.end_epoch * num_iter_perepoch, optimizer, scale_lr, epoch_start_iter, power=0.9)


                    x_s = x[s * (args.batch_size // num_gpu):(s + 1) * (args.batch_size // num_gpu) - 1, ...]

                    crop1_output, whole_output, decision_output, GFLOPs_output, choice = model(x_s, crop=True,decide=True, quantify_loss=True,epoch= epoch)

                    loss_crop1, loss1_crop1, loss2_crop1, loss3_crop1 = criterion(crop1_output, target[s * (
                                args.batch_size // num_gpu):(s + 1) * (args.batch_size // num_gpu) - 1, ...])
                    
                    loss_whole, loss1_whole, loss2_whole, loss3_whole = criterion(whole_output, target[
                                    s * (args.batch_size // num_gpu):(s + 1) * (args.batch_size // num_gpu) - 1,...])


                    loss_decision, loss1_decision, loss2_decision, loss3_decision = criterions.softmax_dice3(decision_output, GFLOPs_output, target[s * (
                                args.batch_size // num_gpu):(s + 1) * (args.batch_size // num_gpu) - 1, ...])

                    loss = (loss_crop1 + loss_whole + 2 * loss_decision) / 4 

                    loss1 = (loss1_crop1 + loss1_whole + 2 * loss1_decision) / 4
                    loss2 = (loss2_crop1 + loss2_whole + 2 * loss2_decision) / 4
                    loss3 = (loss3_crop1 + loss3_whole + 2 * loss3_decision) / 4

                    reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
                    reduce_loss1 = all_reduce_tensor(loss1, world_size=num_gpu).data.cpu().numpy()
                    reduce_loss2 = all_reduce_tensor(loss2, world_size=num_gpu).data.cpu().numpy()
                    reduce_loss3 = all_reduce_tensor(loss3, world_size=num_gpu).data.cpu().numpy()

                    if args.local_rank == 0:

                        logging.info('Epoch: {}_Iter:{}_Slice:{}  loss: {:.5f}  decision_loss:{:.5f} || 1:{:.4f} | 2:{:.4f} | 3:{:.4f} ||'
                                     .format(epoch, i, s, reduce_loss, loss_decision, reduce_loss1, reduce_loss2, reduce_loss3))
                        logging.info('crop1 predicition: NET:{:.4f} | ED:{:.4f} | ET:{:.4f}'.format(loss1_crop1, loss2_crop1, loss3_crop1))
                        logging.info('whole predicition: NET:{:.4f} | ED:{:.4f} | ET:{:.4f}'.format(loss1_whole, loss2_whole, loss3_whole))
                        logging.info('decision predicition: NET:{:.4f} | ED:{:.4f} | ET:{:.4f}'.format(loss1_decision, loss2_decision, loss3_decision))
                        logging.info('choice: {}'.format(choice))

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

        end_epoch = time.time()

        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 3) == 0 \
                    or epoch == (train_epoch[1] - 1):
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

            writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('loss:', reduce_loss, epoch)
            writer.add_scalar('loss1:', reduce_loss1, epoch)
            writer.add_scalar('loss2:', reduce_loss2, epoch)
            writer.add_scalar('loss3:', reduce_loss3, epoch)

        if args.local_rank == 0:
            epoch_time_minute = (end_epoch - start_epoch) / 60
            remaining_time_hour = (args.end_epoch - epoch - 1) * epoch_time_minute / 60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    if args.local_rank == 0:
        writer.close()

        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)

    end_time = time.time()
    training_total_time = (end_time - start_time_training) / 3600
    logging.info('The total training time is {:.2f} hours'.format(training_total_time))
    logging.info('----------------------------------The training process finished!-----------------------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)


def warm_up_learning_rate_adjust1(init_lr, epoch, warm_epoch, max_epoch, optimizer):
    for param_group in optimizer.param_groups:
        if epoch < warm_epoch:
            param_group['lr'] = init_lr * (epoch + 1) / (warm_epoch + 1)
        else:
            param_group['lr'] = init_lr * (math.cos(math.pi * (epoch - warm_epoch) / max_epoch) + 1) / 2


def warm_up_learning_rate_adjust2(init_lr, epoch, warm_epoch, max_epoch, optimizer):
    for param_group in optimizer.param_groups:
        if epoch < warm_epoch:
            param_group['lr'] = init_lr * (1 - math.cos(math.pi / 2 * (epoch + 1) / (warm_epoch)))
        else:
            param_group['lr'] = init_lr * (math.cos(math.pi * (epoch - warm_epoch) / max_epoch) + 1) / 2


def warm_up_learning_rate_adjust_iter(init_lr, cur_iter, warmup_iter, max_iter, optimizer, power=0.9):
    for param_group in optimizer.param_groups:
        if cur_iter < warmup_iter:
            param_group['lr'] = init_lr * cur_iter / (warmup_iter + 1e-8)
        else:
            param_group['lr'] = init_lr * ((1 - float(cur_iter - warmup_iter) / (max_iter - warmup_iter)) ** (power))


def warm_up_learning_rate_adjust_iter2(init_lr, cur_iter, warmup_iter, max_iter, optimizer, scale_lr, epoch_start_iter, power=0.9):
    i = 0
    for param_group in optimizer.param_groups:
        if i == 0:
            if cur_iter < warmup_iter:
                param_group['lr'] = init_lr * cur_iter / (warmup_iter + 1e-8)
            else:
                param_group['lr'] = init_lr * ((1 - float(cur_iter - warmup_iter) / (max_iter - warmup_iter)) ** (power))
            i += 1
        else: 
            if (cur_iter - epoch_start_iter) < warmup_iter:
                param_group['lr'] = init_lr * (cur_iter - epoch_start_iter) / (warmup_iter + 1e-8) * scale_lr
            else:
                param_group['lr'] = init_lr * ((1 - float((cur_iter - epoch_start_iter) - warmup_iter) / ((max_iter - epoch_start_iter) - warmup_iter)) ** (power))  * scale_lr


def warm_up_learning_rate_adjust_iter_cosine(init_lr, cur_iter, warmup_iter, max_iter, optimizer):
    for param_group in optimizer.param_groups:
        if cur_iter < warmup_iter:
            param_group['lr'] = init_lr * (1 - math.cos(math.pi / 2 * (cur_iter) / (warmup_iter)))
        else:
            param_group['lr'] = init_lr * (math.cos(math.pi * (cur_iter - warmup_iter) / max_iter) + 1) / 2



def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
