import argparse
import os
import time
import random
import numpy as np
import setproctitle

from models.backup_models.UNet.Med_DANet_V2 import UNet_crop
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from utils.predict import validate_softmax
cudnn.benchmark = True

from data.BraTS import BraTS
# from data.BraTS_2020 import BraTS

parser = argparse.ArgumentParser()

parser.add_argument('--user', default='shr', type=str)

parser.add_argument('--root', default='./dataset/BraTS2019/', type=str)
# parser.add_argument('--root', default='./dataset/BraTS2020/', type=str)
parser.add_argument('--valid_dir', default='Valid', type=str)

parser.add_argument('--valid_file', default='valid.txt', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--heatmap_dir', default='heatmap', type=str)

parser.add_argument('--experiment', default='multiscale_inference_MedDANet_V2_brast2019', type=str)

parser.add_argument('--test_date', default='2023-04-24', type=str)

parser.add_argument('--post_process', default=True, type=bool)

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

parser.add_argument('--crop_H', default=240, type=int)

parser.add_argument('--crop_W', default=240, type=int)

parser.add_argument('--crop_D', default=155, type=int)

parser.add_argument('--seed', default=1024, type=int)

parser.add_argument('--load_dir', default='./checkpoint/BraTS2019/model_epoch_last.pth', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='1', type=str)

parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)  
    random.seed(
        args.seed)  
    np.random.seed(args.seed)

    model = UNet_crop(input_channels=4, num_classes=4, mode_train=False, decision_net_predict=True) 

    decision_load_file = args.load_dir
   
    model = torch.nn.DataParallel(model).cuda()
    if os.path.exists(decision_load_file):
        checkpoint = torch.load(decision_load_file, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])
        print('Successfully load checkpoint of both segmentation network and decision network!')
    else:
        print('There is no checkpoint file to load!')

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_list, valid_root, mode='test')
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    submission = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                              args.submission, args.experiment + args.test_date)
    visual = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                          args.visual, args.experiment + args.test_date)
    heatmap = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                           args.heatmap_dir, args.experiment + args.test_date)
    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)
    if not os.path.exists(heatmap):
        os.makedirs(heatmap)

    start_time = time.time()


    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=model,
                         heatmap_use=False,
                         savepath=submission,
                         visual=visual,
                         heatmap_dir=heatmap,
                         names=valid_set.names,
                         save_format=args.save_format,
                         snapshot=True,
                         postprocess=False
                         )

    end_time = time.time()
    full_test_time = (end_time - start_time) / 60
    average_time = full_test_time / len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    setproctitle.setproctitle('{}: Testing_single'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()



