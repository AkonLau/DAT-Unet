import argparse
import torch
import numpy as np
import random

import os
import time
from torch.utils.data.sampler import SubsetRandomSampler
from data.loaders import RadioUNetDataset
from data.loaders_pcl_5d import RadioMap5DDataset

from torch.utils.data import DataLoader
from datetime import datetime

from models.unet import UNet
from models.unet_tx import UNet_Tx

from util.tools import SeedContextManager


def split_set(dataset):
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Radio Map Estimation')
    parser.add_argument('--out_path', type=str, default='./experiments-100M-DAT-UNET', help='log folder path')
    parser.add_argument('--model_name', type=str, default='unet', help='Name of the experiment')
    parser.add_argument('--model_type', type=str, default='UNet', help='Name of model name')

    # image size for different dataset
    parser.add_argument('--data_name', type=str, default='sear', help='Name of the dataset: sear, pp5d')
    parser.add_argument('--image_size', type=int, default=128, help='image size for different dataset: 128 or 256')
    parser.add_argument('--simulation', type=str, default='DPM', help='simulation type for RadioMapSear dataset: DPM, IRT2, IRT4')

    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of train batch per domain')
    parser.add_argument('--dataroot', type=str, default="/userhome/data/PP-Datasets", metavar='N',
                        help='data path')
    parser.add_argument('--epochs', type=int, default=50, help='number of epoch to save model')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of test batch per domain')
    parser.add_argument('--sp_ratio', type=int, default=1, help='sample ratio: (%)')

    parser.add_argument('--free_space_only', action='store_true', default=False, help='loss w/o free_space_only')

    parser.add_argument('--visualization', action='store_true', default=False, help='whether visualization the results')

    parser.add_argument('--train_simulation', type=str, default=None,
                        help='simulation type for RadioMapSear dataset: None, DPM, IRT2, IRT4')
    parser.add_argument('--test_simulation', type=str, default=None,
                        help='simulation type for RadioMapSear dataset: None, DPM, IRT2, IRT4')

    # in_channels
    parser.add_argument('--in_channels', type=int, default=2, help='in_channels for different dataset: 1 or 3')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
    print('------------------------------test start---------------------------------')

    if args.train_simulation is None:
        args.train_simulation = args.simulation
    if args.test_simulation is not None:
        args.simulation = args.test_simulation
    print(f'training from {args.train_simulation} to testing on {args.simulation}')

    #     args.out_path = './experiments-0530/pp'
    if args.cuda:
        print('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print('Using CPU')
        raise NotImplementedError

    sp_base_number = 5
    dB_max = 1
    dB_min = 0

    args.out_path = args.out_path + '/' + args.data_name

    if args.data_name == 'pp5d':

        train_dataset = RadioMap5DDataset(args.dataroot, 'train', imsize=args.image_size)
        test_dataset = RadioMap5DDataset(args.dataroot, 'test', imsize=args.image_size, heights=[2])
    elif args.data_name == 'pp5d-h0':

        train_dataset = RadioMap5DDataset(args.dataroot, 'train', imsize=args.image_size, heights=[0])
        test_dataset = RadioMap5DDataset(args.dataroot, 'test', imsize=args.image_size, heights=[0])
    elif args.data_name == 'pp5d-h1':

        train_dataset = RadioMap5DDataset(args.dataroot, 'train', imsize=args.image_size, heights=[1])
        test_dataset = RadioMap5DDataset(args.dataroot, 'test', imsize=args.image_size, heights=[1])
    elif args.data_name == 'pp5d-h2':

        train_dataset = RadioMap5DDataset(args.dataroot, 'train', imsize=args.image_size, heights=[2])
        test_dataset = RadioMap5DDataset(args.dataroot, 'test', imsize=args.image_size, heights=[2])

    elif args.data_name == 'sear':
        if args.in_channels == 3:
            print("using cars images for pp model testing!")
            test_dataset = RadioUNetDataset(args.dataroot, 'test', simulation=args.simulation, carsSimul='yes', carsInput='yes')
        else:
            test_dataset = RadioUNetDataset(args.dataroot, 'test', simulation=args.simulation)
    else:
        raise ValueError

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4, shuffle=False)
    print('test dataset size: {}'.format(len(test_dataset)))

    print('using model {}'.format(args.model_name))
    timestr = datetime.now().strftime('%Y-%m-%d_%I-%M-%S_%p')
    model_name = f'{args.model_name}'
    print(model_name)

    if args.model_name.lower() == "unet":
        model = UNet(model_name,
                     model_type=args.model_type,
                     img_size=args.image_size,
                     in_channels=args.in_channels,
                     ).to(device)

        model_type = args.model_type
    elif args.model_name.lower() == "unet_tx":
        model = UNet_Tx(model_name,
                     model_type=args.model_type,
                     img_size=args.image_size,
                     in_channels=args.in_channels,
                     ).to(device)
        model_type = args.model_type
    else:
        raise ValueError("Invalid model name.")

    print('**** Setup ****')
    print('Total params: %.2fK' % (sum(p.numel() for p in model.parameters()) * 10 ** -3))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) * 10 ** -6))
    print('************')

    # here need to be modified for different datasets and models
    run_dir = f"{sp_base_number * args.sp_ratio}-{sp_base_number * args.sp_ratio} samples, {model_type}"
    if args.model_name.lower() == "unet_tx":
        run_dir = f"tx-only, {model_type}"
    if args.train_simulation != 'DPM' and args.data_name == 'sear':
        run_dir += f', {args.train_simulation}'
    if args.free_space_only:
        run_dir += ', free space only'
    if args.in_channels != 3:
        run_dir += f", {args.in_channels}-in_channels"
    elif args.in_channels == 3 and args.data_name == 'sear':
        run_dir += f", {args.in_channels}-in_channels, cars"
    if args.seed != 1:
        run_dir += f", seed-{args.seed}"

    checkpoint_path = os.path.join(args.out_path, args.model_name, run_dir,
                                       f'{args.epochs} epochs state dict.pth')

    print(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))

    start_time = time.time()
    with SeedContextManager(seed=args.seed):
        test_rmse_loss, ssim_values, psnr_values = model.evaluate(test_loader,
                                        min_samples=sp_base_number * args.sp_ratio,
                                        max_samples=sp_base_number * args.sp_ratio,
                                        dB_max=dB_max,
                                        dB_min=dB_min,
                                        free_space_only=True,
                                        pre_sampled=False,
                                        method=args.itp_method,
                                        args=args,
                                        run_dir=run_dir)

    print('test_rmse_loss: ', test_rmse_loss)

    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    
    t_s = elapsed / len(test_dataset)
    print(f"单个样本耗时：{t_s} 秒")
    print(f"总耗时: {int(h)} 小时 {int(m)} 分钟 {s:.2f} 秒")
    
            
    with open(f"{args.out_path}/{args.model_name}/{run_dir}/result.txt", "a") as f:
        params_logs = 'Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) * 10 ** -6)
        f.write(f"{params_logs}\n")
        f.write(f"{checkpoint_path}\n")

        f.write(f"test_rmse_loss: {test_rmse_loss}\n")
        f.write(f"单个样本耗时：{t_s} 秒\n")
        f.write(f"总耗时: {int(h)} 小时 {int(m)} 分钟 {s:.2f} 秒")
    print('------------------------------test end---------------------------------\n')