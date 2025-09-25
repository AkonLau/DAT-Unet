import argparse
import torch
import numpy as np
import random

from torch.utils.data.sampler import SubsetRandomSampler
from data.loaders import RadioUNetDataset
from data.loaders_pcl_5d import RadioMap5DDataset

from torch.utils.data import DataLoader
from datetime import datetime

from models.unet import UNet
from models.unet_tx import UNet_Tx

import os

# if you want to use wandb, please set environement variables for storing wandb outputs
os.environ['WANDB_API_KEY'] = '***' # the wandb api key for syncing
# os.environ["WANDB_MODE"] = "online" # set to online to sync to wandb
os.environ["WANDB_MODE"] = "offline" # set to offline to prevent syncing to wandb


def split_set(dataset):
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Radio Map Estimation')
    parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
    parser.add_argument('--model_name', type=str, default='unet', help='Name of the experiment')
    parser.add_argument('--model_type', type=str, default='DAT_UNet', help='Name of model name')
    # image size for different dataset
    parser.add_argument('--data_name', type=str, default='sear', help='Name of the dataset: sear, pp5d')
    parser.add_argument('--simulation', type=str, default='DPM', help='simulation type for RadioMapSear dataset: DPM, IRT2, IRT4')
    parser.add_argument('--image_size', type=int, default=256, help='image size for different dataset: 128 or 256')
    parser.add_argument('--dataroot', type=str, default="/userhome/data", metavar='N',
                        help='data path')
    # experimental setting
    parser.add_argument('--epochs', type=int, default=50, help='number of epoch to train')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of train batch per domain')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of test batch per domain')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    # training parameter
    parser.add_argument('--sp_ratio', type=int, default=1, help='sample ratio: 1 2 3 5 10 20 (%) ')
    parser.add_argument('--free_space_only', action='store_true', default=False, help='loss w/o free_space_only')
    parser.add_argument('--wandb', action='store_true', default=True, help='using wandb or not')
    parser.add_argument('--in_channels', type=int, default=3, help='in_channels for different dataset: 1 or 3')
    # resume
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")

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

    args.out_path = args.out_path + '/' + args.data_name
    sp_base_number = 5 # the base number for spatial sampling
    # normalized range of the dB value of the input image
    dB_max, dB_min = 1, 0 # the max and min of the dB value of the input image

    # dataset selection
    if args.data_name == 'pp5d':
        train_dataset = RadioMap5DDataset(args.dataroot, 'train', imsize=args.image_size)
        test_dataset = RadioMap5DDataset(args.dataroot, 'test', imsize=args.image_size)
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
            print("using cars images for pp model training!")
            train_dataset = RadioUNetDataset(args.dataroot, 'train', simulation=args.simulation, carsSimul='yes', carsInput='yes')
            test_dataset = RadioUNetDataset(args.dataroot, 'test', simulation=args.simulation, carsSimul='yes', carsInput='yes')
        else:
            train_dataset = RadioUNetDataset(args.dataroot, 'train', simulation=args.simulation)
            test_dataset = RadioUNetDataset(args.dataroot, 'test', simulation=args.simulation)
    else:
        raise ValueError

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4, shuffle=False)
    print('train set size: {}, test set size: {}'.format(len(train_dataset), len(test_dataset)))

    print('using model {}'.format(args.model_name))
    timestr = datetime.now().strftime('%Y-%m-%d_%I-%M-%S_%p')
    model_name = f'{args.model_name}'

    if args.model_name.lower() == "unet":
        model = UNet(model_name,
                     model_type=args.model_type,
                     img_size=args.image_size,
                     in_channels=args.in_channels,
                     ).to(device)

    elif args.model_name.lower() == "unet_tx":
        model = UNet_Tx(model_name,
                     model_type=args.model_type,
                     img_size=args.image_size,
                     in_channels=args.in_channels,
                     ).to(device)
    else:
        raise ValueError("Invalid model name.")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs)

    print('**** Setup ****')
    print('Total params: %.2fK' % (sum(p.numel() for p in model.parameters()) * 10 ** -3))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) * 10 ** -6))
    print('************')
    
    model.fit_wandb(train_loader, test_loader, optim, scheduler,
                    min_samples=sp_base_number * args.sp_ratio,
                    max_samples=sp_base_number * args.sp_ratio,
                    dB_max=dB_max,
                    dB_min=dB_min,
                    epochs=args.epochs,
                    save_model_dir=args.out_path,
                    free_space_only=args.free_space_only,
                    save_model_epochs=args.epochs,
                    project_name='DAT-UNET',
                    args=args,
                    model=model
                    )

