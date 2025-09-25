import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

class RadioMap5DDataset(Dataset):
    def __init__(self, dataroot, partition='train', imsize=128, frequency=[i for i in range(5)], heights=[i for i in range(3)], terrains=[i for i in range(11)]):
        '''
        You can download the 5d dataset from https://spectrum-net.github.io/
        5d dataset contains 5 frequency, 3 heights, 11 terrains
        example:
        04.Lake
        a rme data: T04C0D0000_n00_f00_ss_z00.png
        file name:T04C0D0000 2nd and 3rd numbers represent the terrain
        n00: the squence number
        f00: the frequency
        z00: the height
        the rme data is response to a npz file, which is T04C0D0000_n00_bdtr.npz

        use input frequency, heights, terrains to filter the data
        '''
        dataroot = dataroot + '/PPData5D-success'

        self.png_dir = dataroot + '/png'
        self.npz_dir = dataroot + '/npz'
        self.imsize = imsize
        self.frequency = frequency
        self.heights = heights
        self.terrains = terrains
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.imsize, self.imsize), antialias=False),

        ])
        self._imgs_list = []

        for terrain in os.listdir(self.png_dir):
            if int(terrain.split('.')[0]) - 1 in self.terrains:
                self.sub_img_dir = os.path.join(self.png_dir, terrain)
                for img in os.listdir(self.sub_img_dir):
                    parts = img.split('_') # ['T04C0D0052', 'n04', 'f02', 'ss', 'z01.png']
                    filename = parts[0]
                    terrain_num = int(filename[1: 3])
                    frequency_num = int(parts[2][1:])
                    height_num = int(parts[4][1: 3])
                    if terrain_num not in self.terrains \
                            or height_num not in self.heights \
                            or frequency_num not in self.frequency:
                        continue
                    else:
                        self._imgs_list.append(os.path.join(self.png_dir, terrain, img))
        self._imgs_list = sorted(self._imgs_list)
        np.random.shuffle(self._imgs_list )

        if partition == 'train':
            self._imgs_list = self._imgs_list[:int(0.7 * len(self._imgs_list))]
        elif partition == 'val':
            self._imgs_list = self._imgs_list[int(0.7 * len(self._imgs_list)):int(0.8 * len(self._imgs_list))]
        elif partition == 'test':
            self._imgs_list = self._imgs_list[int(0.8 * len(self._imgs_list)):]
        else:
            raise ValueError('partition must be train or test')

    def __len__(self):
        return len(self._imgs_list)

    def __getitem__(self, index):
        img_name = self._imgs_list[index]
        img_path = os.path.join(self.sub_img_dir, img_name)

        parts = img_name.split('/')[-1].split('_')
        height_num = int(parts[4][1: 3])
        npz_name = f'{parts[0]}_{parts[1]}_bdtr.npz'
        npz_path = os.path.join(self.npz_dir, npz_name)
        npz = np.load(npz_path)

        building_map = npz['inBldg_zyx'][height_num]
        terrains_map = npz['terrain_yx']
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
            building_map = self.transform(building_map)
            terrains_map = self.transform(terrains_map)
            terrains_map = self.normalize_coords(terrains_map)
        complete_map = img
        building_mask = 1.0 - building_map
        return complete_map, building_mask, terrains_map

    def normalize_coords(self, input):
        return (input-input.min()) / (input.max() - input.min() + 1e-5)
