"""
refer from https://github.com/RonLevie/RadioUNet
"""
from __future__ import print_function, division
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler

def split_set(dataset):
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler

class RadioUNetDataset(Dataset):
    def __init__(self,
                 dataroot,
                 partition="train",

                 numTx=80,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",

                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 ):

        self.maps_inds=np.arange(0,700,1,dtype=np.int32)

        self.dataroot = dataroot + '/RadioMapSeer/'
        self.numTx=  numTx

        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput

        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dataroot+"gain/DPM/"
            else:
                self.dir_gain=self.dataroot+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dataroot+"gain/IRT2/"
            else:
                self.dir_gain=self.dataroot+"gain/carsIRT2/"

        elif simulation=="IRT4":
            numTx = 2
            self.numTx = numTx
            if carsSimul=="no":
                self.dir_gain=self.dataroot+"gain/IRT4/"
            else:
                self.dir_gain=self.dataroot+"gain/carsIRT4/"

        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dataroot+"gain/DPM/"
                self.dir_gainIRT2=self.dataroot+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dataroot+"gain/carsDPM/"
                self.dir_gainIRT2=self.dataroot+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dataroot+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dataroot+"png/buildings_missing" # a random index will be concatenated in the code

        self.dir_antennas = self.dataroot+"png/antennas/"
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dataroot+ "png/cars/" 

        if partition == 'train':
            idx1 = 0*numTx
            idx2 = 500*numTx
            self.ss_nyx_samples = np.arange(idx1,idx2,1,dtype=np.int32)

        elif partition == 'valid':
            idx1 = 500*numTx
            idx2 = 600*numTx
            self.ss_nyx_samples = np.arange(idx1,idx2,1,dtype=np.int32)

        elif partition == 'test':
            idx1 = 600*numTx
            idx2 = 700*numTx
            self.ss_nyx_samples = np.arange(idx1,idx2,1,dtype=np.int32)

    def __len__(self):
        return len(self.ss_nyx_samples)
    
    def __getitem__(self, idx):

        idx = self.ss_nyx_samples[idx]
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx


        dataset_map_ind=self.maps_inds[idxr]+1

        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(Image.open(img_name_buildings)) / 255.0

        #Load antennas:
        img_name_antennas = os.path.join(self.dir_antennas, name2)
        image_antennas = np.asarray(Image.open(img_name_antennas)) / 255.0

        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(Image.open(img_name_gain)),axis=0)/255.0

        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2)
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(Image.open(img_name_gainIRT2)),axis=0)/255.0  \
                        + (1-w)*np.expand_dims(np.asarray(Image.open(img_name_gainDPM)),axis=0)/255.0

        complete_map = image_gain
        building_mask = 1.0 - image_buildings
        building_mask = np.expand_dims(building_mask, axis=0)
        image_antennas = np.expand_dims(image_antennas, axis=0)
        if self.carsInput == "no":
            return complete_map, building_mask, image_antennas
        else:
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(Image.open(img_name_cars)) / 255.0
            image_cars = np.expand_dims(image_cars, axis=0)
            return complete_map, building_mask, image_cars, image_antennas
