import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image
from torchvision import transforms, utils

class MyDataSet(data.Dataset):
    def __init__(self, output_size=(112, 112), dir_file_path='', count=8, status='train', mode='dual'):
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform_set = []
        self.resize = transforms.Resize(output_size)
        self.path = dir_file_path
        self.status = status
        self.mode = mode
        self.count = count

        # load image file
        self.length = 0
        image_list = []
        with open(self.path, 'r') as f:
            for line in f:
                cur_pair = line.strip().split(' ')
                image_list.append(cur_pair)
        self.image_list = image_list
        self.length = len(image_list)    

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        imgs = []
        srcs = []
        tars = []
        # return: a list of images (number=count)
        # img = swapped image
        # src = gt image (the source face)
        # tar = the target face
        img_paths = self.image_list[idx][:self.count]
        src_paths = self.image_list[idx][self.count:2*self.count]
        tar_paths = None
        if self.mode == 'dual' and len(self.image_list[idx]) == (3 * self.count):
            tar_paths = self.image_list[idx][2*self.count:]        
        
        if self.path.split('/')[-2][:9] == 'new_swaps':
            label = int(tar_paths[0].split('/')[-1].split('.')[0])
        else:
            label = int(tar_paths[0].split('/')[-3])

        # if any augmentation is needed, just modify this
        transform_aug = transforms.RandomApply(self.transform_set, p=1) 
        aug = transforms.Compose([
            transform_aug,
            transforms.ToTensor()
        ])
        # each list contains self.count image
        for index in range(self.count):
            img = Image.open(img_paths[index])
            img = self.resize(img)
            img = aug(img)
            if img.size(0) == 1:
                img = torch.cat((img, img, img), dim=0)
            img = self.normalize(img)
            imgs.append(img)

            # src image is the source image (when deepfaked) or original image (when not deepfaked)
            src = Image.open(src_paths[index])
            src = self.resize(src)
            src = aug(src)
            if src.size(0) == 1:
                src = torch.cat((src, src, src), dim=0)
            src = self.normalize(src)
            srcs.append(src)

            # get tars (if necessary), for raw image return 0 instead (may need to change)
            if tar_paths is not None and self.mode == 'dual':
                tar = Image.open(tar_paths[index])
                tar = self.resize(tar)
                tar = aug(tar)
                if tar.size(0) == 1:
                    tar = torch.cat((tar, tar, tar), dim=0)
                tar = self.normalize(tar)
                tars.append(tar)
            else:
                tars.append(torch.zeros_like(tars[0]))

        return imgs, srcs, tars, label