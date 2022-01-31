import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import cv2
import torchvision
import random

num_channels=1
class get_data(Dataset):
    def __init__(
        self,
        input_size,
        output_size,
        save_dir,
        split
    ):
        self.input_size=input_size
        self.output_size=output_size
        self.save_dir = save_dir
        self.split=split

        if self.split == 'train':
            self.patterns=np.load(self.save_dir+'train_patterns.npy')
            self.targets = np.load(self.save_dir + 'train_targets.npy')
        else:
            self.patterns = np.load(self.save_dir + 'val_patterns.npy')
            self.targets = np.load(self.save_dir + 'val_targets.npy')

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern_ = np.load(self.patterns[idx])
        #img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        r, g, b = cv2.split(pattern_)
        pattern_ = np.dstack((b, g, r))
        target_ = cv2.imread(self.targets[idx])


        c = random.randint(0, 2)
        pattern=pattern_[:,:,c]
        target = target_[:, :, c]


        pattern = (pattern - pattern.mean()) / pattern.std()
        target = cv2.resize(target, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
        target=cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX)
        return np.reshape(pattern, (num_channels,self.input_size, self.input_size)), np.reshape(target, (num_channels,self.output_size, self.output_size))