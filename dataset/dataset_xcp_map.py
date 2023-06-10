import os
import random
import numpy as np
from dataset.baseset import BalanceSet, UniformSet
from dataset.remove_landmarks import remove_landmark
import cv2
import torch
import torchvision.transforms as transforms

class XCP_MASKSET(BalanceSet):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        random.seed(cfg.RANDOM_SEED)
        self.imgs = self._get_img_list()
        self.mask_real = torch.zeros((1, 19, 19))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])

        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((19,19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def _load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.transform_mask(mask)
       
        return mask

    def __getitem__(self, idx):
        img_name, label = self.imgs[idx]
        mask_name = img_name.replace('faces', 'mask')

        image = self._load_image(img_name)
        image = self.transform(image)

        if label == 1:
            mask = self._load_mask(mask_name)
        else:
            mask = self.mask_real

        return image, label, mask


class XCP_SET(BalanceSet):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        random.seed(cfg.RANDOM_SEED)
        self.imgs = self._get_img_list()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])

    def __getitem__(self, idx):
        img_name, label = self.imgs[idx]
        image = self._load_image(img_name)
        image = self.transform(image)

        return image, label


class XCP_VALID(UniformSet):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        random.seed(cfg.RANDOM_SEED)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        self.imgs = self._get_img_list()

    def __getitem__(self,idx):
        img_name, label = self.imgs[idx]
        image = self._load_image(img_name)
        image = self.transform(image)
        
        return image, label
