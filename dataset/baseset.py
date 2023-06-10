from torch.utils.data import Dataset
import torch
import json, os, random, time
import cv2
import torchvision.transforms as transforms
import numpy as np
import glob
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
class BalanceSet(Dataset):
    def __init__(self, cfg=None):
        random.seed(cfg.RANDOM_SEED)
        self.cfg = cfg
        self.rootpath = cfg.DATASET.ROOT
        self.train_txt = cfg.DATASET.TRAIN_TXT
        self.frames = cfg.DATASET.TRAIN_FRAMES
        self.realfake = cfg.DATASET.REALFAKE
        self.transform = None
        self.imgs = self._get_img_list()
        

    def _get_img_list(self):
        datapath = os.path.join(self.rootpath, self.train_txt)
        imgsfolderPath = open(datapath,'r')
        real_num = int(self.realfake * self.frames)
        fake_num = int(self.frames)
        imgs = []
        
        for line in imgsfolderPath:
            line = line.rstrip()
            words = line.split()
            if self.cfg.DATASET.NEWFACE:
                words[0] = words[0].replace('faces', self.cfg.DATASET.NEWFACE_NAME)
            filelist = glob.glob(os.path.join(self.rootpath, words[0], '*.png'))
            if int(words[1]) == 0:
                try:
                    for real_path in filelist[:real_num]:
                        imgs.append((real_path, int(words[1])))
                except(IndexError):
                    pass       
            else:
                try:
                    for fake_path in filelist[:fake_num]:
                        imgs.append((fake_path, int(words[1])))    
                except(IndexError):
                    pass
        # print('real{}, fake{}'.format(real_num, fake_num))
        return imgs

    def __len__(self):
        return(len(self.imgs))

    def _load_image(self, path):
        # image = cv2.imread(path, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(path).convert('RGB')
        return image

    def __getitem__(self,idx):
        img_name, label = self.imgs[idx]
        image = self._load_image(img_name)
        image = self.transform(image)
        
        return image, label


class UniformSet(Dataset):

    def __init__(self, cfg=None):
        random.seed(cfg.RANDOM_SEED)
        self.cfg = cfg
        self.rootpath = cfg.DATASET.ROOT
        self.valid_txt = cfg.DATASET.VALID_TXT
        self.frames = cfg.DATASET.VALID_FRAMES
        self.transform = None
        self.imgs = self._get_img_list()
        

    def _get_img_list(self):
        datapath = os.path.join(self.rootpath, self.valid_txt)
        imgsfolderPath = open(datapath,'r')
        real_num = int(self.frames)
        fake_num = int(self.frames)
        imgs = []
        
        for line in imgsfolderPath:
            line = line.rstrip()
            words = line.split()
            if self.cfg.DATASET.NEWFACE:
                words[0] = words[0].replace('faces', self.cfg.DATASET.NEWFACE_NAME)
            filelist = glob.glob(os.path.join(self.rootpath, words[0], '*.png'))
            if int(words[1]) == 0:
                try:
                    for real_path in filelist[:real_num]:
                        imgs.append((real_path, int(words[1])))
                except(IndexError):
                    pass       
            else:
                try:
                    for fake_path in filelist[:fake_num]:
                        imgs.append((fake_path, int(words[1])))    
                except(IndexError):
                    pass
        # print('real{}, fake{}'.format(real_num, fake_num))
        return imgs

    def __len__(self):
        return(len(self.imgs))

    def _load_image(self, path):
        # image = cv2.imread(path, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(path).convert('RGB')
        
        return image

    def __getitem__(self,idx):
        img_name, label = self.imgs[idx]
        image = self._load_image(img_name)
        image = self.transform(image)
        
        return image, label

