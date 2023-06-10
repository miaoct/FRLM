import os
import random
import numpy as np
from dataset.baseset import BalanceSet, UniformSet
from dataset.remove_landmarks import remove_landmark, remove_landmark_image
import cv2
import torch
import torchvision.transforms as transforms


class XCP_SET(BalanceSet):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        random.seed(cfg.RANDOM_SEED)
        self.imgs = self._get_img_list()
        self.transform = transforms.Compose([
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])

    def __getitem__(self, idx):
        img_name, label = self.imgs[idx]
        image = self._load_image(img_name)
        image = self.transform(image)

        return image, label


class XCP_MASKSET(BalanceSet):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        random.seed(cfg.RANDOM_SEED)
        self.imgs = self._get_img_list()
        self.mask_real2 = torch.zeros((1, 19, 19))
        self.mask_real5 = torch.zeros((1, 19, 19))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])

        self.transform_mask2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((19, 19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        self.transform_mask5 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((19, 19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def _load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask2 = self.transform_mask2(mask)
        mask5 = self.transform_mask5(mask)
        return mask2, mask5

    def __getitem__(self, idx):
        img_name, label = self.imgs[idx]
        mask_name = img_name.replace('faces', 'mask')

        # mask_name_list = mask_name.split('/')
        # mask_name_list[-3] = 'raw'
        # mask_path = '/'.join(mask_name_list)

        image = self._load_image(img_name)
        image = self.transform(image)
        # if label == 1:
        #     mask2, mask5 = self._load_mask(mask_path)
        # else:
        #     mask2 = self.mask_real2
        #     mask5 = self.mask_real5

        # return image, label, mask2, mask5
        if label == 0:
            mask2 = self.mask_real2
            mask5 = self.mask_real5
            return image, label, mask2, mask5
        elif os.path.exists(mask_name) and label == 1:
            mask2, mask5 = self._load_mask(mask_name)
            return image, label, mask2, mask5
        else:
             return None, None, None, None


class XCP_LANDMARKSET(BalanceSet):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        random.seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)
        self.cfg = cfg
        self.imgs = self._get_img_list()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        self.transform_landmark2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((19, 19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        self.transform_landmark5 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((19, 19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def _load_landmark(self, path, image):
        landmarks = np.load(path)
        landmarks_mask = remove_landmark(image, landmarks, self.cfg)
        landmarks_mask2 = self.transform_landmark2(landmarks_mask)
        landmarks_mask5 = self.transform_landmark5(landmarks_mask)
        return landmarks_mask2, landmarks_mask5

    def __getitem__(self,idx):
        img_name, label = self.imgs[idx]
        landmarks_name = img_name.replace('faces', 'landmarks')
        landmarks_path = landmarks_name.split('.')[0] + '.npy'

        if os.path.exists(landmarks_path):
            try:
                label_valid = label
                image = self._load_image(img_name)
                landmarks_mask2, landmarks_mask5 = self._load_landmark(landmarks_path, image)
                image = self.transform(image)
        
                return label_valid, image, landmarks_mask2, landmarks_mask5
            except Exception as e:
                print(e)
                print("error", img_name)
                return None, None, None, None
        else:
            return None, None, None, None


class XCP_LANDMARK_MASKSET_AUG(BalanceSet):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        random.seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)
        self.cfg = cfg
        self.imgs = self._get_img_list()
        self.mask_real2 = torch.zeros((1, 19, 19))
        self.mask_real5 = torch.zeros((1, 19, 19))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        self.transform_2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((19, 19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        self.transform_5 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((19, 19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def _load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask2 = self.transform_2(mask)
        mask5 = self.transform_5(mask)
        return mask2, mask5

    def _load_landmark(self, path, image):
        landmarks = np.load(path)
        landmarks_mask = remove_landmark(image, landmarks, self.cfg)
        landmarks_mask2 = self.transform_2(landmarks_mask)
        landmarks_mask5 = self.transform_5(landmarks_mask)
        return landmarks_mask2, landmarks_mask5

    def __getitem__(self,idx):
        img_name, label = self.imgs[idx]
        landmarks_name = img_name.replace('faces', 'landmarks')
        landmarks_path = landmarks_name.split('.')[0] + '.npy'

        if os.path.exists(landmarks_path):
            mask_name = img_name.replace('faces', 'mask')
            if label == 1:
                mask2, mask5 = self._load_mask(mask_name)
            else:
                mask2 = self.mask_real2
                mask5 = self.mask_real5

            try:
                label_valid = label
                image = self._load_image(img_name)
                landmarks_mask2, landmarks_mask5 = self._load_landmark(landmarks_path, image)
                image = self.transform(image)
        
                return label_valid, image, landmarks_mask2, landmarks_mask5, mask2, mask5
            except Exception as e:
                print(e)
                print("error", img_name)
                return None, None, None, None, None, None
        else:
            return None, None, None, None, None, None

class XCP_LANDMARK_MASKSET(BalanceSet):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        random.seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)
        self.cfg = cfg
        self.imgs = self._get_img_list()
        self.mask_real2 = torch.zeros((1, 19, 19))
        self.mask_real5 = torch.zeros((1, 19, 19))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        self.transform_2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((19, 19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        self.transform_5 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((19, 19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def _load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask2 = self.transform_2(mask)
        mask5 = self.transform_5(mask)
        return mask2, mask5

    def _load_landmark(self, path, image):
        landmarks = np.load(path)
        landmarks_mask = remove_landmark(image, landmarks, self.cfg)
        landmarks_mask2 = self.transform_2(landmarks_mask)
        landmarks_mask5 = self.transform_5(landmarks_mask)
        return landmarks_mask2, landmarks_mask5

    def __getitem__(self,idx):
        img_name, label = self.imgs[idx]
        landmarks_name = img_name.replace('faces', 'landmarks')
        landmarks_path = landmarks_name.split('.')[0] + '.npy'

        # landmarks_name = img_name.replace('faces', 'landmarks')
        # landmarks_name_list = landmarks_name.split('/')
        # landmarks_name_list[-3] = 'you_end'
        # landmarks_name = '/'.join(landmarks_name_list)
        # landmarks_path = landmarks_name.split('.')[0] + '.npy'

        if os.path.exists(landmarks_path):
            mask_path = img_name.replace('faces', 'mask')

            # mask_name = img_name.replace('faces', 'mask')
            # mask_name_list = mask_name.split('/')
            # mask_name_list[-3] = 'you_end'
            # mask_path = '/'.join(mask_name_list)

            if label == 1:
                mask2, mask5 = self._load_mask(mask_path)
            else:
                mask2 = self.mask_real2
                mask5 = self.mask_real5

            try:
                label_valid = label
                image = self._load_image(img_name)
                landmarks_mask2, landmarks_mask5 = self._load_landmark(landmarks_path, image)
                image = self.transform(image)
        
                return label_valid, image, landmarks_mask2, landmarks_mask5, mask2, mask5
            except Exception as e:
                print(e)
                print("error", img_name)
                return None, None, None, None, None, None
        else:
            return None, None, None, None, None, None

class XCP_VALID(UniformSet):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        random.seed(cfg.RANDOM_SEED)
        self.transform = transforms.Compose([
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        self.imgs = self._get_img_list()

    def __getitem__(self,idx):
        img_name, label = self.imgs[idx]
        image = self._load_image(img_name)
        image = self.transform(image)
        
        return image, label
    
class XCP_VALID_AUG(UniformSet):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        random.seed(cfg.RANDOM_SEED)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        self.imgs = self._get_img_list()

    def __getitem__(self,idx):
        img_name, label = self.imgs[idx]
        image = self._load_image(img_name)
        image = self.transform(image)
        
        return image, label

class XCP_LANDMARKIMAGESET(BalanceSet):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        random.seed(cfg.RANDOM_SEED)
        torch.manual_seed(cfg.RANDOM_SEED)
        self.imgs = self._get_img_list()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])

    def _load_landmark(self, path, image):
        landmarks = np.load(path)
        landmarks_image = remove_landmark_image(image, landmarks, self.cfg)
        
        return landmarks_image

    def __getitem__(self, idx):
        img_name, label = self.imgs[idx]
        landmarks_name = img_name.replace('faces', 'landmarks')
        landmarks_path = landmarks_name.split('.')[0] + '.npy'

        if os.path.exists(landmarks_path):
            try:
                label_valid = label
                image = self._load_image(img_name)

                random_tensor = torch.rand([], dtype=torch.float32) + 0.9
                binary_tensor = random_tensor.floor()
                if binary_tensor == 1:
                    image = self._load_landmark(landmarks_path, image)

                image = self.transform(image)
        
                return image, label_valid
            except Exception as e:
                print(e)
                print("error", img_name)
                return None, None
        else:
            return None, None