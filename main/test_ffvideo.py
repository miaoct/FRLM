import sys
sys.path.append('/mnt/lvdisk1/miaodata/ff++_code/')
import os
import random
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json, random, time
import cv2
import glob
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pandas as pd
from evaluate import get_EER_states, calculate_threshold
from network import EfficientNet
from network import EfficientNet_mask
from network import Xception
from network import Xception_mask


class TestSet(Dataset):

    def __init__(self, test_frames, face_path, model_type):
        random.seed(8664)
        self.facepath = face_path
        self.frames = test_frames
        self.model_type = model_type
        self.transform_eff = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380,380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_xcp = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        self.imgs = self._get_img_list()
        

    def _get_img_list(self):
        imgs = []
        
        filelist = glob.glob(os.path.join(self.facepath, '*.png'))
        try:
            for path in filelist[:self.frames]:
                imgs.append(path)
        except(IndexError):
            pass       
    
        return imgs

    def __len__(self):
        return(len(self.imgs))

    def _load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image

    def __getitem__(self,idx):
        img_name = self.imgs[idx]
        image = self._load_image(img_name)
        if self.model_type == "eff":
            image = self.transform_eff(image)
        else:
            image = self.transform_xcp(image)
        
        return image

def load_model(model, model_path):
    pretrain_dict = torch.load(
        model_path, map_location="cuda"
    )
    pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
    model_dict = model.state_dict()
    from collections import OrderedDict
    new_dict = OrderedDict()
    for k, v in pretrain_dict.items():
        if k.startswith("module"):
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    # print("Model has been loaded {}".format(model_path))
    return model

def test_video(testloder, model):
    model.eval()
    with torch.no_grad():
        preds = []
        for img in testloder:
            img = img.cuda()

            classes = model(img)

            classes = classes.view(1,-1).squeeze(0)
            classes = torch.sigmoid(classes)
            prob = classes.data.cpu().float().numpy()
            preds.append(prob)

    return np.mean(preds)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default="/mnt/lvdisk1/miaodata/FF/")
    parser.add_argument('--test_dir', default="test_c23.txt")
    parser.add_argument('--model_type', type=str, default="eff")
    parser.add_argument('--test_frames', type=int, default=5)
    parser.add_argument('--model_path', default='checkpoints/binary_faceforensicspp', help='folder to output model checkpoints')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    random.seed(8664)
    np.random.seed(8664)
    torch.manual_seed(8664)
    torch.cuda.manual_seed_all(8664)
    args = parse_args()

    if args.model_type == "eff":
        model = EfficientNet.EfficientNetAutoB7()
        # model = EfficientNet_mask.EfficientNetAutoADLB7_mask()
    else:
        model = Xception.xception()
        #model = Xception_mask.xception_mask()
    model = load_model(model, args.model_path)
    model = nn.DataParallel(model) 
    model.cuda()

    tol_label = np.array([], dtype=float)
    tol_pred = np.array([], dtype=float)
    tol_pred_prob = np.array([], dtype=np.float)

    datapath = os.path.join(args.dataset_root, args.test_dir)
    video_list = [line.strip().split() for line in open(datapath,'r')]
    for words in tqdm(video_list):
        try:
            face_path = os.path.join(args.dataset_root, words[0])
            testdataset = TestSet(test_frames=args.test_frames,
                            face_path=face_path, 
                            model_type=args.model_type)
            assert testdataset
        except:
            #print(words)
            continue
        
        testloder = DataLoader(
            testdataset,
            batch_size=args.test_frames,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        pred = test_video(testloder, model)
        preds = np.array([pred], dtype=float)
        output_pred = np.zeros((preds.shape[0]), dtype=np.float)
        for i in range(preds.shape[0]):
            if preds[i] >= 0.5:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        label = np.array([float(words[1])], dtype=float)
        tol_label = np.concatenate((tol_label, label))
        tol_pred = np.concatenate((tol_pred, preds))
        tol_pred_prob = np.concatenate((tol_pred_prob, output_pred))

    logloss = metrics.log_loss(tol_label, tol_pred)
    auc = metrics.roc_auc_score(tol_label, tol_pred)
    acc = metrics.accuracy_score(tol_label, tol_pred_prob)
    fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # submission_df = pd.DataFrame({"filename": video_list, "label": tol_pred_prob})
    # submission_df.to_csv('/mnt/lvdisk1/miaodata/ff++_code/submission_110.csv', index=False)

    print('TEST MODEL: ' + args.model_path)
    print('TESTSET: ' + args.test_dir)
    print('TESTFRAMES:{}'.format(args.test_frames))
    print(' EER: {:.4f}, AUC:{:.4f}, logloss:{:.4f}, ACC:{:.4f}'.format(eer*100, auc*100, logloss, acc*100))


