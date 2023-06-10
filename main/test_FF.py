import sys
sys.path.append('/mnt/lvdisk1/miaodata/ff++_code/')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from dataset import *
from configs.default import _C as cfg
from configs.default import update_config
from network import EfficientNet
from network import Xception
from evaluate import get_EER_states, calculate_threshold




def parse_args():
    parser = argparse.ArgumentParser(description="BBN evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=True,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="best model file"
    )

    args = parser.parse_args()
    return args

def load_model(model, model_path, cfg):
    pretrain_dict = torch.load(
        model_path, map_location="cpu" if cfg.CPU_MODE else "cuda"
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
    print("Model has been loaded {}".format(model_path))
    return model
    

def test_model(dataLoader, model, cfg, devices):
    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    with torch.no_grad():
        for image, label in tqdm(dataLoader):
            img_label = label.numpy().astype(np.float)
            image = image.to(device)

            classes = model(image)

            classes = classes.view(1,-1).squeeze(0)
            classes = torch.sigmoid(classes)
            output_dis = classes.data.cpu().numpy()
            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)
            for i in range(output_dis.shape[0]):
                if output_dis[i] >= 0.5:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0
            tol_pred = np.concatenate((tol_pred, output_dis))
            tol_label = np.concatenate((tol_label, img_label))
            tol_pred_prob = np.concatenate((tol_pred_prob, output_pred))
            
        logloss = metrics.log_loss(tol_label, tol_pred)
        auc = metrics.roc_auc_score(tol_label, tol_pred)
        acc = metrics.accuracy_score(tol_label, tol_pred_prob)
        fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    print("TEST: logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
            logloss, acc*100, auc*100, eer*100
        ))

if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    test_set = eval(cfg.DATASET.DATASET_VALID)(cfg)
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")

    # model = EfficientNet.EfficientNetAutoB7()
    model = Xception.xception()

    model = load_model(model, args.model_path, cfg)
    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()
    model.eval()

    testLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    test_model(testLoader, model, cfg, device)
