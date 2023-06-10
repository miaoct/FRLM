import sys
sys.path.append('/mnt/lvdisk1/miaodata/ff++_code/')
import argparse
import os
import timeit
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from evaluate import get_EER_states, calculate_threshold
from network import EfficientNet
from network import Xception_map
from network import Xception_adlmask



mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]

def get_boundingbox(box_path):
    box_dict = {}
    with open(box_path, 'r') as data:
        lines = data.readlines()
        for line in lines:
            boxs = []
            path, boxstr = line.split(':')
            framenum = path.split('/')[-1].split('.')[0].split('_')[0]
            num = boxstr.translate(str.maketrans('', '', '[|]|\n| ')).split(',')
            box = [int(i) for i in num]
            assert len(box) == 4

            if framenum in box_dict:
                boxs.extend(box_dict[framenum])
                boxs.append(box)
            else:
                boxs = [box]

            box_dict[framenum] = boxs
    return box_dict


def large_bbox(bbox, shape, scale):
    [x0, y0, sizex, sizey] = bbox
    center_x, center_y = x0 + sizex // 2, y0 + sizey // 2
    sizex, sizey = int(sizex * scale), int(sizey * scale)
    x0 = max(int(center_x - sizex // 2), 0)
    y0 = max(int(center_y - sizey // 2), 0)
    sizex = min(shape[1] - x0, sizex)
    sizey = min(shape[0] - y0, sizey)
    return x0, y0, sizex, sizey


def get_face(frame, boxs):
    faces = []
    bboxs = []
    for bbox in boxs:
        x, y, sizex, sizey = large_bbox(bbox, frame.shape, scale=1.1)
        face = frame[y:y + sizey, x:x + sizex, :]
        faces.append(face)
        bboxs.append([x, y, sizex, sizey])

    return faces, bboxs


def extract_videos_faces(video_path, vid, masks_path, imgsize, batchsize, test_compression):
    testdataset = []
    if 'DFDC' in video_path:
        bbox_path = os.path.join(masks_path, vid.split('.')[0] + '.txt')
    else:
        bbox_path = os.path.join(masks_path, vid.split('.')[0].replace('/', '--') + '.txt')
    bboxs = get_boundingbox(bbox_path)
    if 'DeepFakeDetection' in video_path and test_compression != 'c23':
        vid = vid.replace('c23', test_compression)
    vid_path = os.path.join(video_path, vid)
    reader = cv2.VideoCapture(vid_path)
    framenum = -1
    while True:
        success, frame = reader.read()
        if not success:
            break
        framenum += 1
        index = '%03d' % framenum
        if index in bboxs:
            faces, bboxs_new = get_face(frame, bboxs[index])
            if faces == []:
                continue
            for i, face in enumerate(faces):
                img = cv2.resize(face, (imgsize, imgsize))
                image = img.astype(np.float32)[:, :, ::-1] / 255.
                image -= mean
                image /= std
                image = image.transpose((2, 0, 1))
                testdataset.append(image)

    reader.release()

    testloader = torch.utils.data.DataLoader(
        testdataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    return testloader

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

def test(testloader, model):
    model.eval()
    tol_pred = np.array([], dtype=float)
    tol_pred_prob = np.array([], dtype=np.float)
    with torch.no_grad():
        for _, batch in enumerate(testloader):
            images = batch.cuda()
            result_avg = model(images)
            # tol_pred = np.concatenate((tol_pred, pred[1].cpu().numpy().squeeze(1)))
            # \\\\\\\two output///////
            # classes = torch.max(result_avg, dim=1)[0]
            # output_dis = classes.data.cpu().numpy()
            # result_avg = result_avg.data.cpu().numpy()
            # output_pred = np.zeros((result_avg.shape[0]), dtype=np.float)
            # for i in range(result_avg.shape[0]):
            #     if result_avg[i,1] >= result_avg[i,0]:
            #         output_pred[i] = 1.0
            #     else:
            #         output_pred[i] = 0.0
            # \\\\\\\\one output////////
            classes = result_avg.view(1,-1).squeeze(0)
            classes = torch.sigmoid(classes)
            output_dis = classes.data.cpu().numpy()
            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)
            for i in range(output_dis.shape[0]):
                if output_dis[i] >= 0.5:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_pred = np.concatenate((tol_pred, output_dis)) 
            tol_pred_prob = np.concatenate((tol_pred_prob, output_pred))

    return tol_pred, tol_pred_prob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default="/mnt/lvdisk1/miaodata/Celeb-DF/")
    parser.add_argument('--test_list', default="/mnt/lvdisk1/miaodata/Celeb-DF/List_of_testing_videos.txt")
    parser.add_argument('--test_compression', type=str, default="c23")
    parser.add_argument('--image_size', type=int, default=380)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--gpu_id', default=(1,))
    parser.add_argument('--bbox_path', default="/mnt/lvdisk1/miaodata/Celeb-DF/testing_videos_bboxs/")
    parser.add_argument('--model_path', default='checkpoints/binary_faceforensicspp', help='folder to output model checkpoints')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    gpu_list = list(args.gpu_id)
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    
    model = EfficientNet.EfficientNetAutoB7()
    # model = Xception_map.xception_map(num_classes=1)
    # model = xception_landmark.xception_adl()

    model = load_model(model, args.model_path)
    model = nn.DataParallel(model) 
    model.cuda()
    model.eval()

    # prepare data
    start = timeit.default_timer()
    print('TEST MODEL: ' + args.model_path)
    if 'DeepFakeDetection' in args.dataset_root:
        print('TESTSET: ' + args.dataset_root + '  Dataset_compression:' + args.test_compression)
    else:
        print('TESTSET: ' + args.dataset_root)
    tol_label = np.array([])
    tol_pred = np.array([])
    tol_pred_prob = np.array([])
    vid_list = [line.strip().split() for line in open(args.test_list, 'r')]
    for item in tqdm(vid_list):
        label, vid = item
        if os.path.exists(os.path.join(args.dataset_root, vid)) == 0:
            print("can't find the video:" + vid)
            continue
        testloader = extract_videos_faces(args.dataset_root, vid, args.bbox_path, args.image_size,
                                        args.batchsize, args.test_compression)
        preds, pred_probs = test(testloader, model)
        if 'Celeb-DF' in args.dataset_root:
            labels = np. array([1.0-float(label)] * len(preds), dtype=float)
        else:
            labels = np.array([float(label)] * len(preds), dtype=float)
        tol_label = np.concatenate((tol_label, labels))
        tol_pred = np.concatenate((tol_pred, preds))
        tol_pred_prob = np.concatenate((tol_pred_prob, pred_probs))


    logloss = metrics.log_loss(tol_label, tol_pred)
    auc = metrics.roc_auc_score(tol_label, tol_pred)
    acc = metrics.accuracy_score(tol_label, tol_pred_prob)
    fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    print(' EER: {:.4f}, AUC:{:.4f}, logloss:{:.4f}, ACC:{:.4f}'.format(eer*100, auc*100, logloss, acc*100))
    end = timeit.default_timer()
    print('Mins: %d' % np.int((end - start) / 60))


if __name__ == '__main__':
    main()
