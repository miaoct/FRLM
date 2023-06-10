import math
import os
import random
import cv2
import numpy as np
import skimage.draw
from scipy.ndimage import binary_erosion, binary_dilation


def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def remove_eyes(image, landmarks, cfg):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2] #取坐标点
    mask = np.zeros_like(image[..., 0]) #全0数组，黑
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=cfg.DATASET.LANDMARK.THICK) #两点之间画白线
    w = dist((x1, y1), (x2, y2)) #两点之间欧氏距离
    dilation = int(w // cfg.DATASET.LANDMARK.EYE_DILA) #计算膨胀重复次数
    line = binary_dilation(line, iterations=dilation) #膨胀运算是将与白线接触的所有背景像素（黑色区域）合并到该物体中的过程
    image[line, :] = 0 #将图片中膨胀后白线位置全部置0，变黑
    return image


def remove_nose(image, landmarks, cfg):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=cfg.DATASET.LANDMARK.THICK)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // cfg.DATASET.LANDMARK.NOSE_DILA)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_mouth(image, landmarks, cfg):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[-2:]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=cfg.DATASET.LANDMARK.THICK)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // cfg.DATASET.LANDMARK.MOUTH_DILA)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_landmark(image, landmarks, cfg):
    random.seed(cfg.RANDOM_SEED)
    if random.random() > cfg.DATASET.LANDMARK.EYE_RATE: # > 0.5
        image = remove_eyes(image, landmarks, cfg)
    elif random.random() > cfg.DATASET.LANDMARK.NOSE_RATE:
        image = remove_nose(image, landmarks, cfg)
    elif random.random() > cfg.DATASET.LANDMARK.MOUTH_RATE:
        image = remove_mouth(image, landmarks, cfg)
    
    # image = remove_eyes(image, landmarks)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
    _, mask = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)

    return np.uint8(mask*255)


def remove_landmark_image(image, landmarks, cfg):
    random.seed(cfg.RANDOM_SEED)
    if random.random() > cfg.DATASET.LANDMARK.EYE_RATE: # > 0.5
        image = remove_eyes(image, landmarks, cfg)
    elif random.random() > cfg.DATASET.LANDMARK.NOSE_RATE:
        image = remove_nose(image, landmarks, cfg)
    elif random.random() > cfg.DATASET.LANDMARK.MOUTH_RATE:
        image = remove_mouth(image, landmarks, cfg)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image