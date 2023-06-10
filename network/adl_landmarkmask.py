"""
Original repository: https://github.com/junsukchoe/ADL
"""

import torch
import torch.nn as nn
import random


class ADL_landmarkmask(nn.Module):
    def __init__(self, adl_drop_rate=0.75, adl_drop_threshold=0.8, seed=8664):
        super(ADL_landmarkmask, self).__init__()
        torch.manual_seed(seed)
        if not (0 <= adl_drop_rate <= 1):
            raise ValueError("Drop rate must be in range [0, 1].")
        self.adl_drop_rate = adl_drop_rate #随机选择drop-mask和importance map
        self.attention = None
        self.drop_mask = None

    def forward(self, input_, landmarks_mask):
        if not self.training:
            return input_, None
        else:
            attention = torch.mean(input_, dim=1, keepdim=True) #计算self-attention map, b_size*1*H*W
            importance_map = torch.sigmoid(attention)
            drop_mask = self._drop_mask_2(attention, landmarks_mask)
            selected_map = self._select_map(importance_map, drop_mask)
            return input_.mul(selected_map), importance_map

    def _select_map(self, importance_map, drop_mask):
        random_tensor = torch.rand([], dtype=torch.float32) + self.adl_drop_rate
        binary_tensor = random_tensor.floor()
        return (1. - binary_tensor) * importance_map + binary_tensor * drop_mask

    def _drop_mask_1(self, attention, landmarks_mask):
        landmarks_mask = attention.mul(landmarks_mask)
        landmarks_map = torch.sigmoid(landmarks_mask)
        return landmarks_map

    def _drop_mask_2(self, attention, landmarks_mask):
        return landmarks_mask

    def _drop_mask_3(self, importance_map, landmarks_mask):
        landmarks_map = importance_map.mul(landmarks_mask)
        return landmarks_map