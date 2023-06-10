from functools import partial

import numpy as np
import torch
from timm.models import skresnext50_32x4d
from timm.models.dpn import dpn92, dpn131
from timm.models.efficientnet import EfficientNet
from timm.models.efficientnet_builder import decode_arch_def
from timm.models.efficientnet_blocks import round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import load_pretrained, adapt_model_from_file
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d

from network.adl_landmarkmask import ADL_landmarkmask


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }

default_cfgs = {
    'tf_efficientnet_b5_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth',
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth',
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_l2_ns_475': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns_475-bebbd00a.pth',
        input_size=(3, 475, 475), pool_size=(15, 15), crop_pct=0.936),
    'tf_efficientnet_l2_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns-df73bb44.pth',
        input_size=(3, 800, 800), pool_size=(25, 25), crop_pct=0.96),

}

_DEBUG = False


def _create_model(model_kwargs, default_cfg, pretrained=False):
    if model_kwargs.pop('features_only', False):
        load_strict = False
        model_kwargs.pop('num_classes', 0)
        model_kwargs.pop('num_features', 0)
        model_kwargs.pop('head_conv', None)
        model_class = EfficientNetFeatures
    else:
        load_strict = True
        model_class = EfficientNetAutoADL
    variant = model_kwargs.pop('variant', '')
    model = model_class(**model_kwargs)
    model.default_cfg = default_cfg
    if '_pruned' in variant:
        model = adapt_model_from_file(model, variant)
    if pretrained:
        load_pretrained(
            model,
            default_cfg,
            num_classes=model_kwargs.get('num_classes', 0),
            in_chans=model_kwargs.get('in_chans', 3),
            strict=load_strict)
    return model


def _gen_efficientnet(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_kwargs=resolve_bn_args(kwargs),
        variant=variant,
        **kwargs,
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def tf_efficientnet_b7_ns(pretrained=False, **kwargs):
    """ EfficientNet-B7 NoisyStudent. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b7_ns', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model

class FeatureExtractor(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


"""
EfficientNetAutoADL
"""


class EfficientNetAutoADL(EfficientNet):
    def init_att(self, att_block_idx=[2,5], adl_drop_rate=0.75, seed=8664):
        """
        Initialize attention
        :param model: efficientnet-bx, x \in {0,..,7}
        :param depth: attention width
        :return:
        """
        
        self.att_block_idx = att_block_idx
        self.adl_drop_rate = adl_drop_rate
        self.seed = seed
        self.adl_landmarkmask = ADL_landmarkmask(adl_drop_rate=self.adl_drop_rate, seed=self.seed)

    
    def extract_features(self, x: torch.Tensor, landmark_mask2: torch.Tensor, landmark_mask5: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Blocks
        for idx, block in enumerate(self.blocks):
            
            x = block(x) 
            if idx == self.att_block_idx[0]:
                x, mask2= self.adl_landmarkmask(x, landmark_mask2)
            if idx == self.att_block_idx[1]:
                x, mask5 = self.adl_landmarkmask(x, landmark_mask5)

        # Head
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x, mask2, mask5


class EfficientNetGenAutoADL(FeatureExtractor):
    def __init__(self, dropout_rate=0, att_block_idx=[2,5], adl_drop_rate=0.75, seed=8664):
        super(EfficientNetGenAutoADL, self).__init__()
        self.efficientnet = tf_efficientnet_b7_ns(pretrained=True, drop_path_rate=0.2)
        self.efficientnet.init_att(att_block_idx, adl_drop_rate, seed)
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(2560, 1)

    def features(self, x: torch.Tensor, landmark_mask2: torch.Tensor, landmark_mask5: torch.Tensor) -> torch.Tensor:
        x, mask2, mask5 = self.efficientnet.extract_features(x, landmark_mask2, landmark_mask5)
        return x, mask2, mask5

    def forward(self, x, landmark_mask2, landmark_mask5):
        x, mask2, mask5 = self.features(x, landmark_mask2, landmark_mask5)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x, mask2, mask5

class EfficientNetAutoADLB7(EfficientNetGenAutoADL):
    def __init__(self, dropout_rate=0, att_block_idx=[2,5], adl_drop_rate=0.75, seed=8664):
        super(EfficientNetAutoADLB7, self).__init__()