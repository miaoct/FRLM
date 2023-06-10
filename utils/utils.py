import logging
import time
import os
import torch
from torch import nn
from utils.lr_scheduler import WarmupMultiStepLR
from fvcore.nn import sigmoid_focal_loss

from network import EfficientNet
from network import EfficientNet_adlmask
from network import Xception
from network import Xception_adlmask
from network.Xception_new import TransferModel


def create_logger(cfg):
    dataset = cfg.DATASET.DATASET_TRAIN
    net_type = cfg.BACKBONE.TYPE
    log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}_{}.log".format(dataset, net_type, time_str)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file


def get_optimizer(cfg, model):
    base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR
    
    if cfg.TRAIN.OPTIMIZER.TYPE == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True,
        )
    elif cfg.TRAIN.OPTIMIZER.TYPE == "map_ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=base_lr,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY
        )
    elif cfg.TRAIN.OPTIMIZER.TYPE == "ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.999),
            eps=1e-08
        )
    else:
        print("optimizer error! ")
    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.TRAIN.LR_SCHEDULER.TYPE == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "cosine":
        if cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END, eta_min=1e-4
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.MAX_EPOCH, eta_min=1e-4
            )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
            warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARM_EPOCH,
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
            patience=cfg.TRAIN.LR_SCHEDULER.PATIENCE,
            cooldown=2 * cfg.TRAIN.LR_SCHEDULER.PATIENCE,
            min_lr= cfg.TRAIN.OPTIMIZER.BASE_LR * 1e-5,
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))

    return scheduler


def get_model(cfg, device):
    if cfg.BACKBONE.TYPE == "EfficientNetAutoB7":
        model = EfficientNet.EfficientNetAutoB7(dropout_rate=cfg.BACKBONE.DROPOUT_RATE)
    elif cfg.BACKBONE.TYPE == "EfficientNetAutoADLB7":
        model = EfficientNet_adlmask.EfficientNetAutoADLB7(
            dropout_rate=cfg.BACKBONE.DROPOUT_RATE,
            att_block_idx=cfg.BACKBONE.BLOCK_IDX,
            adl_drop_rate=cfg.BACKBONE.DROP_RATE,
            seed=cfg.RANDOM_SEED 
        )
    elif cfg.BACKBONE.TYPE == "xception":
        model = Xception.xception()
    elif cfg.BACKBONE.TYPE == "xception_new":
        model = TransferModel(modelchoice='xception', num_out_classes=2)
        model.set_trainable_up_to(True)
    elif cfg.BACKBONE.TYPE == "xception_landmark_mask":
        model = Xception_adlmask.xception_landmark_mask(adl_drop_rate=cfg.BACKBONE.DROP_RATE, seed=cfg.RANDOM_SEED)
    else:
        raise NotImplementedError("Unsupported Model: {}".format(cfg.BACKBONE.TYPE))

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    return model


def get_loss(cfg):
    if cfg.LOSS.LOSS1_TYPE == "BCE_Loss":
        loss1 = nn.BCEWithLogitsLoss().cuda()
    elif cfg.LOSS.LOSS1_TYPE == "Focal_Loss":
        loss1 = sigmoid_focal_loss
    elif cfg.LOSS.LOSS1_TYPE == "CE_Loss":
        loss1 = nn.CrossEntropyLoss().cuda()
    else:
        raise NotImplementedError("Unsupported loss: {}".format(cfg.LOSS.LOSS1_TYPE))

    loss2 = nn.L1Loss().cuda()
    return loss1, loss2

