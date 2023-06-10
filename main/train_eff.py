import sys
sys.path.append('/mnt/lvdisk1/miaodata/ff++_code/')
import random
import numpy as np

from dataset import *
from configs.default import _C as cfg
from configs.default import update_config
from utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_model,
    get_loss,
)
from function import train_EFF, valid_EFF

import torch
import os, shutil
from torch.utils.data import DataLoader
import argparse
import warnings
import click
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import ast


def parse_args():
    parser = argparse.ArgumentParser(description="codes for FF++_Eff")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type= ast.literal_eval,
        dest = 'auto_resume',
        required=False,
        default= True,
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    logger, log_file = create_logger(cfg)
    warnings.filterwarnings("ignore")

    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    cudnn.benchmark = True

    auto_resume = args.auto_resume

    train_set = eval(cfg.DATASET.DATASET_TRAIN)(cfg)
    valid_set = eval(cfg.DATASET.DATASET_VALID)(cfg)

    
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")

    criterion1, criterion2 = get_loss(cfg)
    epoch_number = cfg.TRAIN.MAX_EPOCH

    # ----- BEGIN MODEL BUILDER -----
    model = get_model(cfg, device)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    # ----- END MODEL BUILDER -----

    trainLoader = DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        collate_fn=eval(cfg.TRAIN.COLLATE_FN),
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    validLoader = DataLoader(
        valid_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    # close loop
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "codes")
    tensorboard_dir = (
        os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboard")
        if cfg.TRAIN.TENSORBOARD.ENABLE
        else None
    )

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        logger.info(
            "This directory has already existed, Please remember to modify your cfg.NAME"
        )
        if not click.confirm(
            "\033[1;31;40mContinue and override the former directory?\033[0m",
            default=False,
        ):
            exit(0)
        shutil.rmtree(code_dir)
        if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
    print("=> output model will be saved in {}".format(model_dir))
    this_dir = os.path.dirname(__file__)
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
    )
    shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)

    if tensorboard_dir is not None:
        writer = SummaryWriter(log_dir=tensorboard_dir)
    else:
        writer = None

    best_logloss, best_acc, best_auc, best_eer, best_epoch, start_epoch = 1, 0, 0, 1, 0, 1
    # ----- BEGIN RESUME ---------
    all_models = os.listdir(model_dir)
    if len(all_models) <= 1 or auto_resume == False:
        auto_resume = False
    else:
        all_models.remove("best_model_logloss.pth")
        all_models.remove("best_model_acc.pth")
        resume_epoch = max([int(name.split(".")[0].split("_")[-1]) for name in all_models])
        resume_model_path = os.path.join(model_dir, "epoch_{}.pth".format(resume_epoch))

    if cfg.RESUME_MODEL != "" or auto_resume:
        if cfg.RESUME_MODEL == "":
            resume_model = resume_model_path
        else:
            resume_model = cfg.RESUME_MODEL if '/' in cfg.RESUME_MODEL else os.path.join(model_dir, cfg.RESUME_MODEL)
        logger.info("Loading checkpoint from {}...".format(resume_model))
        checkpoint = torch.load(
            resume_model, map_location="cpu" if cfg.CPU_MODE else "cuda"
        )
        if cfg.CPU_MODE:
            model.load_model(resume_model)
        else:
            model.module.load_model(resume_model)
        if cfg.RESUME_MODE != "state_dict":
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_logloss = checkpoint['best_logloss']
            best_epoch = checkpoint['best_epoch']
    # ----- END RESUME ---------

    logger.info(
        "-------------------Train start :{}-------------------".format(
            cfg.BACKBONE.TYPE
        )
    )

    for epoch in range(start_epoch, epoch_number + 1):
        # scheduler.step()
        loss_train, logloss_train, acc_train, auc_train, eer_train = train_EFF(
            trainLoader,
            model,
            epoch,
            epoch_number,
            optimizer,
            criterion1,
            cfg,
            logger,
            writer=writer,
        )
        model_save_path = os.path.join(
            model_dir,
            "epoch_{}.pth".format(epoch),
        )
        if epoch % cfg.SAVE_STEP == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_logloss': best_logloss,
                'best_epoch': best_epoch,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, model_save_path)

        loss_dict = {"train_loss": loss_train}
        logloss_dict = {"train_logloss": logloss_train}
        acc_dict = {"train_acc": acc_train}
        auc_dict = {"train_auc": auc_train}
        eer_dict = {"train_eer": eer_train}

        if cfg.VALID_STEP != -1 and epoch % cfg.VALID_STEP == 0:
            loss_valid, logloss_valid, acc_valid, auc_valid, eer_valid = valid_EFF(
                validLoader, epoch, model, cfg, criterion1, logger, device, writer=writer
            )

            scheduler.step(loss_valid)

            loss_dict["valid_loss"] = loss_valid
            logloss_dict["valid_logloss"] = logloss_valid
            acc_dict["valid_acc"] = acc_valid
            auc_dict["valid_auc"] = auc_valid
            eer_dict["valid_eer"] = eer_valid

            if logloss_valid < best_logloss:
                best_logloss, best_epoch = logloss_valid, epoch
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_logloss': best_logloss,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, "best_model_logloss.pth")
                )

            if acc_valid > best_acc:
                best_acc, best_epoch = acc_valid, epoch
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_result': best_acc,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, "best_model_acc.pth")
                )

            if auc_valid > best_auc:
                best_auc, best_epoch = auc_valid, epoch
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_result': best_auc,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, "best_model_auc.pth")
                )

            if eer_valid < best_eer:
                best_eer, best_epoch = eer_valid, epoch
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_logloss': best_eer,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, "best_model_eer.pth")
                )

            logger.info(
                "---Best_Epoch:{:>3d} Best_logloss:{:>5.4f} Best_Acc:{:>5.4f}% Best_Auc:{:>5.4f}% Best_EER:{:>5.4f}% ---".format(
                    best_epoch, best_logloss, best_acc*100, best_auc*100, best_eer*100
                )
            )
        if cfg.TRAIN.TENSORBOARD.ENABLE:
            writer.add_scalars("scalar/eer", eer_dict, epoch)
            writer.add_scalars("scalar/auc", auc_dict, epoch)
            writer.add_scalars("scalar/acc", acc_dict, epoch)
            writer.add_scalars("scalar/logloss", logloss_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)
    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()
    logger.info(
        "-------------------Train Finished :{}-------------------".format(cfg.NAME)
    )
