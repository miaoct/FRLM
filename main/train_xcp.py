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
from function import train_XCP, valid_XCP

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

    
    # ----- BEGIN RESUME ---------
    all_models = os.listdir(model_dir)
    if len(all_models) <= 1 or auto_resume == False:
        auto_resume = False
    else:
        all_models.remove("best_model_acc.pth")
        resume_epoch = max([int(name.split(".")[0].split("_")[-1]) for name in all_models])
        resume_model_path = os.path.join(model_dir, 'checkpoint.pth.tar')

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
            best_acc = checkpoint['best_acc']
    # ----- END RESUME ---------

    logger.info(
        "-------------------Train start :{}-------------------".format(
            cfg.BACKBONE.TYPE
        )
    )

    best_logloss, best_acc, best_auc, best_eer, epoch = 1, 0, 0, 1, 0
    is_best_list = []   
    train_iter = iter(trainLoader)
    iter_per_epoch = len(train_iter)
    iter_per_valid = int(iter_per_epoch / 10.0)
    max_iter = (cfg.TRAIN.NITER - epoch) * iter_per_epoch
    iter_num0 = epoch * iter_per_epoch

    for iter_num in range(max_iter+1):
        iter_num += iter_num0
        
        if (iter_num !=0 and iter_num % iter_per_epoch == 0):
            model_save_path = os.path.join(model_dir,'checkpoint.pth.tar')
            train_iter = iter(trainLoader)
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_acc': best_acc,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, model_save_path)
            if len(is_best_list) == 20 and sum(is_best_list) == 0:
                    break
            # if epoch % 2 == 0:
            #     is_best_list = []
            epoch = epoch + 1

        loss_batch, acc_batch = train_XCP(
            train_iter,
            model,
            iter_num,
            max_iter,
            optimizer,
            criterion1,
            cfg,
            logger,
            writer=writer,
        )

        loss_dict = {"train_loss": loss_batch}
        acc_dict = {"train_acc": acc_batch}

        if (iter_num != 0 and (iter_num+1) % iter_per_valid == 0):
            loss_valid, acc_valid = valid_XCP(
                validLoader, iter_num, model, cfg, criterion1, logger, device, writer=writer
            )

            #scheduler.step(loss_valid)

            loss_dict["valid_loss"] = loss_valid
            acc_dict["valid_acc"] = acc_valid

            is_best = False
            # if logloss_valid < best_logloss:
            #     best_logloss = logloss_valid
            #     torch.save({
            #             'state_dict': model.state_dict(),
            #             'epoch': epoch,
            #             'best_logloss': best_logloss,
            #             'scheduler': scheduler.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #     }, os.path.join(model_dir, "best_model_logloss.pth")
            #     )

            if acc_valid > best_acc:
                is_best = True
                is_best_list = []
                best_acc = acc_valid
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_acc': best_acc,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, "best_model_acc.pth")
                )

            # if auc_valid > best_auc:
            #     best_auc = auc_valid
            #     torch.save({
            #             'state_dict': model.state_dict(),
            #             'epoch': epoch,
            #             'best_auc': best_auc,
            #             'scheduler': scheduler.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #     }, os.path.join(model_dir, "best_model_auc.pth")
            #     )

            # if eer_valid < best_eer:
            #     best_eer = eer_valid
            #     torch.save({
            #             'state_dict': model.state_dict(),
            #             'epoch': epoch,
            #             'best_eer': best_eer,
            #             'scheduler': scheduler.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #     }, os.path.join(model_dir, "best_model_eer.pth")
            #     )
            is_best_list.append(is_best)

            logger.info(
                "---Epoch:{:>3d} Best_Acc:{:>5.4f}%  ---".format(
                    epoch, best_acc*100
                )
            )
        if cfg.TRAIN.TENSORBOARD.ENABLE:
            writer.add_scalars("scalar/acc", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)
    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()
    logger.info(
        "-------------------Train Finished :{}-------------------".format(cfg.NAME)
    )
