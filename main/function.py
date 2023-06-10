from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import torch
import time
from tqdm import tqdm
from evaluate import get_EER_states, calculate_threshold


def train_EFF_landmark_mask(
    trainLoader,
    model,
    epoch,
    epoch_number,
    optimizer,
    criterion_1,
    criterion_2,
    cfg,
    logger,
    **kwargs
):
    model.train()
    loss_train = 0
    loss_l1 = 0
    count = 0
    tol_label = np.array([], dtype=np.float) 
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)
    number_batch = len(trainLoader)

    for label, img_data, landmark_mask2, landmark_mask5, mask_gt2, mask_gt5 in tqdm (trainLoader):
        img_label = label.numpy().astype(np.float)

        labels_data = label.cuda().float()
        img_data = img_data.cuda()
        landmark_mask2 = landmark_mask2.cuda()
        landmark_mask5 = landmark_mask5.cuda()
        mask_gt2 = mask_gt2.cuda()
        mask_gt5 = mask_gt5.cuda()
        
        classes, mask2, mask5 = model(img_data, landmark_mask2, landmark_mask5)
        
        image_loss = criterion_1(classes, labels_data.view(-1, 1))
        mask_loss = cfg.TRAIN.LOSS_RATIO.MASK2 * criterion_2(mask2, mask_gt2) + cfg.TRAIN.LOSS_RATIO.MASK5 * criterion_2(mask5, mask_gt5)
        loss = image_loss + cfg.TRAIN.LOSS_RATIO.MASK * mask_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        
        loss_l1_data = mask_loss.item()
        loss_dis_data = loss.item()
        loss_train += loss_dis_data
        loss_l1 += loss_l1_data
        count += 1

        if count % cfg.SHOW_STEP == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_L1loss:{:>5.3f}".format(
                epoch, count, number_batch, loss_dis_data, loss_l1_data
            )
            logger.info(pbar_str)

    logloss_train = metrics.log_loss(tol_label, tol_pred)
    auc_train = metrics.roc_auc_score(tol_label, tol_pred)
    acc_train = metrics.accuracy_score(tol_label, tol_pred_prob)
    fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
    eer_train = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    loss_train /= count
    loss_l1 /= count
    
    pbar_str = "Train: Epoch:{:>3d}/{}  Avg_Loss:{:>5.3f}  Avg_L1Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
        epoch, epoch_number, loss_train, loss_l1, logloss_train, acc_train*100, auc_train*100, eer_train*100
    )
    logger.info(pbar_str)
    return loss_train, loss_l1, logloss_train, acc_train, auc_train, eer_train


def valid_EFF_landmark_mask(
    validLoader, epoch_number, model, cfg, criterion_1, logger, device, **kwargs
):
    model.eval()
    
    with torch.no_grad():
        loss_valid = 0
        count = 0
        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)
        tol_pred_prob = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(validLoader):
            img_label = labels_data.numpy().astype(np.float)

            img_data = img_data.cuda()
            labels_data = labels_data.cuda().float()

            classes, _, _ = model(img_data, None, None)
        
            image_loss = criterion_1(classes, labels_data.view(-1, 1)) 
            
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

            loss_dis_data = image_loss.item()
            loss_valid += loss_dis_data
            count += 1
            
        logloss_valid = metrics.log_loss(tol_label, tol_pred)
        auc_valid = metrics.roc_auc_score(tol_label, tol_pred)
        acc_valid = metrics.accuracy_score(tol_label, tol_pred_prob)
        fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
        eer_valid = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        loss_valid /= count

        pbar_str = "Valid: Epoch:{:>3d}  Avg_Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
            epoch_number, loss_valid, logloss_valid, acc_valid*100, auc_valid*100, eer_valid*100
        )
        logger.info(pbar_str)
    return loss_valid, logloss_valid, acc_valid, auc_valid, eer_valid

def train_EFF_mask(
    trainLoader,
    model,
    epoch,
    epoch_number,
    optimizer,
    criterion_1,
    criterion_2,
    cfg,
    logger,
    **kwargs
):
    model.train()
    loss_train = 0
    loss_l1 = 0
    count = 0
    tol_label = np.array([], dtype=np.float) 
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)
    number_batch = len(trainLoader)

    for img_data, label, mask_gt2, mask_gt5 in tqdm (trainLoader):
        img_label = label.numpy().astype(np.float)

        labels_data = label.cuda().float()
        img_data = img_data.cuda()
        mask_gt2 = mask_gt2.cuda()
        mask_gt5 = mask_gt5.cuda()
        
        classes, mask2, mask5 = model(img_data)
        
        image_loss = criterion_1(classes, labels_data.view(-1, 1))
        mask_loss = cfg.TRAIN.LOSS_RATIO.MASK2 * criterion_2(mask2, mask_gt2) + cfg.TRAIN.LOSS_RATIO.MASK5 * criterion_2(mask5, mask_gt5)
        loss = image_loss + cfg.TRAIN.LOSS_RATIO.MASK * mask_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        
        loss_l1_data = mask_loss.item()
        loss_dis_data = loss.item()
        loss_train += loss_dis_data
        loss_l1 += loss_l1_data
        count += 1

        if count % cfg.SHOW_STEP == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_L1loss:{:>5.3f}".format(
                epoch, count, number_batch, loss_dis_data, loss_l1_data
            )
            logger.info(pbar_str)

    logloss_train = metrics.log_loss(tol_label, tol_pred)
    auc_train = metrics.roc_auc_score(tol_label, tol_pred)
    acc_train = metrics.accuracy_score(tol_label, tol_pred_prob)
    fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
    eer_train = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    loss_train /= count
    loss_l1 /= count
    
    pbar_str = "Train: Epoch:{:>3d}/{}  Avg_Loss:{:>5.3f}  Avg_L1Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
        epoch, epoch_number, loss_train, loss_l1, logloss_train, acc_train*100, auc_train*100, eer_train*100
    )
    logger.info(pbar_str)
    return loss_train, loss_l1, logloss_train, acc_train, auc_train, eer_train

def valid_EFF_mask(
    validLoader, epoch_number, model, cfg, criterion_1, logger, device, **kwargs
):
    model.eval()
    
    with torch.no_grad():
        loss_valid = 0
        count = 0
        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)
        tol_pred_prob = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(validLoader):
            img_label = labels_data.numpy().astype(np.float)

            img_data = img_data.cuda()
            labels_data = labels_data.cuda().float()

            classes, _, _ = model(img_data)
        
            image_loss = criterion_1(classes, labels_data.view(-1, 1)) 
            
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

            loss_dis_data = image_loss.item()
            loss_valid += loss_dis_data
            count += 1
            
        logloss_valid = metrics.log_loss(tol_label, tol_pred)
        auc_valid = metrics.roc_auc_score(tol_label, tol_pred)
        acc_valid = metrics.accuracy_score(tol_label, tol_pred_prob)
        fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
        eer_valid = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        loss_valid /= count

        pbar_str = "Valid: Epoch:{:>3d}  Avg_Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
            epoch_number, loss_valid, logloss_valid, acc_valid*100, auc_valid*100, eer_valid*100
        )
        logger.info(pbar_str)
    return loss_valid, logloss_valid, acc_valid, auc_valid, eer_valid

def train_EFF_landmark(
    trainLoader,
    model,
    epoch,
    epoch_number,
    optimizer,
    criterion_1,
    cfg,
    logger,
    **kwargs
):
    model.train()
    loss_train = 0
    loss_l1 = 0
    count = 0
    tol_label = np.array([], dtype=np.float) 
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)
    number_batch = len(trainLoader)

    for label, img_data, landmark_mask2, landmark_mask5 in tqdm (trainLoader):
        img_label = label.numpy().astype(np.float)

        labels_data = label.cuda().float()
        img_data = img_data.cuda()
        landmark_mask2 = landmark_mask2.cuda()
        landmark_mask5 = landmark_mask5.cuda()
        
        classes = model(img_data, landmark_mask2, landmark_mask5)
        
        image_loss = criterion_1(classes, labels_data.view(-1, 1))
        
        loss = image_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        
        loss_dis_data = loss.item()
        loss_train += loss_dis_data
        count += 1

        if count % cfg.SHOW_STEP == 0:
            pbar_str = "-------Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f} --------".format(
                epoch, count, number_batch, loss_dis_data
            )
            logger.info(pbar_str)

    logloss_train = metrics.log_loss(tol_label, tol_pred)
    auc_train = metrics.roc_auc_score(tol_label, tol_pred)
    acc_train = metrics.accuracy_score(tol_label, tol_pred_prob)
    fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
    eer_train = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    loss_train /= count
    
    pbar_str = "Train: Epoch:{:>3d}/{}  Avg_Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
        epoch, epoch_number, loss_train, logloss_train, acc_train*100, auc_train*100, eer_train*100
    )
    logger.info(pbar_str)
    return loss_train, logloss_train, acc_train, auc_train, eer_train

def valid_EFF_landmark(
    validLoader, epoch_number, model, cfg, criterion_1, logger, device, **kwargs
):
    model.eval()
    
    with torch.no_grad():
        loss_valid = 0
        count = 0
        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)
        tol_pred_prob = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(validLoader):
            img_label = labels_data.numpy().astype(np.float)

            img_data = img_data.cuda()
            labels_data = labels_data.cuda().float()

            classes = model(img_data, None, None)
        
            image_loss = criterion_1(classes, labels_data.view(-1, 1)) 
            
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

            loss_dis_data = image_loss.item()
            loss_valid += loss_dis_data
            count += 1
            
        logloss_valid = metrics.log_loss(tol_label, tol_pred)
        auc_valid = metrics.roc_auc_score(tol_label, tol_pred)
        acc_valid = metrics.accuracy_score(tol_label, tol_pred_prob)
        fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
        eer_valid = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        loss_valid /= count

        pbar_str = "Valid: Epoch:{:>3d}  Avg_Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
            epoch_number, loss_valid, logloss_valid, acc_valid*100, auc_valid*100, eer_valid*100
        )
        logger.info(pbar_str)
    return loss_valid, logloss_valid, acc_valid, auc_valid, eer_valid

def train_EFF(
    trainLoader,
    model,
    epoch,
    epoch_number,
    optimizer,
    criterion_1,
    cfg,
    logger,
    **kwargs
):
    model.train()
    loss_train = 0
    loss_l1 = 0
    count = 0
    tol_label = np.array([], dtype=np.float) 
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)
    number_batch = len(trainLoader)

    for img_data, label in tqdm (trainLoader):
        img_label = label.numpy().astype(np.float)

        labels_data = label.cuda().float()
        img_data = img_data.cuda()
        
        classes = model(img_data)
        
        image_loss = criterion_1(classes, labels_data.view(-1, 1))
        
        loss = image_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        
        loss_dis_data = loss.item()
        loss_train += loss_dis_data
        count += 1

        if count % cfg.SHOW_STEP == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_L1loss:{:>5.3f}".format(
                epoch, count, number_batch, loss_dis_data, loss_l1_data
            )
            logger.info(pbar_str)

    logloss_train = metrics.log_loss(tol_label, tol_pred)
    auc_train = metrics.roc_auc_score(tol_label, tol_pred)
    acc_train = metrics.accuracy_score(tol_label, tol_pred_prob)
    fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
    eer_train = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    loss_train /= count
    
    pbar_str = "Train: Epoch:{:>3d}/{}  Avg_Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
        epoch, epoch_number, loss_train, logloss_train, acc_train*100, auc_train*100, eer_train*100
    )
    logger.info(pbar_str)
    return loss_train, logloss_train, acc_train, auc_train, eer_train


def valid_EFF(
    validLoader, epoch_number, model, cfg, criterion_1, logger, device, **kwargs
):
    model.eval()
    
    with torch.no_grad():
        loss_valid = 0
        count = 0
        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)
        tol_pred_prob = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(validLoader):
            img_label = labels_data.numpy().astype(np.float)

            img_data = img_data.cuda()
            labels_data = labels_data.cuda().float()

            classes = model(img_data)
        
            image_loss = criterion_1(classes, labels_data.view(-1, 1)) 
            
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

            loss_dis_data = image_loss.item()
            loss_valid += loss_dis_data
            count += 1
            
        logloss_valid = metrics.log_loss(tol_label, tol_pred)
        auc_valid = metrics.roc_auc_score(tol_label, tol_pred)
        acc_valid = metrics.accuracy_score(tol_label, tol_pred_prob)
        fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
        eer_valid = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        loss_valid /= count

        pbar_str = "Valid: Epoch:{:>3d}  Avg_Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
            epoch_number, loss_valid, logloss_valid, acc_valid*100, auc_valid*100, eer_valid*100
        )
        logger.info(pbar_str)
    return loss_valid, logloss_valid, acc_valid, auc_valid, eer_valid


def train_XCP_landmark_mask(
    train_iter,
    model,
    iter_num,
    max_iter,
    optimizer,
    criterion_1,
    criterion_2,
    cfg,
    logger,
    **kwargs
):
    model.train()
    loss_train = 0
    loss_l1 = 0
    count = 0
    tol_label = np.array([], dtype=np.float) 
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    label, img_data, landmark_mask2, landmark_mask5, mask_gt2, mask_gt5 = train_iter.next() 
    img_label = label.numpy().astype(np.float)

    labels_data = label.cuda().float()
    img_data = img_data.cuda()
    landmark_mask2 = landmark_mask2.cuda()
    landmark_mask5 = landmark_mask5.cuda()
    mask_gt2 = mask_gt2.cuda()
    mask_gt5 = mask_gt5.cuda()
    
    classes, mask2, mask5 = model(img_data, landmark_mask2, landmark_mask5)
    
    image_loss = criterion_1(classes, labels_data.view(-1, 1))
    mask_loss = cfg.TRAIN.LOSS_RATIO.MASK2 * criterion_2(mask2, mask_gt2) + cfg.TRAIN.LOSS_RATIO.MASK5 * criterion_2(mask5, mask_gt5)
    loss = image_loss + cfg.TRAIN.LOSS_RATIO.MASK * mask_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
    
    loss_l1_data = mask_loss.item()
    loss_dis_data = loss.item()
    loss_train += loss_dis_data
    loss_l1 += loss_l1_data
    count += 1

    # logloss_train = metrics.log_loss(tol_label, tol_pred)
    # auc_train = metrics.roc_auc_score(tol_label, tol_pred)
    acc_train = metrics.accuracy_score(tol_label, tol_pred_prob)
    # fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
    # eer_train = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    loss_train /= count
    loss_l1 /= count
    
    if iter_num % cfg.SHOW_STEP == 0:
        pbar_str = "-------Train_Batch: iter_num:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_L1Loss:{:>5.3f}  Acc:{:>5.4f}% ----------".format(
            iter_num, max_iter, loss_train, loss_l1, acc_train*100
        )
        logger.info(pbar_str)
    return loss_train, loss_l1, acc_train


def valid_XCP_landmark_mask(
    validLoader, epoch_number, model, cfg, criterion_1, logger, device, **kwargs
):
    model.eval()
    
    with torch.no_grad():
        loss_valid = 0
        count = 0
        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)
        tol_pred_prob = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(validLoader):
            img_label = labels_data.numpy().astype(np.float)

            img_data = img_data.cuda()
            labels_data = labels_data.cuda().float()

            classes, _, _ = model(img_data, None, None)
        
            image_loss = criterion_1(classes, labels_data.view(-1, 1)) 
            
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

            loss_dis_data = image_loss.item()
            loss_valid += loss_dis_data
            count += 1
            
        logloss_valid = metrics.log_loss(tol_label, tol_pred)
        auc_valid = metrics.roc_auc_score(tol_label, tol_pred)
        acc_valid = metrics.accuracy_score(tol_label, tol_pred_prob)
        fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
        eer_valid = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        loss_valid /= count

        pbar_str = "Valid: iter_num:{:>3d}  Avg_Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
            epoch_number, loss_valid, logloss_valid, acc_valid*100, auc_valid*100, eer_valid*100
        )
        logger.info(pbar_str)
    return loss_valid, logloss_valid, acc_valid, auc_valid, eer_valid

def train_XCP_landmark(
    train_iter,
    model,
    iter_num,
    max_iter,
    optimizer,
    criterion_1,
    cfg,
    logger,
    **kwargs
):
    model.train()
    loss_train = 0
    loss_l1 = 0
    count = 0
    tol_label = np.array([], dtype=np.float) 
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    label, img_data, landmark_mask2, landmark_mask5 = train_iter.next() 
    img_label = label.numpy().astype(np.float)

    labels_data = label.cuda().float()
    img_data = img_data.cuda()
    landmark_mask2 = landmark_mask2.cuda()
    landmark_mask5 = landmark_mask5.cuda()
    
    classes = model(img_data, landmark_mask2, landmark_mask5)
    
    image_loss = criterion_1(classes, labels_data.view(-1, 1))
    
    loss = image_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
    
    loss_dis_data = loss.item()
    loss_train += loss_dis_data
    count += 1

    # logloss_train = metrics.log_loss(tol_label, tol_pred)
    # auc_train = metrics.roc_auc_score(tol_label, tol_pred)
    acc_train = metrics.accuracy_score(tol_label, tol_pred_prob)
    # fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
    # eer_train = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    loss_train /= count
    
    if iter_num % cfg.SHOW_STEP == 0:
        pbar_str = "-------Train_Batch: iter_num:{:>3d}/{}  Batch_Loss:{:>5.3f}   Acc:{:>5.4f}% ----------".format(
            iter_num, max_iter, loss_train, acc_train*100
        )
        logger.info(pbar_str)
    return loss_train, acc_train

def valid_XCP_landmark(
    validLoader, epoch_number, model, cfg, criterion_1, logger, device, **kwargs
):
    model.eval()
    
    with torch.no_grad():
        loss_valid = 0
        count = 0
        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)
        tol_pred_prob = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(validLoader):
            img_label = labels_data.numpy().astype(np.float)

            img_data = img_data.cuda()
            labels_data = labels_data.cuda().float()

            classes = model(img_data, None, None)
        
            image_loss = criterion_1(classes, labels_data.view(-1, 1)) 
            
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

            loss_dis_data = image_loss.item()
            loss_valid += loss_dis_data
            count += 1
            
        logloss_valid = metrics.log_loss(tol_label, tol_pred)
        auc_valid = metrics.roc_auc_score(tol_label, tol_pred)
        acc_valid = metrics.accuracy_score(tol_label, tol_pred_prob)
        fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
        eer_valid = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        loss_valid /= count

        pbar_str = "Valid: iter_num:{:>3d}  Avg_Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
            epoch_number, loss_valid, logloss_valid, acc_valid*100, auc_valid*100, eer_valid*100
        )
        logger.info(pbar_str)
    return loss_valid, logloss_valid, acc_valid, auc_valid, eer_valid

def train_XCP_mask(
    train_iter,
    model,
    iter_num,
    max_iter,
    optimizer,
    criterion_1,
    criterion_2,
    cfg,
    logger,
    **kwargs
):
    model.train()
    loss_train = 0
    loss_l1 = 0
    count = 0
    tol_label = np.array([], dtype=np.float) 
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    img_data, label, mask_gt2, mask_gt5 = train_iter.next() 
    img_label = label.numpy().astype(np.float)

    labels_data = label.cuda().float()
    img_data = img_data.cuda()
    mask_gt2 = mask_gt2.cuda()
    mask_gt5 = mask_gt5.cuda()
    
    classes, mask2, mask5 = model(img_data)
    
    image_loss = criterion_1(classes, labels_data.view(-1, 1))
    mask_loss = cfg.TRAIN.LOSS_RATIO.MASK2 * criterion_2(mask2, mask_gt2) + cfg.TRAIN.LOSS_RATIO.MASK5 * criterion_2(mask5, mask_gt5)
    loss = image_loss + cfg.TRAIN.LOSS_RATIO.MASK * mask_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
    
    loss_l1_data = mask_loss.item()
    loss_dis_data = loss.item()
    loss_train += loss_dis_data
    loss_l1 += loss_l1_data
    count += 1

    # logloss_train = metrics.log_loss(tol_label, tol_pred)
    # auc_train = metrics.roc_auc_score(tol_label, tol_pred)
    acc_train = metrics.accuracy_score(tol_label, tol_pred_prob)
    # fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
    # eer_train = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    loss_train /= count
    loss_l1 /= count
    
    if iter_num % cfg.SHOW_STEP == 0:
        pbar_str = "-------Train_Batch: iter_num:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_L1Loss:{:>5.3f}  Acc:{:>5.4f}% ----------".format(
            iter_num, max_iter, loss_train, loss_l1, acc_train*100
        )
        logger.info(pbar_str)
    return loss_train, loss_l1, acc_train

def valid_XCP_mask(
    validLoader, epoch_number, model, cfg, criterion_1, logger, device, **kwargs
):
    model.eval()
    
    with torch.no_grad():
        loss_valid = 0
        count = 0
        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)
        tol_pred_prob = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(validLoader):
            img_label = labels_data.numpy().astype(np.float)

            img_data = img_data.cuda()
            labels_data = labels_data.cuda().float()

            classes, _, _ = model(img_data)
        
            image_loss = criterion_1(classes, labels_data.view(-1, 1)) 
            
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

            loss_dis_data = image_loss.item()
            loss_valid += loss_dis_data
            count += 1
            
        logloss_valid = metrics.log_loss(tol_label, tol_pred)
        auc_valid = metrics.roc_auc_score(tol_label, tol_pred)
        acc_valid = metrics.accuracy_score(tol_label, tol_pred_prob)
        fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
        eer_valid = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        loss_valid /= count

        pbar_str = "Valid: iter_num:{:>3d}  Avg_Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
            epoch_number, loss_valid, logloss_valid, acc_valid*100, auc_valid*100, eer_valid*100
        )
        logger.info(pbar_str)
    return loss_valid, logloss_valid, acc_valid, auc_valid, eer_valid

def train_XCP(
    train_iter,
    model,
    iter_num,
    max_iter,
    optimizer,
    criterion_1,
    cfg,
    logger,
    **kwargs
):
    model.train()
    loss_train = 0
    loss_l1 = 0
    count = 0
    tol_label = np.array([], dtype=np.float) 
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    img_data, label = train_iter.next() 
    img_label = label.numpy().astype(np.float)

    labels_data = label.cuda().float()#
    img_data = img_data.cuda()
    
    classes = model(img_data)
    
    image_loss = criterion_1(classes, labels_data.view(-1, 1))
    #image_loss = criterion_1(classes, labels_data.data)
    
    loss = image_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    classes = classes.view(1,-1).squeeze(0)
    classes = torch.sigmoid(classes)
    output_dis = classes.data.cpu().numpy()
    output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)
    for i in range(output_dis.shape[0]):
        if output_dis[i] >= 0.5:
            output_pred[i] = 1.0
        else:
            output_pred[i] = 0.0

    # output_dis = classes.data.cpu().numpy()
    # output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

    # for i in range(output_dis.shape[0]):
    #     if output_dis[i,1] >= output_dis[i,0]:
    #         output_pred[i] = 1.0
    #     else:
    #         output_pred[i] = 0.0
    
    tol_pred = np.concatenate((tol_pred, output_dis))
    tol_label = np.concatenate((tol_label, img_label))
    tol_pred_prob = np.concatenate((tol_pred_prob, output_pred))
    
    loss_dis_data = loss.item()
    loss_train += loss_dis_data
    count += 1

    logloss_train = metrics.log_loss(tol_label, tol_pred)
    auc_train = metrics.roc_auc_score(tol_label, tol_pred)
    acc_train = metrics.accuracy_score(tol_label, tol_pred_prob)
    fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
    eer_train = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    loss_train /= count
    
    if iter_num % cfg.SHOW_STEP == 0:
        pbar_str = "-------Train_Batch: iter_num:{:>3d}/{}  Batch_Loss:{:>5.3f}   Acc:{:>5.4f}% ----------".format(
            iter_num, max_iter, loss_train, acc_train*100
        )
        logger.info(pbar_str)
    return loss_train, acc_train


def valid_XCP(
    validLoader, epoch_number, model, cfg, criterion_1, logger, device, **kwargs
):
    model.eval()
    
    with torch.no_grad():
        loss_valid = 0
        count = 0
        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)
        tol_pred_prob = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(validLoader):
            img_label = labels_data.numpy().astype(np.float)

            img_data = img_data.cuda()
            labels_data = labels_data.cuda().float()#

            classes = model(img_data)
        
            image_loss = criterion_1(classes, labels_data.view(-1, 1)) 
            #image_loss = criterion_1(classes, labels_data.data)
            
            classes = classes.view(1,-1).squeeze(0)
            classes = torch.sigmoid(classes)
            output_dis = classes.data.cpu().numpy()
            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)
            for i in range(output_dis.shape[0]):
                if output_dis[i] >= 0.5:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            # output_dis = classes.data.cpu().numpy()
            # output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

            # for i in range(output_dis.shape[0]):
            #     if output_dis[i,1] >= output_dis[i,0]:
            #         output_pred[i] = 1.0
            #     else:
            #         output_pred[i] = 0.0

            tol_pred = np.concatenate((tol_pred, output_dis))
            tol_label = np.concatenate((tol_label, img_label))
            tol_pred_prob = np.concatenate((tol_pred_prob, output_pred))

            loss_dis_data = image_loss.item()
            loss_valid += loss_dis_data
            count += 1
            
        #logloss_valid = metrics.log_loss(tol_label, tol_pred)
        #auc_valid = metrics.roc_auc_score(tol_label, tol_pred)
        acc_valid = metrics.accuracy_score(tol_label, tol_pred_prob)
        # fpr, tpr, _ = metrics.roc_curve(tol_label, tol_pred, pos_label=1)
        # eer_valid = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        loss_valid /= count

        # pbar_str = "Valid: iter_num:{:>3d}  Avg_Loss:{:>5.3f}  logloss:{:>5.4f}  Acc:{:>5.4f}%  Auc:{:>5.4f}%  EER:{:>5.4f}%".format(
        #     epoch_number, loss_valid, logloss_valid, acc_valid*100, auc_valid*100, eer_valid*100
        # )
        # logger.info(pbar_str)
        pbar_str = "Valid: iter_num:{:>3d}  Avg_Loss:{:>5.3f}  Acc:{:>5.4f}% ".format(
            epoch_number, loss_valid, acc_valid*100
        )
        logger.info(pbar_str)
    # return loss_valid, logloss_valid, acc_valid, auc_valid, eer_valid
    return loss_valid, acc_valid