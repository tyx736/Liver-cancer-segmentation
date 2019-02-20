# coding: utf-8

import os
import torch
import numpy as np
from collections import defaultdict
import time

from tqdm import tqdm

from medicaltorch import datasets as mt_datasets
from medicaltorch import models as mt_models
from medicaltorch import transforms as mt_transforms
from medicaltorch import losses as mt_losses
from medicaltorch import metrics as mt_metrics
from medicaltorch import filters as mt_filters

from torchvision import transforms
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torchvision.utils as vutils

import matplotlib.pyplot as plt

cudnn.benchmark = True

os.getcwd()  # '/home/yanxin'

def plot_2_pic(i):  # 对某个人画图
    img = nib.load('./data_ct/' + str(i) + 'Venous_tra_5mm.nii')
    img_arr = img.get_fdata()
    img_gt = nib.load('./data_ct/' + str(i) + 'Venous_tra_5mm_roi.nii')
    img_arr_gt = img_gt.get_fdata()
    for i in range(img_arr.shape[-1]):
        plt.subplot(1, 2, 1)
        plt.imshow(img_arr[:, :, i], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(img_arr_gt[:, :, i], cmap='gray')
        plt.show()

# Normalize标准化
train_transform = transforms.Compose([
    # mt_transforms.Resample(1.6, 1.6),
    # mt_transforms.CenterCrop2D((256, 256)),
    #         mt_transforms.ElasticTransform(alpha_range=(40.0, 60.0),
    #                                        sigma_range=(2.5, 4.0),
    #                                        p=0.3),#弹性形变
    #         mt_transforms.RandomAffine(degrees=4.6,
    #                                    scale=(0.98, 1.02),
    #                                    translate=(0.03, 0.03)),#随机仿射变换
    #         mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),#？？？？随机偏移？
    #         transforms.Resize(256),
    mt_transforms.ToTensor(),
    #         transforms.Resize(256),
    mt_transforms.NormalizeInstance(),
])

val_transform = transforms.Compose([
    # mt_transforms.Resample(1.6, 1.6),
    # mt_transforms.CenterCrop2D((256, 256)),
    mt_transforms.ToTensor(),
    #         transforms.Resize(256),
    mt_transforms.NormalizeInstance(),
])

data_list = os.listdir('/home/yanxin/sorted_data_16')
train_pair_tuple = []
test_pair_tuple = []
for i in range(120):
    for j in [0,1,2]:
        if str(i)+'input_index'+str(j)+'.nii' in data_list:
            train_pair_tuple.append(('/home/yanxin/sorted_data_16/'+str(i)+'input_index'+str(j)+'.nii', '/home/yanxin/sorted_data_16/'+str(i)+'gt_index'+str(j)+'.nii'))
for i in range(120, 150):
    for j in [0,1,2]:
        if str(i)+'input_index'+str(j)+'.nii' in data_list:
            test_pair_tuple.append(('/home/yanxin/sorted_data_16/'+str(i)+'input_index'+str(j)+'.nii', '/home/yanxin/sorted_data_16/'+str(i)+'gt_index'+str(j)+'.nii'))


# # load data

train_dataset = mt_datasets.MRI2DSegmentationDataset(train_pair_tuple, transform=train_transform)
print(len(train_dataset))
test_dataset = mt_datasets.MRI2DSegmentationDataset(test_pair_tuple, transform=val_transform)
print(len(test_dataset))


# 没有打乱顺序shuffle=False， batch_size=16，正好为1个切片
train_loader = DataLoader(train_dataset, batch_size=16,
                          shuffle=False, pin_memory=True,
                          collate_fn=mt_datasets.mt_collate)

val_loader = DataLoader(test_dataset, batch_size=16,
                        shuffle=False, pin_memory=True,
                        collate_fn=mt_datasets.mt_collate)

# 预测函数，最后>0.999判为1，病变区域
def threshold_predictions(predictions, thr=0.999):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


# try:
#     model = torch.load('/home/yanxin/data_ct/U_model.pkl')
# except:
#     print('no model,creating')

# 调参，drop_rate与BN
model = mt_models.Unet(drop_rate=0.4, bn_momentum=0.1)
device_ids = [0, 1]
model = model.cuda()
model = nn.DataParallel(model, device_ids=device_ids)

# 训练轮数、学习率、优化方式
num_epochs = 2000
initial_lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)


def numeric_score(prediction, groundtruth):
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN


def accuracy(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0


for epoch in tqdm(range(1, num_epochs + 1)):

    start_time = time.time()

    scheduler.step()  # 每个eopch调节一下学习率

    lr = scheduler.get_lr()[0]

    model.train()
    train_loss_total = 0.0
    num_steps = 0

    ### Training
    for i, batch in enumerate(train_loader):
        input_samples, gt_samples = batch["input"], batch["gt"]

        var_input = input_samples.cuda()
        var_gt = gt_samples.cuda()

        preds = model(var_input)

        loss = mt_losses.dice_loss(preds, var_gt)
        train_loss_total += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_steps += 1

    train_loss_total_avg = train_loss_total / (len(train_dataset) / 16)
    model.eval()
    val_loss_total = 0.0
    num_steps = 0
    train_acc = accuracy(preds.cpu().detach().numpy(),
                         var_gt.cpu().detach().numpy())

    metric_fns = [mt_metrics.dice_score,
                  mt_metrics.hausdorff_score,
                  mt_metrics.precision_score,
                  mt_metrics.recall_score,
                  mt_metrics.specificity_score,
                  mt_metrics.intersection_over_union,
                  mt_metrics.accuracy_score]

    metric_mgr = mt_metrics.MetricManager(metric_fns)

    ### Validating
    for i, batch in enumerate(val_loader):
        input_samples, gt_samples = batch["input"], batch["gt"]

        with torch.no_grad():
            var_input = input_samples.cuda()
            var_gt = gt_samples.cuda()

            preds = model(var_input)
            loss = mt_losses.dice_loss(preds, var_gt)
            val_loss_total += loss.item()

        # Metrics computation
        gt_npy = gt_samples.numpy().astype(np.uint8)
        gt_npy = gt_npy.squeeze(axis=1)

        preds = preds.data.cpu().numpy()
        preds = threshold_predictions(preds)
        preds = preds.astype(np.uint8)
        preds = preds.squeeze(axis=1)

        metric_mgr(preds, gt_npy)

        num_steps += 1

    metrics_dict = metric_mgr.get_results()
    metric_mgr.reset()
    val_loss_total_avg = val_loss_total / (len(test_dataset) / 16)

    print('\nTrain loss: {:.4f}'.format(train_loss_total_avg))
    print('Val Loss: {:.4f}'.format(val_loss_total_avg))
    with open('score_Unet_size16.txt', 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
        f.write('train:'+str(train_loss_total_avg)+'test'+str(val_loss_total_avg)+'\n')
    # 每过50轮保存一下模型
    if epoch % 50 == 0:
        torch.save(model, 'U_model_size16.pkl')
        print('model saved')
