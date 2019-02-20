"""

现在我们要做 的 就是  对 每一个CT的像素点 进行Vote

"""

import os
from time import time

import torch
import torch.nn.functional as F

import numpy as np
import xlsxwriter as xw
import SimpleITK as sitk
import scipy.ndimage as ndimage
import skimage.morphology as sm
import skimage.measure as measure

import net.DialResUNet as DialResUNet
from net import DialResUNet

# 这个直接用原始的测试


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#
# val_ct_dir = '/home/yanxin/sorted_data_16/test/input/'
#
# val_seg_dir = '/home/yanxin/sorted_data_16/test/gt/'
#
# liver_pred_dir = '/home/yanxin/sorted_data_16/DenseVnet/pred/'
print('model_name:')
model_name = input()

module_dir = model_name

upper = 200
lower = -200
down_scale = 0.5
size = 16
slice_thickness = 5
threshold = 0.5  # 这个看来也是一个参数,应该无所谓
print('ball size:')
ballsize = input()
print('save it?(y/n)')
save=input()
dice_mean_list = []
dice_min_list = []
dice_max_list = []
dice_median_list = []

time_list = []

net = torch.load('U_model_stride_1_all_simpleVnet' + model_name + '.pkl')

for i in range(120, 150):
    # for file_index, file in enumerate(os.listdir(val_ct_dir)):
    start = time()
    # 将CT读入内存，读val的
    ct = sitk.ReadImage('/home/yanxin/data_ct/' + str(i) + 'Venous_tra_5mm.nii', sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    origin_shape = ct_array.shape

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    # 对CT使用双三次算法进行插值，插值之后的array依然是int16
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)

    # 在轴向上进行切块取样,就是int(n/16)或者int(n/16)+1
    flag = False
    start_slice = 0
    end_slice = start_slice + size - 1
    ct_array_list = []
    start_end_slice_list = []
    # start_end_slice_list.append((start_slice,end_slice))

    while end_slice <= ct_array.shape[0] - 1:
        ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])
        start_end_slice_list.append((start_slice, end_slice))
        start_slice += 1
        end_slice = start_slice + size - 1


    outputs_list = []
    with torch.no_grad():
        for ct_array_16 in ct_array_list:
            # try:

            ct_tensor = torch.FloatTensor(ct_array_16).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0)  # 增加维度
            ct_tensor = ct_tensor.unsqueeze(dim=0)

            outputs = net(ct_tensor)  # 预测输出
            # outputs = F.softmax(outputs, dim=1)
            # outputs = outputs[:, 1, :, :, :]
            outputs = outputs.squeeze()  # 16*256*256/512*512

            # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
            outputs_list.append(outputs.cpu().detach().numpy())  # 全放进outputlist
            del outputs
            # except:
            #     pass
    # 在这里建立两个列表，分别是start_还有end_,前面做好了
    # 在这里开始重新组合
    all_pred_slice_max=[]
    all_pred_slice_min = []
    all_pred_slice_mean = []
    all_pred_slice_median = []
    # 这个非常重要，是 所有 预测到的 切片
    for ct_slice in range(ct_array.shape[0]):
        # print(ct_slice) # 23
        pred_slice=[]
        for s in start_end_slice_list:
            # print(s) # 23 38
            if ct_slice>=s[0] and ct_slice<=s[1]:
                # print('chosed')
                # print('len',len(outputs_list)) # 23
                # print(ct_array.shape[0]) # 38 对 38-15=23
                # print(len(start_end_slice_list)) # 这个应该也是23
                # print(start_end_slice_list.index(s))
                # print(outputs_list[start_end_slice_list.index(s)].shape)
                pred_slice.append(outputs_list[start_end_slice_list.index(s)][ct_slice-s[0],:,:])
        pred_seg = np.array(pred_slice)
        # print(pred_seg.shape)
        # print(np.max(pred_seg))
        # print(np.min(pred_seg))
        # print(np.mean(pred_seg))
        # print(np.median(pred_seg))
        pred_max=np.max(pred_seg,0)
        pred_min = np.min(pred_seg, 0)
        pred_mean = np.mean(pred_seg, 0)
        pred_median = np.median(pred_seg, 0)
        all_pred_slice_max.append(pred_max)
        all_pred_slice_min.append(pred_min)
        all_pred_slice_mean.append(pred_mean)
        all_pred_slice_median.append(pred_median)

                # all_pred_slice.append(outputs_list[start_end_slice_list.index(s)][ct_slice-s[0]+start_end_slice_list.index(s)])




    # pred_seg = np.concatenate(outputs_list[0:-1], axis=0)
    # if flag is False:  # 整除
    #     pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=0)
    # else:  # 没有整除
    #     pred_seg = np.concatenate([pred_seg, outputs_list[-1][-count:]], axis=0)

    # 将金标准读入内存来计算dice系数
    seg = sitk.ReadImage('/home/yanxin/data_ct/' + str(i) + 'Venous_tra_5mm_roi.nii', sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    # seg_array[seg_array > 0] = 1

    # 使用线性插值将预测的分割结果缩放到原始nii大小
    # pred_seg = torch.FloatTensor(pred_seg).unsqueeze(dim=0).unsqueeze(dim=0)

    # pred_seg = F.upsample(pred_seg, seg_array.shape, mode='trilinear').squeeze().detach().numpy()
    # all_pred_slice_mean=np.array(all_pred_slice_mean)
    # print(all_pred_slice_mean.shape)
    pred_seg_mean = (np.array(all_pred_slice_mean)> threshold).astype(np.int16)
    pred_seg_max = (np.array(all_pred_slice_max) > threshold).astype(np.int16)
    pred_seg_min = (np.array(all_pred_slice_min) > threshold).astype(np.int16)
    pred_seg_median = (np.array(all_pred_slice_median) > threshold).astype(np.int16)

    # 先进行腐蚀 不一定要
    # 将0值扩充到邻近像素。扩大黑色部分，减小白色部分。可用来提取骨干信息，去掉毛刺，去掉孤立的像素。
    # pred_seg = sm.binary_erosion(pred_seg_max, sm.ball(5))

    # 取三维最大连通域，移除小区域
    # 这个也不一定要
    # 这个不太稳定，有时候能改善，有时候不能
    # pred_seg = measure.label(pred_seg_max, 4)
    # props = measure.regionprops(pred_seg)
    #
    # max_area = 0
    # max_index = 0
    # for index, prop in enumerate(props, start=1):
    #     if prop.area > max_area:
    #         max_area = prop.area
    #         max_index = index
    #
    # pred_seg[pred_seg != max_index] = 0
    # pred_seg[pred_seg == max_index] = 1
    #
    # pred_seg_max = pred_seg.astype(np.uint8)

    # # 进行膨胀恢复之前的大小
    # 找到像素值为1的点，将它的邻近像素点都设置成这个值。一般用来扩充边缘或填充小的孔洞。

    pred_seg_max = sm.binary_dilation(pred_seg_max, sm.ball(int(ballsize)))
    pred_seg_max = pred_seg_max.astype(np.uint8)

    print('size of pred: ', pred_seg_max.shape)
    print('size of GT: ', seg_array.shape)

    dice_mean = (2 * pred_seg_mean * seg_array).sum() / (pred_seg_mean.sum() + seg_array.sum())
    dice_mean_list.append(dice_mean)
    dice_min = (2 * pred_seg_min * seg_array).sum() / (pred_seg_min.sum() + seg_array.sum())
    dice_min_list.append(dice_min)
    dice_max = (2 * pred_seg_max * seg_array).sum() / (pred_seg_max.sum() + seg_array.sum())
    dice_max_list.append(dice_max)
    dice_median = (2 * pred_seg_median * seg_array).sum() / (pred_seg_median.sum() + seg_array.sum())
    dice_median_list.append(dice_median)

    print('dice_mean: {:.3f}'.format(dice_mean))
    print('dice_min: {:.3f}'.format(dice_min))
    print('dice_max: {:.3f}'.format(dice_max))
    print('dice_median: {:.3f}'.format(dice_median))

    if save=='y':

        # 将预测的结果保存为nii数据
        pred_seg_max = sitk.GetImageFromArray(pred_seg_max)

        pred_seg_max.SetDirection(ct.GetDirection())
        pred_seg_max.SetOrigin(ct.GetOrigin())
        pred_seg_max.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(pred_seg_max, '/home/yanxin/DenseV_net/pred/'+ str(i) + 'Venous_tra_5mm_pred.nii')
        del pred_seg_max

    casetime = time() - start
    time_list.append(casetime)

    # worksheet.write(2, file_index + 1, casetime)

    print('this case use {:.3f} s'.format(casetime))
    print('-----------------------')

# 输出整个测试集的平均dice系数和平均处理时间
print('dice mean per case: {}'.format(sum(dice_mean_list) / len(dice_mean_list)))
print('dice max per case: {}'.format(sum(dice_max_list) / len(dice_max_list)))
print('dice min per case: {}'.format(sum(dice_min_list) / len(dice_mean_list)))
print('dice median per case: {}'.format(sum(dice_median_list) / len(dice_mean_list)))
# print(dice_max_list)
print('time per case: {}'.format(sum(time_list) / len(time_list)))
print(dice_max_list)

