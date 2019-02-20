"""
validation
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

dice_list = []
time_list = []

# 定义网络并加载参数
# net = torch.nn.DataParallel(DialResUNet.DialResUNet(training=False)).cuda()
# net.load_state_dict(torch.load(module_dir))
# net.eval()

# 加载模型进行测试
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

    while end_slice <= ct_array.shape[0] - 1:
        ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])

        start_slice = end_slice + 1
        end_slice = start_slice + size - 1
    # #
    # # 当无法整除的时候反向取最后一个block
    if end_slice is not ct_array.shape[0] - 1 + size:
        flag = True
        count = ct_array.shape[0] - start_slice  # 这个要么是0要么是
        ct_array_list.append(ct_array[-size:, :, :])  # 再加最后size个

    outputs_list = []
    with torch.no_grad():
        for ct_array in ct_array_list:
            # try:

            ct_tensor = torch.FloatTensor(ct_array).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0)  # 增加维度
            ct_tensor = ct_tensor.unsqueeze(dim=0)

            outputs = net(ct_tensor)  # 预测输出
            # outputs = F.softmax(outputs, dim=1)
            # outputs = outputs[:, 1, :, :, :]
            outputs = outputs.squeeze()

            # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
            outputs_list.append(outputs.cpu().detach().numpy())
            del outputs
            # except:
            #     pass

    # 执行完之后开始拼接结果

    pred_seg = np.concatenate(outputs_list[0:-1], axis=0)
    if flag is False:  # 整除
        pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=0)
    else:  # 没有整除
        pred_seg = np.concatenate([pred_seg, outputs_list[-1][-count:]], axis=0)

    # 将金标准读入内存来计算dice系数
    seg = sitk.ReadImage('/home/yanxin/data_ct/' + str(i) + 'Venous_tra_5mm_roi.nii', sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    # seg_array[seg_array > 0] = 1

    # 使用线性插值将预测的分割结果缩放到原始nii大小
    # pred_seg = torch.FloatTensor(pred_seg).unsqueeze(dim=0).unsqueeze(dim=0)
    # pred_seg = F.upsample(pred_seg, seg_array.shape, mode='trilinear').squeeze().detach().numpy()
    pred_seg = (pred_seg > threshold).astype(np.int16)

    # # 先进行腐蚀 不一定要
    # pred_seg = sm.binary_erosion(pred_seg, sm.ball(5))

    # 取三维最大连通域，移除小区域
    # # 这个也不一定要
    # pred_seg = measure.label(pred_seg, 4)
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
    # pred_seg = pred_seg.astype(np.uint8)
    #
    # # # 进行膨胀恢复之前的大小
    # pred_seg = sm.binary_dilation(pred_seg, sm.ball(5))
    # pred_seg = pred_seg.astype(np.uint8)

    print('size of pred: ', pred_seg.shape)
    print('size of GT: ', seg_array.shape)
    try:

        dice = (2 * pred_seg * seg_array).sum() / (pred_seg.sum() + seg_array.sum())
        dice_list.append(dice)

        print('dice: {:.3f}'.format(dice))
    except:
        pass
    # 将预测的结果保存为nii数据
    # pred_seg = sitk.GetImageFromArray(pred_seg)
    #
    # pred_seg.SetDirection(ct.GetDirection())
    # pred_seg.SetOrigin(ct.GetOrigin())
    # pred_seg.SetSpacing(ct.GetSpacing())

    # sitk.WriteImage(pred_seg, os.path.join(liver_pred_dir, file.replace('volume', 'pred')))
    # del pred_seg

    casetime = time() - start
    time_list.append(casetime)


    print('this case use {:.3f} s'.format(casetime))
    print('-----------------------')

# 输出整个测试集的平均dice系数和平均处理时间
print('dice per case: {}'.format(sum(dice_list) / len(dice_list)))
print('time per case: {}'.format(sum(time_list) / len(time_list)))


