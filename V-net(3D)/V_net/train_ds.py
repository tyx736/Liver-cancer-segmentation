from time import time
import os
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from loss.Dice_loss import DiceLoss

# 看具体情况用哪一个NET

# from net.DialResUNet import net
from net.simpleVnet import net
from dataset.dataset_random import train_ds, val_ds  # 继 承的dataset
from medicaltorch import losses as mt_losses

# 定义超参数
# on_server = True

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cudnn.benchmark = True
Epoch = 3000
leaing_rate_base = 2 * 1e-5
alpha = 0.0

batch_size = 4

num_workers = 0
pin_memory = True
# net = torch.nn.DataParallel(net).cuda()#下次这里指定GPU
device_ids = [0, 1]

# model = net.cuda()
model = net.cuda(device_ids[0])
model = torch.nn.DataParallel(model, device_ids=device_ids)
net.train()

# 定义数据加载
train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)
val_dl = DataLoader(val_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

# 定义损失函数
loss_func = DiceLoss()

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate_base)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [1500])  # 默认0.1衰减

# 训练网络
start = time()
for epoch in tqdm(range(1, Epoch + 1)):
    # for epoch in range(Epoch):

    lr_decay.step()

    mean_loss = []

    for step1, (ct, seg) in enumerate(train_dl):
        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)
        # 注释掉的是densenet的结构，有4个损失

        # loss1 = loss_func(outputs[0], seg)
        # loss2 = loss_func(outputs[1], seg)
        # loss3 = loss_func(outputs[2], seg)
        # loss4 = loss_func(outputs[3], seg)
        #
        # loss = (loss1 + loss2 + loss3) * alpha + loss4
        loss4 = loss_func(outputs, seg)
        mean_loss.append(loss4.item())

        opt.zero_grad()
        loss4.backward()
        opt.step()
    val_loss_total = []
    for step2, (ct, seg) in enumerate(val_dl):
        with torch.no_grad():
            ct = ct.cuda()
            seg = seg.cuda()

            outputs = net(ct)
            loss2 = loss_func(outputs, seg)
            val_loss_total.append(loss2.item())

        # # Metrics computation
    #     gt_npy = gt_samples.numpy().astype(np.uint8)
    #     gt_npy = gt_npy.squeeze(axis=1)
    #
    #     preds = preds.data.cpu().numpy()
    #     preds = threshold_predictions(preds)
    #     preds = preds.astype(np.uint8)
    #     preds = preds.squeeze(axis=1)
    #
    #     metric_mgr(preds, gt_npy)
    #
    #     num_steps += 1
    #
    # metrics_dict = metric_mgr.get_results()
    # metric_mgr.reset()
    val_loss_total_avg = sum(val_loss_total) / (len(val_loss_total))

    # if step % 20 is 0:
    #     print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
    #           .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))

    mean_loss = sum(mean_loss) / len(mean_loss)
    print('epoch:', epoch, '    mean_loss:', mean_loss - 1, 'val_loss_total_avg:', val_loss_total_avg - 1)
    with open('score_Unet_simple.txt', 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
        f.write('train:' + str(mean_loss - 1) + 'test' + str(val_loss_total_avg - 1) + '\n')
    # print()

    if epoch % 10 is 0 and epoch is not 0:
        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        torch.save(net, 'U_model_stride_1_all_simpleVnet' + str(epoch) + '.pkl')
        print('model saved')
        # torch.save(net.state_dict(), '/home/yanxin/DenseV_net/DenseVnet{}.pth'.format(epoch))
        # print('model saved')

    if epoch % 15 is 0 and epoch is not 0:
        alpha *= 0.8

# 深度监督的系数变化
# 1.000
# 0.800
# 0.640
# 0.512
# 0.410
# 0.328
# 0.262
# 0.210
# 0.168
# 0.134
# 0.107
# 0.086
# 0.069
# 0.055
# 0.044
# 0.035
# 0.028
# 0.023
# 0.018
# 0.014
# 0.012
# 0.009
# 0.007
# 0.006
# 0.005
# 0.004
# 0.003
# 0.002
# 0.002
# 0.002
# 0.001
# 0.001
# 0.001
# 0.001
# 0.001
# 0.000
# 0.000
