from model import get_model
from sklearn.model_selection import train_test_split
import gc
from network.model_repository import Resnet18_8s
from torch.nn import DataParallel
from torch.nn import DataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn, optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import lr_scheduler
import random
# 自己写的数据集类
import pandas as pd
from dataloader import CAR as CarDataset
from train import train,eval
from config import cfg
import gc


cfg = cfg().parse()
cfg.cfg = 'cfg.yaml'
cfg.exp_mod = 'test'
cfg.dataset = 'lmo'
cfg.object = 'driller'




def _worker_init_fn():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2 ** 32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)

def adjust_learning_rate(optimizer, epoch, lr_decay_rate, lr_decay_epoch, min_lr=1e-5):
    if ((epoch + 1) % lr_decay_epoch) != 0:
        return

    for param_group in optimizer.param_groups:
        # print(param_group)
        lr_before = param_group['lr']
        param_group['lr'] = param_group['lr'] * lr_decay_rate
        param_group['lr'] = max(param_group['lr'], min_lr)
    print('changing learning rate {:5f} to {:.5f}'.format(lr_before, max(param_group['lr'], min_lr)))


if __name__ == "__main__":

    path_csv = 'D:\\jieya\\Pku-cdpn\\train_data.csv'
    path_img = 'D:\\jieya\\pku_test\\train_images'
    path_mask = 'D:\\jieya\\pku_test\\cdpn\\dataset\\save_box_mask\\mask'

    # 问题是，我的训练数据既没有render也没有fuse，还有一些亮度、对比度、色调的问题，这个等以后再解决
    # 数据增强还有个坑
    # 先写kaggle版本的

    # 注意数据的读取，要用到values
    # train和test就是单纯按顺序划分的
    # 这里的shuffle先不用，后面用torch加载的时候再用
    data = pd.read_csv(path_csv)
    df_train, df_val = train_test_split(data,shuffle=False,test_size=0.01,random_state=23)

    # 因为一开始不动怎么操作，所以还要重新写
    # 第一次写，难免出现这种状况
    # 如何避免？按着顺序来
    train_dataset = CarDataset(df_train,path_img=path_img,path_mask=path_mask)
    val_dataset = CarDataset(df_val,path_img=path_img,path_mask=path_mask)

    # 用pytorch对数据集处理了一遍，应该就可以了
    train_dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=0,worker_init_fn=_worker_init_fn())
    val_dataloader = DataLoader(val_dataset,batch_size=4,shuffle=True,num_workers=0,worker_init_fn=_worker_init_fn())



    # 简单model的准备工作
    net = Resnet18_8s(ver_dim= 195, seg_dim=2, inp_dim=3)

    # 这个也不好使

    # 有了这个多并行cuda，应该就不用.to(device)了吧
    net=DataParallel(net).cuda(cfg.pytorch.gpu)
    n_epochs = 100
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=(max(n_epochs, 10) * len(train_dataloader) // 3), gamma=0.1)

    history = pd.DataFrame()

    # 现在，有一些配置文件，我可以不使用这些配置文件，只是利用配置文件，当成一些基本信息
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        # 使用的是Pvnet的学习率调整方法
        adjust_learning_rate(optimizer, epoch, 0.5, 20)

        # train 输入： 模型，优化器，数据集，epoch次数(epoch次数应该不会有什么影响吧)
        train(optimizer=optimizer,model=net,cfg=cfg,dataloader=train_dataloader,epoch = epoch,history = history)
        #val(net)
        eval(optimizer=optimizer, model=net, cfg=cfg, dataloader=val_dataloader, epoch=epoch, history=history)

        if epoch % 10 == 0:
            torch.save(net.state_dict(), './{}_model.pth'.format(epoch))


    history.to_csv('history.csv')
    history['train_loss'].iloc[100:].plot();
    series = history.dropna()['dev_loss']
    plt.scatter(series.index, series);