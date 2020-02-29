from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import random
import time
import datetime
import torch
import torch.utils.data
import ref
import pprint
import cv2

cv2.ocl.setUseOpenCL(False)
from model import get_model
from utils import get_ply_model
from test import test
import json
from config import cfg


# test dataset
from dataloader_detected_bbox import LMO as Dataset

model_dir = ref.lmo_model_dir
cfg = cfg().parse()

def main():
    model = get_model(cfg)
    if cfg.pytorch.gpu > -1:
        # print('Using GPU{}'.format(cfg.pytorch.gpu))
        model = model.cuda(cfg.pytorch.gpu)


    ## load test set
    # 这个还不知道有啥用，多线程？
    def _worker_init_fn():
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)

    test_loader = torch.utils.data.DataLoader(
        Dataset(cfg, 'test'),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=_worker_init_fn()
    )

    ## test
    # 都用的绝对路径
    model_dir = 'D:\jieya\pku_test\cdpn\models_eval'
    # 注意要先读取，才能load，如果不读取会发生什么呢？
    # 这是每个模型的信息，最大尺寸，坐标等
    with open('D:\jieya\pku_test\cdpn\models_eval\models_info.json', 'r') as f_model_eval:
        models_info = json.load(f_model_eval)
        # 将字符型的key转换成int型的key
        for k in list(models_info.keys()):
            models_info[int(k)] = models_info.pop(k)

    models_vtx = {}
    for obj_name in [cfg.pytorch.object]:

        if cfg.pytorch.dataset.lower() == 'lmo':
            obj_id = ref.lmo_obj2id(obj_name)

        # 右对齐，共六位
        models_vtx[obj_name] = get_ply_model(os.path.join(model_dir, 'obj_{:06d}.ply'.format(obj_id)))

    print('testload00',test_loader)
    test(cfg, test_loader, model, models_info, models_vtx)

if __name__ == '__main__':
    main()