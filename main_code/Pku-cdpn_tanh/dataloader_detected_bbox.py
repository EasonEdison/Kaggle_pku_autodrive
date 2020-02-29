# encoding: utf-8
'''
@author: Zhigang Li
@license: (C) Copyright.
@contact: aaalizhigang@163.com
@software: Pose6D
@file: LineMOD.py
@time: 18-10-24 下午10:24
@desc: load LineMOD dataset
'''

import torch.utils.data as data
import numpy as np
import ref
import cv2
import os, sys
import pickle
import json
from utils import *
import pickle as pkl


class LMO(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.root_dir = 'D:\jieya\pku_test\cdpn\dataset'
        # print('==> initializing {} {} data.'.format(cfg.pytorch.dataset, split))
        ## load dataset
        annot = []
        # root_dir  = dataset / lmo_bop19
        print('self root',self.root_dir)
        te_obj_dir = os.path.join(self.root_dir, 'test')
        # 就当是有很多框了
        f_bbox = os.path.join(ref.bbox_dir, 'lmo', 'lmo_000002_detection_result.json')
        with open(f_bbox, 'r') as f:
            annot_bbox = json.load(f)
        # merge annots
        for k in annot_bbox.keys():       
            score = {}
            for i in range(len(annot_bbox[k])):
                if annot_bbox[k][i]['obj_id'] not in score.keys():
                    score[annot_bbox[k][i]['obj_id']] = []
                score[annot_bbox[k][i]['obj_id']].append(annot_bbox[k][i]['score'])         
            for l in range(len(annot_bbox[k])):
                annot_temp = {}
                annot_temp['obj_id']         = annot_bbox[k][l]['obj_id']
                if ref.lmo_id2obj[annot_temp['obj_id']] not in [cfg.pytorch.object]:
                    continue
                if annot_bbox[k][l]['score'] == max(score[annot_temp['obj_id']]) and max(score[annot_temp['obj_id']]) > 0.2:
                    annot_temp['bbox'] = annot_bbox[k][l]['bbox']
                    annot_temp['score'] = annot_bbox[k][l]['score']
                else:
                    continue
                annot_temp['rgb_pth']        = os.path.join(te_obj_dir, 'rgb', '{:06d}.png'.format(int(k)))   
                annot_temp['scene_id'] = 2
                annot_temp['image_id'] = int(k)                             
                annot.append(annot_temp)
            break
        self.annot = annot
        self.nSamples = len(annot)
        # print('Loaded LineMOD {} {} samples'.format(split, self.nSamples))


    def GetPartInfo(self, index):
        """
        Get infos ProjEmbCrop, DepthCrop, box, mask, c, s  from index
        :param index:
        :return:
        """
        cls_idx = self.annot[index]['obj_id']
        rgb = cv2.imread(self.annot[index]['rgb_pth'])
        # print('rgb size:',np.shape(rgb))
        box = self.annot[index]['bbox'].copy()  # bbox format: [left, upper, width, height]

        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        print('c',c)
        scale = 1.5
        s = max(box[3], box[2]) * scale
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        # 注意，这里可能有些问题，其实是先扩大了，然后再resize，对mask也再做一遍？
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        # print('rgb_crop size:',np.shape(rgb_crop))

        return cls_idx, rgb_crop, box, c, s,scale

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s,scale = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        pose = np.zeros((3, 4))
        center = c
        size = s
        box = np.asarray(box)
        imgPath = self.annot[index]['rgb_pth']
        scene_id = self.annot[index]['scene_id']
        image_id = self.annot[index]['image_id']
        score = self.annot[index]['score']
        return inp, pose, box, center, size, cls_idx, imgPath, scene_id, image_id, score,scale


    def __len__(self):
        return self.nSamples

