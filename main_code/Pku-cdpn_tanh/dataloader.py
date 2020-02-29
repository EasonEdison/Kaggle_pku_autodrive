import torch.utils.data as data
import numpy as np
import ref
import cv2
import os, sys
import pickle
import json
from utils import *
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from math import cos ,sin

def str2coords(s):
    b = s.split(', ')
    b[0] = b[0][1:]
    b[-1] = b[-1][:-1]
    bbox = [float(b[0]), float(b[1]), float(b[2])-float(b[0]), float(b[3])-float(b[1])]
    xyz = [float(b[4]), float(b[5]), float(b[6])]
    pyr = [float(b[7]), float(b[8]), float(b[8])]

    return bbox, xyz, pyr

def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


class CAR(Dataset):
    def __init__(self,df_data,path_img,path_mask):
        self.mess = df_data.values
        self.img = path_img
        self.mask  =path_mask

    def __len__(self):
        return len(self.mess)


    def getinfo(self, idx):
        # 不知道这个有什么用
        if torch.is_tensor(idx):
            idx = idx.tolist()


        ID_image = self.mess[idx][0].split('+')[0]

        # 获取image
        img_path = os.path.join(self.img,'{}.jpg'.format(ID_image))
        # imgh',img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(640,480))

        # 获取mask

        mask_id = self.mess[idx][0].split('+')[-1]
        mask_path = os.path.join(self.mask,'mask_{}_{}.jpg'.format(ID_image,mask_id))
        # mask已经是 640 x 480 的了
        mask = cv2.imread(mask_path)



        # 获取bbox，RT
        # bbox: x y w h
        box, xyz, pyr = str2coords(self.mess[idx][1])
        box = np.array(box)
        xyz = np.array(xyz).T


        Rt = np.eye(4)

        pose = euler_to_Rot(pyr[1],pyr[0],pyr[2])
        Rt[:3, :3] = pose
        Rt[:3, 3] = xyz
        Rt = Rt[:3, :]

        # 获取中心坐标，尺度
        # 中心点为什么要反着计算？先这么算着吧，作者后面也是这么用的
        center =  np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        scale = 1.5
        s = max(box[3],box[2]) * scale
        s = min(s,max(480,640)) * 1.0

        rgb_crop = Crop_by_Pad(img,center,s,128,channel=3,interpolation=cv2.INTER_LINEAR)
        mask_crop = Crop_by_Pad(mask,center,s,128,channel=3,interpolation=cv2.INTER_LINEAR)

        return img_path,rgb_crop,mask_crop,box,center,scale,s,Rt

    def __getitem__(self, index):
        img_path, rgb_crop, mask, box, center, scale,size,Rt = self.getinfo(index)
        # 对输入进行处理，将3通道放到了第一维，并且变成0-1的范围
        input =  rgb_crop.transpose(2,0,1).astype(np.float32) / 255.



        return input, Rt, box, center,size, img_path,scale,mask