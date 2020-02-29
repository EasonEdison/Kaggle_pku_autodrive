import torch.utils.data as data
import numpy as np
import ref
import cv2
import os, sys
import pickle
import json
from utils import *
from torch.utils.data import Dataset, DataLoader

# 我需要：根目录，就没了？

class CAR(Dataset):
    def __init__(self):
        # 预测的bbox和mask目录
        self.predict_bm_dir =
        # 真实的bbox和rt目录
        self.gt_brt_dir     =
