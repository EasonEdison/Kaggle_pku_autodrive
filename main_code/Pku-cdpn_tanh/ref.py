# encoding: utf-8
'''
@author: Zhigang Li
@license: (C) Copyright.
@contact: aaalizhigang@163.com
@software: Pose6D
@file: ref.py
@time: 18-10-24 下午9:00
@desc: 
'''
import paths
import numpy as np
import os

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
root_dir = paths.rootDir
data_cache_dir = os.path.join(root_dir, 'data')
exp_dir = os.path.join(root_dir, 'exp') 
data_dir = os.path.join(root_dir, 'dataset')
bbox_dir = os.path.join(root_dir, 'bbox_retinanet')
save_models_dir = os.path.join(root_dir, 'trained_models/{}/obj_{}.checkpoint')

lmo_camera_matrix = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])

# ---------------------------------------------------------------- #
# LINEMOD OCCLUSION DATASET
# ---------------------------------------------------------------- #
lmo_dir = os.path.join(data_dir, 'lmo_bop19')
lmo_test_dir = os.path.join(lmo_dir, 'test')
lmo_model_dir = os.path.join(lmo_dir, 'models_eval')
# object info
lmo_objects = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']
lmo_id2obj = {
             1: 'ape',
             5: 'can',
             6: 'cat',
             8: 'driller',
             9: 'duck',
             10: 'eggbox',
             11: 'glue',
             12: 'holepuncher',
             }
lmo_obj_num = len(lmo_id2obj)
def lmo_obj2id(obj_name):
    for k, v in lmo_id2obj.items():
        if v == obj_name:
            return k
# Camera info
lmo_width = 640
lmo_height = 480
lmo_center = (lmo_height / 2, lmo_width / 2)
