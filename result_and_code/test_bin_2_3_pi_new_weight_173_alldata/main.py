import torch
#############################
BATCH_SIZE = 4
n_epochs = 10
IMG_WIDTH = 1024
#Train = False
Train = True
device = torch.device("cuda:0")
save_dir = 'get49'
test_epoch = 10
if_final = True
train_dev = 0.2
load_pretrain = False
#load_pretrain = True
load_epoch = 10
gamma_set = 0.1

save_epoch = 2
depth_loss_control = False
mask_loss_control = False

mask_weight = 0.1
reg_weight = 0.7
rot_weight = 0.3
#############################
num_class = 15



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from tqdm import tqdm#_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import os
from scipy.optimize import minimize
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from efficientnet_pytorch import EfficientNet
import csv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils

#PATH = '/home/ql-b423/CXH/Myproject/pku-autonomous-driving/'
PATH = '/home/ql-b423/Desktop/TXH/pku-autonomous-driving/'

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img


def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords


def rotate(x, angle):
    x = x + angle
    # 将x限制在0-2 pi的范围，超出就会减掉2 pi
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x

def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    # 这个应该就是讲的映射，将第三个维度变成1
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys

from math import sin, cos

# convert euler angle to rotation matrix
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

def sigmoid_(x):
    return 1.0 / (1.0 + np.exp(-x))


def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    #regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    regr_dict.pop('roll')
    return regr_dict




def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    # 狗日的，又转回来了，那我就直接不变
    # regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)


    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict



def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:, ::-1]
    return (img / 255).astype('float32')

def get_rot(regr):
    # bin 0 管 -pi ~ 0
    # bin 1 管 0 ~ pi
    #rot_dict = copy.deepcopy(regr)
    # rot_dict= rotate(regr, np.pi)
    if regr > 0:
        rot_bin = [0,1]
    else:
        rot_bin = [1,0]
    rot_res = [regr + 2.0/3.0 * np.pi, regr - 2.0/3.0 * np.pi]


    return rot_bin,rot_res


def get_mask_and_regr(img, labels, flip=False):
    # mask shape 40 128 是特征图的关键点的位置吧

    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 6], dtype='float32')
    rot_bin = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 2], dtype='long')
    rot_res = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 2], dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)

    # 将对应的x，y变成特征图维度下的数值

    for x, y, regr_dict in zip(xs, ys, coords):
        x, y = y, x
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4 / 3) / MODEL_SCALE
        y = np.round(y).astype('int')
        # 不是，这个xy本来就是中心点，只是判断中心点在不在合理范围内，在的话就设为1 ，其他的地方都是0

        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1

            rot_bin[x,y],rot_res[x,y] = get_rot(regr_dict['roll'])

            regr_dict = _regr_preprocess(regr_dict, flip)
            aaaa=sorted(regr_dict)
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]

    if flip:
        mask = np.array(mask[:, ::-1])
        regr = np.array(regr[:, ::-1])
        rot_bin = np.array(rot_bin[:, ::-1])
        rot_res = np.array(rot_res[:, ::-1])
    return mask, regr,rot_bin,rot_res

DISTANCE_THRESH_CLEAR = 2


def convert_3d_to_2d(x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy


def optimize_xy(r, c, x0, y0, z0, flipped=False):
    def distance_fn(xyz):
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx, z]])[0] - y) ** 2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / MODEL_SCALE
        return max(0.2, (x - r) ** 2 + (y - c) ** 2) + max(0.4, slope_err)

    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new


def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2) ** 2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]

def get_alpha(rot):
    idx = rot[1,:,:] > rot[5,:,:]
    a = rot[0,:,:]
    b = rot[1,:,:]
    c = rot[4,:,:]
    d = rot[5,:,:]
    # 下面这两个是算不同的bin的角度？
    # tan的定义域是 -pi/2 ~ pi/2
    # 正好左转pi/2，右转pi/2,就能覆盖2pi的范围
    # 合着就是，bin1 管 -pi ~ 0， bin2 管 0 ~ pi
    # test-
    al1 = np.arctan(rot[2,:,:] / rot[3,:,:])
    al2 = np.arctan(rot[6,:,:] / rot[7,:,:])
    alpha1 = np.arctan(rot[2,:,:] / rot[3,:,:]) - 2.0/3.0 * np.pi
    alpha2 = np.arctan(rot[6,:,:] / rot[7,:,:]) + np.pi * 2.0/3.0
    # 输出对应的bin
    return alpha1 * idx + alpha2 * (1 - idx)



def extract_coords(prediction, flipped=False):
    logits = prediction[0]
    regr_output = prediction[1:7]
    bin_output  = prediction[7:]
    roll = get_alpha(bin_output)[None,:,:]
    points = np.argwhere(logits > 0)
    #col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    col_names = ['pitch_cos', 'pitch_sin','x', 'y', 'yaw', 'z', 'roll']
    coords = []
    for r, c in points:
        # r c是行列信息
        # bbbb = regr_output[:, r, c]
        # bbbba = roll[:,r,c]
        # aaaa = np.concatenate((regr_output[:, r, c],roll[:,r,c]),axis=0)
        regr_dict = dict(zip(col_names, np.concatenate((regr_output[:, r, c],roll[:,r,c]),axis=0)))
        # 我是想在这里进行还原，不知可不可行
        coords.append(_regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = \
            optimize_xy(r, c,
                        coords[-1]['x'],
                        coords[-1]['y'],
                        coords[-1]['z'], flipped)
    coords = clear_duplicates(coords)
    return coords


def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)


def compute_bin_loss(output, target, mask):
    # 将mask广播，只算关键点位置的loss
    mask = mask[:,None,:].expand_as(output)
    output = output * mask.float()
    # 计算真实值和预测值的交叉熵，为什么要两个值，p 1-p 不就行了？
    return F.cross_entropy(output, target, reduction='mean')


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    batch_size = target_res.shape[0]
    # inde = output.contiguous().nonzero()
    # bb = output[inde[0][0],:,inde[0][2],inde[0][3]]
    output = output.contiguous().view(batch_size,8,-1)
    output_index = output.nonzero()


    target_bin = target_bin.reshape(batch_size, 2,-1)
    index_bin = target_bin.nonzero()
    aaa  =target_bin[index_bin[0][0],:,index_bin[0][2]]
    target_res = target_res.reshape(batch_size, 2,-1)
    index_res = target_res.nonzero()

    bbb  =target_res[index_res[0][0],:,index_res[0][2]]

    mask = mask.reshape(batch_size,-1)
    ########################test

    # a = torch.masked_select(target_res,iindex)
    # b = a.view(-1,2)
    ########################################

    # target_bin = target_bin.view(-1, 2)
    # target_res = target_res.view(-1, 2)


    loss_bin1 = compute_bin_loss(output[:, 0:2,:], target_bin[:, 0,:], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6,:], target_bin[:, 1,:], mask)

    loss_res = torch.zeros_like(loss_bin1)
    # made youwenti
    if target_bin[:, 0,:].nonzero().shape[0] > 0:
        # 这个index1就是非0元素的行了，也就是指向那个非0元素
        idx1 = target_bin[:, 0,:].nonzero()
        # 通过target将应属于bin1的预测挑选了出来
        for i in range(len(idx1)):
        # 将对应的真实值也挑选了出来
        # 这里应该就是常规的算正样本的loss
            loss_sin1 = compute_res_loss(output[idx1[i][0], 2,idx1[i][1]], torch.sin(target_res[idx1[i][0], 0,idx1[i][1]]))
            loss_cos1 = compute_res_loss(output[idx1[i][0], 3,idx1[i][1]], torch.cos(target_res[idx1[i][0], 0,idx1[i][1]]))
            loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1,:].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1,:].nonzero()
        for i in range(len(idx2)):
            loss_sin2 = compute_res_loss(output[idx2[i][0], 6,idx2[i][1]], torch.sin(target_res[idx2[i][0], 1,idx2[i][1]]))
            loss_cos2 = compute_res_loss(output[idx2[i][0], 7,idx2[i][1]], torch.cos(target_res[idx2[i][0], 1,idx2[i][1]]))
            loss_res += loss_sin2 + loss_cos2

        # valid_output2 = torch.index_select(output, 0, idx2.long())
        # valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        # loss_sin2 = compute_res_loss(
        #   valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        # loss_cos2 = compute_res_loss(
        #   valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        # loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


# def compute_rot_loss(output, target_bin, target_res, mask):
#     output = output.contiguous().view(-1, 8)
#     target_bin = target_bin.reshape(-1, 2)
#     target_res = target_res.reshape(-1, 2)
#     # target_bin = target_bin.view(-1, 2)
#     # target_res = target_res.view(-1, 2)
#     iddd = target_res.nonzero()
#     aaa = target_res[iddd[0],:]
#
#     mask = mask.view(-1, 1)
#
#     loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
#     loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
#
#     loss_res = torch.zeros_like(loss_bin1)
#     # made youwenti
#     if target_bin[:, 0].nonzero().shape[0] > 0:
#         # 这个index1就是非0元素的行了，也就是指向那个非0元素
#         idx1 = target_bin[:, 0].nonzero()[:, 0]
#         # 通过target将应属于bin1的预测挑选了出来
#         aaa = target_res[idx1,:]
#         valid_output1 = torch.index_select(output, 0, idx1.long())
#         # 将对应的真实值也挑选了出来
#         valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
#         # 这里应该就是常规的算正样本的loss
#         loss_sin1 = compute_res_loss(
#           valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
#         loss_cos1 = compute_res_loss(
#           valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
#         loss_res += loss_sin1 + loss_cos1
#     if target_bin[:, 1].nonzero().shape[0] > 0:
#         idx2 = target_bin[:, 1].nonzero()[:, 0]
#         valid_output2 = torch.index_select(output, 0, idx2.long())
#         valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
#         loss_sin2 = compute_res_loss(
#           valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
#         loss_cos2 = compute_res_loss(
#           valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
#         loss_res += loss_sin2 + loss_cos2
#     return loss_bin1 + loss_bin2 + loss_res

############  训练区

def criterion(prediction, mask, regr, rot_bin, rot_res,size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    # mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()

    # Regression L1 loss
    pred_regr = prediction[:, 1:7]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    # rot bin loss 第七个开始才是bin
    rot_bin_loss = compute_rot_loss(prediction[:,7:],rot_bin,rot_res,mask)
    # Sum
    loss = mask_weight * mask_loss + reg_weight * regr_loss + rot_weight * rot_bin_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss

def train_model(epoch, history=None):
    model.train()

    for batch_idx, (img_batch, mask_batch, regr_batch,rot_bin_batch,rot_res_batch) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        rot_bin_batch = rot_bin_batch.to(device)
        rot_res_batch = rot_res_batch.to(device)

        optimizer.zero_grad()
        output = model(img_batch)

        loss = criterion(output, mask_batch, regr_batch, rot_bin_batch, rot_res_batch)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()

        loss.backward()

        optimizer.step()
        exp_lr_scheduler.step()

    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data))
    return loss.item()


def evaluate_model(epoch, history=None):
    model.eval()
    loss = 0

    with torch.no_grad():
        for img_batch, mask_batch, regr_batch,rot_bin_batch,rot_res_batch  in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)
            rot_bin_batch = rot_bin_batch.to(device)
            rot_res_batch = rot_res_batch.to(device)

            output = model(img_batch)

            loss += criterion(output, mask_batch, regr_batch, rot_bin_batch, rot_res_batch).data

    loss /= len(dev_loader.dataset)

    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()

    print('Dev loss: {:.4f}'.format(loss))
    return loss.cpu().numpy()

###############   数据集区
class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 得到名字和训练label数据
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(10) == 1

        # Read image
        img0 = imread(img_name, True)
        img = preprocess_image(img0, flip=flip)
        # 三个通道的维度变成了第一维
        img = np.rollaxis(img, 2, 0)

        # Get mask and regression maps
        mask, regr,rot_bin,rot_res = get_mask_and_regr(img0, labels, flip=flip)
        # 128 40 6 -> 6 40 128
        regr = np.rollaxis(regr, 2, 0)
        # 128 40 2 -> 2 40 128
        rot_bin = np.rollaxis(rot_bin, 2, 0)
        rot_res = np.rollaxis(rot_res, 2, 0)
        # reg pitch_cos pitch_sin x y yaw z
        return [img, mask, regr, rot_bin, rot_res]


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh


class MyUNet(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')

        self.conv0 = double_conv(5, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)

        self.mp = nn.MaxPool2d(2)

        self.up1 = up(1282 + 1024, 512)
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        feats = self.base_model.extract_features(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
        feats = torch.cat([bg, feats, bg], 3)

        # Add positional info
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        feats = torch.cat([feats, mesh2], 1)

        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x

############### main区

if Train:
    train_control = True
    test_control = False
else:
    train_control = False
    test_control = True

if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8

train = pd.read_csv(os.path.join(PATH , 'train.csv'))
# train = pd.read_csv(os.path.join(PATH , 'augu_last.csv'))
test = pd.read_csv(os.path.join(PATH , 'sample_submission.csv'))

# From camera.zip
IMG_SHAPE = (2710,3384,3)

points_df = pd.DataFrame()
for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
    arr = []
    for ps in train['PredictionString']:
        coords = str2coords(ps)
        # 为什么要加在一块
        # 这里是排在后面
        arr += [c[col] for c in coords]
    points_df[col] = arr
zy_slope = LinearRegression()
X = points_df[['z']]
y = points_df['y']
zy_slope.fit(X, y)

# Will use this model later
xzy_slope = LinearRegression()
X = points_df[['x', 'z']]
y = points_df['y']
xzy_slope.fit(X, y)

train_images_dir = PATH + 'train_images/{}.jpg'
test_images_dir = PATH + 'test_images/{}.jpg'

df_train, df_dev = train_test_split(train, test_size=train_dev, random_state=42)
#df_train = train
df_test = test

# Create dataset objects
train_dataset = CarDataset(df_train, train_images_dir, training=True)
dev_dataset = CarDataset(df_dev, train_images_dir, training=False)
test_dataset = CarDataset(df_test, test_images_dir, training=False)


# Create data generators - they will produce batches
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


# Gets the GPU if there is one, otherwise the cpu


model = MyUNet(num_class).to(device)
if load_pretrain:
    model.load_state_dict(torch.load('{}/{}_{}.pth'.format(save_dir,save_dir,load_epoch)))

optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=gamma_set)


import gc

if train_control:
    history = pd.DataFrame()
    headers = ['Train','Dev']
    rows = []
    for epoch in range(n_epochs):
        epoch = epoch +1
        if load_pretrain:
            epoch = epoch + load_epoch
        gc.collect()
        train_loss = train_model(epoch, history)
        dev_loss = evaluate_model(epoch, history)
        rows.append((train_loss,dev_loss))

        if epoch == n_epochs:
            torch.save(model.state_dict(), '{}/{}_final.pth'.format(save_dir,save_dir))
            history.to_csv('{}/history_final.csv'.format(save_dir))
            with open('{}/{}_final.csv'.format(save_dir,save_dir),'w',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)  # 写入标签
                writer.writerows(rows)  # 写入样本数据
            break
        if epoch % save_epoch ==0:
            torch.save(model.state_dict(), '{}/{}_{}.pth'.format(save_dir,save_dir,epoch))
            history.to_csv('{}/history_{}.csv'.format(save_dir,epoch))
            with open('{}/{}_{}e.csv'.format(save_dir,save_dir,epoch),'w',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)  # 写入标签
                writer.writerows(rows)  # 写入样本数据



if test_control:
    predictions = []
    if if_final:
        model.load_state_dict(torch.load('{}/{}_final.pth'.format(save_dir,save_dir)))
    else:
        model.load_state_dict(torch.load('{}/{}_{}.pth'.format(save_dir,save_dir,test_epoch)))

    model.eval()
    for img, _, _,_,_ in tqdm(test_loader):
        with torch.no_grad():
            output = model(img.to(device))
        output = output.data.cpu().numpy()
        for out in output:
            coords = extract_coords(out)
            s = coords2str(coords)
            predictions.append(s)

    test = pd.read_csv(PATH + 'sample_submission.csv')
    test['PredictionString'] = predictions
    if if_final:
        test.to_csv('{}/pred_{}_final.csv'.format(save_dir,save_dir,test_epoch), index=False)
    else:
        test.to_csv('{}/pred_{}_{}e.csv'.format(save_dir,save_dir,test_epoch), index=False)
