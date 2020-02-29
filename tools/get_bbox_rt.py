import os
import cv2
import numpy as np
from math import sin, cos
import pandas as pd

train = pd.read_csv('train.csv')

coor = np.array([0.9,0.8,2.31])


eight_point = np.zeros((8,3))
for i in range(0,8):
    coor[2] = -coor[2]
    if (i) % 4 ==0:
        coor[0] = -coor[0]
    if (i) %2 == 0:
        coor[1] = -coor[1]
    #print('eight_point',np.shape(eight_point))

    eight_point[i,0] = coor[0]
    eight_point[i,1] = coor[1]
    eight_point[i,2] = coor[2]

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

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

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

def get_bbox(img_name,image_mess,coor_3d_box, line_show=False,path=None):
    point_map = np.zeros((len(coor_3d_box), 2))
    # img = img.copy()
    # for point in Rt:
    #     # Get values
    #     x, y, z = point['x'], point['y'], point['z']
    #     yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
    #     # Math
    #     Rt = np.eye(4)
    #     t = np.array([x, y, z])
    #     Rt[:3, 3] = t
    #     Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
    #     Rt = Rt[:3, :]
    bbox_all = []
    rt_all = []
    for point in image_mess:
        cornet_get = []
        for i in range(len(coor_3d_box)):

            # Get values
            x, y, z = point['x'], point['y'], point['z']
            yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
            # Math
            Rt = np.eye(4)
            t = np.array([x, y, z])
            Rt[:3, 3] = t
            Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
            Rt = Rt[:3, :]

            P = np.array([coor_3d_box[i][0], coor_3d_box[i][1], coor_3d_box[i][2], 1])

            img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
            img_cor_points = img_cor_points.T

            img_cor_points[0] /= img_cor_points[2]
            img_cor_points[1] /= img_cor_points[2]
            img_cor_points = img_cor_points.astype(int)
            point_map[i][0] = img_cor_points[0]
            point_map[i][1] = img_cor_points[1]
            # 应该已经是int了啊
            cornet_get.append(point_map[i][:2])
            # print('point',point_map[i][:2])


        #print('corn', cornet_get)
        x_max, y_max = np.max(cornet_get, axis=0)
        x_min, y_min = np.min(cornet_get, axis=0)

        y_max = (y_max * 480 / 2710).astype(int)
        y_min = (y_min * 480 / 2710).astype(int)
        x_min = (x_min * 640 / 3384).astype(int)
        x_max = (x_max * 640 / 3384).astype(int)

        # 这里忘了取范围了
        bbox = [x_min,y_min,x_max,y_max]
        bbox[0] = np.clip(bbox[0],0,640)
        bbox[1] = np.clip(bbox[1],0,480)
        bbox[2] = np.clip(bbox[2],0,640)
        bbox[3] = np.clip(bbox[3],0,480)
        bbox_all.append(bbox)
        rt = [point['x'], point['y'], point['z'],point['pitch'], point['yaw'], point['roll']]
        rt_all.append(rt)

        if line_show:
            img = cv2.imread(path)
            img = cv2.resize(img,(640,480))
            img = cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(255,0,0),1)
            cv2.imshow('aaa',img)
            cv2.waitKey()
            # cv2.line(img, (x_max, y_max), (x_max, y_min), (255,0,0), 1)
            # cv2.line(img, (x_max, y_min), (x_min, y_min), (255,0,0), 1)
            # cv2.line(img, (x_min, y_min), (x_min, y_max), (255,0,0), 1)
            # cv2.line(img, (x_min, y_max), (x_max, y_max), (255,0,0), 1)
    # np.save('D:\\jieya\\pku_test\\train_bbox\\bbox_{}'.format(img_name),bbox_all)
    # np.save('D:\\jieya\\pku_test\\train_rt\\rt_{}'.format(img_name),rt_all)



def imread(path, fast_mode=False):
    cv2.destroyAllWindows()
    img = cv2.imread(path)

    img = cv2.resize(img,(640,480))


    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

index = 1260

for i in range(len(train['ImageId'])):


    str_mess = str2coords(train['PredictionString'][i])
    root = 'D:\\jieya\\pku_test\\train_images'
    path = os.path.join(root,'{}.jpg'.format(train['ImageId'][i]))
    # 想要show，就在后面加True，加地址参数
    print('name',path)
    get_bbox(train['ImageId'][i],str_mess,eight_point,line_show=True,path=path)
    # cv2.imshow('a',img)
    # cv2.waitKey(111111)
    print('{} / {}'.format(i,len(train['ImageId'])))