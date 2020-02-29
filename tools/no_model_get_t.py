import matplotlib.pyplot as plt
import numpy as np
from tools.visualize_all import *
import os
import pandas as pd
import copy


camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        print('2D:{} {}'.format(p_x,p_y))
        cv2.circle(image, (p_x, p_y), 20, (0, 255, 0), -1)
#         if p_x > image.shape[1] or p_y > image.shape[0]:
#             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image

def visualize(img, coords):
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31


    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        print('3D:{} {} {}'.format(x,y,z))
        img = draw_points(img, img_cor_points[-1:])

    return img

result_1 = pd.read_csv('D:\Dataset\\pku-autonomous-driving\\train.csv')


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
PATH = 'D:\\Dataset\\pku-autonomous-driving'
for i in range(len(result_1)):
    i = 0
    messes = []

    mess_1 = str2coords(result_1['PredictionString'][i])
    # for mes in mess:
    messes.append(mess_1)
    print('IMAGE: {}'.format(result_1['ImageId'][i]))
    img = os.path.join(PATH,'train_images','{}.jpg'.format(result_1['ImageId'][i]))
    images = cv2.imread(img)
    img_visual = visualize(images,mess_1)
    #imgjj = cv2.resize(img_visual,(640,480))
    #cv2.imshow('aaa',imgjj)
    plt.figure('ppp')
    plt.imshow(img_visual)
    plt.show()
    #cv2.waitKey()

