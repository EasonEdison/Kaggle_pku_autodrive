import cv2
import numpy as np
from math import cos,sin

coor = np.array([0.9, 0.8, 2.31])

eight_point = np.zeros((8, 3))
for i in range(0, 8):
    coor[2] = -coor[2]
    if (i) % 4 == 0:
        coor[0] = -coor[0]
    if (i) % 2 == 0:
        coor[1] = -coor[1]
    # print('eight_point',np.shape(eight_point))

    eight_point[i, 0] = coor[0]
    eight_point[i, 1] = coor[1]
    eight_point[i, 2] = coor[2]

import random
def visualize_all(img, mess,camera_matrix, coor_3d_box,line_show_2D=False,line_show_3D=False,point_show = False):
    point_map = np.zeros((len(coor_3d_box),2))
    #point_map = np.zeros((3,2))

    img = img.copy()
    for point in mess:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]


        cornet_get = []
        for i in range(len(coor_3d_box)):
            # Get values
            # x, y, z = point['x'], point['y'], point['z']
            # yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
            # Math
            # t = np.array([x, y, z])

            P = np.array([coor_3d_box[i][0],coor_3d_box[i][1],coor_3d_box[i][2],1])

            img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
            img_cor_points = img_cor_points.T

            img_cor_points[0] /= img_cor_points[2]
            img_cor_points[1] /= img_cor_points[2]
            img_cor_points = img_cor_points.astype(int)
            point_map[i][0] = img_cor_points[0]
            point_map[i][1] = img_cor_points[1]


            cornet_get.append(point_map[i][:2])
            #print('point',point_map[i][:2])

            if point_show:
                cv2.circle(img, (point_map[i][0].astype(int), point_map[i][1].astype(int)), 12, (0, 255, 0), -1)

    x_max, y_max = np.max(cornet_get, axis=0).astype(int)
    x_min, y_min = np.min(cornet_get, axis=0).astype(int)

    if line_show_2D:
        #print('cornet_get',cornet_get)

        cv2.line(img, (x_max, y_max), (x_max, y_min), (255,0,0), 2)
        cv2.line(img, (x_max, y_min), (x_min, y_min), (255,0,0), 2)
        cv2.line(img, (x_min, y_min), (x_min, y_max), (255,0,0), 2)
        cv2.line(img, (x_min, y_max), (x_max, y_max), (255,0,0), 2)
        img = cv2.resize(img,(640,480))
        cv2.imshow('2D box',img)
        cv2.waitKey(1000)

    for i in range(0,8):
        cornet_get[i] = (cornet_get[i][0].astype(int),cornet_get[i][1].astype(int))
    if line_show_3D:
        print('cornet_get',cornet_get)
        for i in range(0,8):
            if (i+1)%2 != 0:
                cv2.line(img,cornet_get[i],cornet_get[i+1],(0,255,0), 2)
            if i <4:
                cv2.line(img,cornet_get[i],cornet_get[i+4],(0,255,0), 2)
        cv2.line(img, cornet_get[0], cornet_get[2], (0, 255, 0), 2)
        cv2.line(img, cornet_get[1], cornet_get[3], (0, 255, 0), 2)
        cv2.line(img, cornet_get[4], cornet_get[6], (0, 255, 0), 2)
        cv2.line(img, cornet_get[5], cornet_get[7], (0, 255, 0), 2)
        img = cv2.resize(img,(1280,960))
        cv2.imwrite('D:\\jieya\\Pku-cdpn\\network\\aa{}.jpg'.format(random.randint(1,1000)),img)
        cv2.imshow('3D box',img)

        cv2.waitKey(10000)
    return img

def imread(path, fast_mode=False):
    cv2.destroyAllWindows()
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

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
