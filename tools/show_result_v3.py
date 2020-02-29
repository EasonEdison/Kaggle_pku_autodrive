import matplotlib.pyplot as plt
import numpy as np
from tools.visualize_all import *
import os
import pandas as pd
import copy
###v2:增加了数据显示
###v3:改成了对比模式

# 改这两个路径就可以了
PATH = 'D:\\Dataset\\pku-autonomous-driving'
#path_result = 'D:\jieya\CenterNet_IamROOKIE\predictions046.csv'
path_result_1 = 'D:\jieya\CenterNet_IamROOKIE\prediction\\1.21\\new_patch.csv'
path_result_2 = 'D:\jieya\CenterNet_IamROOKIE\prediction\\1.21\\change_result.csv'
# # 不看数据的时候设置False
show_data = True
scale = 2
fig_size =  15
name_result_1 = path_result_1.split('\\')[-1]
name_result_2 = path_result_2.split('\\')[-1]


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


def visualize_all(img, messes,name_result_1,name_result_2,camera_matrix, coor_3d_box,line_show_2D=False,line_show_3D=False,point_show = False):
    point_map = np.zeros((len(coor_3d_box),2))
    #point_map = np.zeros((3,2))

    img1 = img.copy()
    img2 = copy.deepcopy(img)
    for index,mess in enumerate(messes):
        if index==0:
            img_ = img1
        else:
            img_ = img2
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
                    cv2.circle(img_, (point_map[i][0].astype(int), point_map[i][1].astype(int)), 12, (0, 255, 0), -1)

            x_max, y_max = np.max(cornet_get, axis=0).astype(int)
            x_min, y_min = np.min(cornet_get, axis=0).astype(int)
            for i in range(0,8):
                cornet_get[i] = (cornet_get[i][0].astype(int),cornet_get[i][1].astype(int))
            if line_show_3D:
                #print('cornet_get',cornet_get)
                for i in range(0,8):
                    if (i+1)%2 != 0:
                        cv2.line(img_,cornet_get[i],cornet_get[i+1],(0,255,0), 2)
                    if i <4:
                        cv2.line(img_,cornet_get[i],cornet_get[i+4],(0,255,0), 2)
                if show_data:
                    txt1 = 'x:{}'.format(float('%.2f'% point['x']))
                    txt2 = 'y:{}'.format(float('%.2f'% point['y']))
                    txt3 = 'z:{}'.format(float('%.2f'% point['z']))
                    txt4 = 'p:{}'.format(float('%.2f'% point['pitch']))
                    txt5 = 'y:{}'.format(float('%.2f'% point['yaw']))
                    txt6 = 'r:{}'.format(float('%.2f'% point['roll']))
                    img_ = cv2.putText(img_,txt6,(((x_min+x_max)/2).astype(int),y_min-50),cv2.FONT_HERSHEY_COMPLEX,scale,(0,255,255),4)
                    img_ = cv2.putText(img_,txt5,(((x_min+x_max)/2).astype(int),y_min-100),cv2.FONT_HERSHEY_COMPLEX,scale,(0,255,255),4)
                    img_ = cv2.putText(img_,txt4,(((x_min+x_max)/2).astype(int),y_min-150),cv2.FONT_HERSHEY_COMPLEX,scale,(0,255,255),4)
                    img_ = cv2.putText(img_,txt3,(((x_min+x_max)/2).astype(int),y_min-200),cv2.FONT_HERSHEY_COMPLEX,scale,(0,0,255),4)
                    img_ = cv2.putText(img_,txt2,(((x_min+x_max)/2).astype(int),y_min-250),cv2.FONT_HERSHEY_COMPLEX,scale,(0,0,255),4)
                    img_ = cv2.putText(img_,txt1,(((x_min+x_max)/2).astype(int),y_min-300),cv2.FONT_HERSHEY_COMPLEX,scale,(0,0,255),4)
                cv2.line(img_, cornet_get[0], cornet_get[2], (0, 255, 0), 2)
                cv2.line(img_, cornet_get[1], cornet_get[3], (0, 255, 0), 2)
                cv2.line(img_, cornet_get[4], cornet_get[6], (0, 255, 0), 2)
                cv2.line(img_, cornet_get[5], cornet_get[7], (0, 255, 0), 2)

                    #cv2.imwrite('D:\\jieya\\Pku-cdpn\\network\\aa{}.jpg'.format(random.randint(1,1000)),img)

    if line_show_2D:
        #print('cornet_get',cornet_get)

        cv2.line(img, (x_max, y_max), (x_max, y_min), (255,0,0), 2)
        cv2.line(img, (x_max, y_min), (x_min, y_min), (255,0,0), 2)
        cv2.line(img, (x_min, y_min), (x_min, y_max), (255,0,0), 2)
        cv2.line(img, (x_min, y_max), (x_max, y_max), (255,0,0), 2)
        img = cv2.resize(img,(640,480))
        cv2.imshow('2D box',img)
        cv2.waitKey(1000)

    # img1 = cv2.resize(img1, (1280, 960))
    # img2 = cv2.resize(img2, (1280, 960))

    img1 =np.array(img1[:, :, ::-1])
    img2 =np.array(img2[:, :, ::-1])
    fig,axes = plt.subplots(1,2,figsize=(fig_size,fig_size))

    axes[0].imshow(img1)
    axes[0].set_title('{}'.format(name_result_1))


    axes[1].imshow(img2)
    axes[1].set_title('{}'.format(name_result_2))

    plt.show()
    # cv2.imshow('3D box',img)
    #
    # cv2.waitKey()
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


camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)
# pitch, yaw, roll, x, y, z and confidence
# 这里是官方名字给错了，先按着错的来
def str2coords(s, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z','conf']):
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



result_1 = pd.read_csv(path_result_1)
result_2 = pd.read_csv(path_result_2)

for i in range(len(result_1)):

    messes = []
    try:
        mess_1 = str2coords(result_1['PredictionString'][i])
        mess_2 = str2coords(result_2['PredictionString'][i])
        # for mes in mess:
        messes.append(mess_1)
        messes.append(mess_2)
        print('IMAGE: {}'.format(result_1['ImageId'][i]))
        img = os.path.join(PATH,'test_images','{}.jpg'.format(result_1['ImageId'][i]))
        images = cv2.imread(img)
        visualize_all(images,messes,name_result_1,name_result_2,camera_matrix,eight_point,line_show_3D=True)
    except:
        continue