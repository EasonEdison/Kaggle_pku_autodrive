import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

# a label and all meta information
Label = namedtuple('Label', [

    'name'        , # The name of a car type
    'id'          , # id for specific car type
    'category'    , # The name of the car category, 'SUV', 'Sedan' etc
    'categoryId'  , # The ID of car category. Used to create ground truth images
                    # on category level.
    ])



# A list of all labels

models = [
    #     name          id   is_valid  category  categoryId
    Label(             'baojun-310-2017',          0,       '2x',          0),
    Label(                'biaozhi-3008',          1,       '2x',          0),
    Label(          'biaozhi-liangxiang',          2,       '2x',          0),
    Label(           'bieke-yinglang-XT',          3,       '2x',          0),
    Label(                'biyadi-2x-F0',          4,       '2x',          0),
    Label(               'changanbenben',          5,       '2x',          0),
    Label(                'dongfeng-DS5',          6,       '2x',          0),
    Label(                     'feiyate',          7,       '2x',          0),
    Label(         'fengtian-liangxiang',          8,       '2x',          0),
    Label(                'fengtian-MPV',          9,       '2x',          0),
    Label(           'jilixiongmao-2015',         10,       '2x',          0),
    Label(           'lingmu-aotuo-2009',         11,       '2x',          0),
    Label(                'lingmu-swift',         12,       '2x',          0),
    Label(             'lingmu-SX4-2012',         13,       '2x',          0),
    Label(              'sikeda-jingrui',         14,       '2x',          0),
    Label(        'fengtian-weichi-2006',         15,       '3x',          1),
    Label(                   '037-CAR02',         16,       '3x',          1),
    Label(                     'aodi-a6',         17,       '3x',          1),
    Label(                   'baoma-330',         18,       '3x',          1),
    Label(                   'baoma-530',         19,       '3x',          1),
    Label(            'baoshijie-paoche',         20,       '3x',          1),
    Label(             'bentian-fengfan',         21,       '3x',          1),
    Label(                 'biaozhi-408',         22,       '3x',          1),
    Label(                 'biaozhi-508',         23,       '3x',          1),
    Label(                'bieke-kaiyue',         24,       '3x',          1),
    Label(                        'fute',         25,       '3x',          1),
    Label(                     'haima-3',         26,       '3x',          1),
    Label(               'kaidilake-CTS',         27,       '3x',          1),
    Label(                   'leikesasi',         28,       '3x',          1),
    Label(               'mazida-6-2015',         29,       '3x',          1),
    Label(                  'MG-GT-2015',         30,       '3x',          1),
    Label(                       'oubao',         31,       '3x',          1),
    Label(                        'qiya',         32,       '3x',          1),
    Label(                 'rongwei-750',         33,       '3x',          1),
    Label(                  'supai-2016',         34,       '3x',          1),
    Label(             'xiandai-suonata',         35,       '3x',          1),
    Label(            'yiqi-benteng-b50',         36,       '3x',          1),
    Label(                       'bieke',         37,       '3x',          1),
    Label(                   'biyadi-F3',         38,       '3x',          1),
    Label(                  'biyadi-qin',         39,       '3x',          1),
    Label(                     'dazhong',         40,       '3x',          1),
    Label(              'dazhongmaiteng',         41,       '3x',          1),
    Label(                    'dihao-EV',         42,       '3x',          1),
    Label(      'dongfeng-xuetielong-C6',         43,       '3x',          1),
    Label(     'dongnan-V3-lingyue-2011',         44,       '3x',          1),
    Label(    'dongfeng-yulong-naruijie',         45,      'SUV',          2),
    Label(                     '019-SUV',         46,      'SUV',          2),
    Label(                   '036-CAR01',         47,      'SUV',          2),
    Label(                 'aodi-Q7-SUV',         48,      'SUV',          2),
    Label(                  'baojun-510',         49,      'SUV',          2),
    Label(                    'baoma-X5',         50,      'SUV',          2),
    Label(             'baoshijie-kayan',         51,      'SUV',          2),
    Label(             'beiqi-huansu-H3',         52,      'SUV',          2),
    Label(              'benchi-GLK-300',         53,      'SUV',          2),
    Label(                'benchi-ML500',         54,      'SUV',          2),
    Label(         'fengtian-puladuo-06',         55,      'SUV',          2),
    Label(            'fengtian-SUV-gai',         56,      'SUV',          2),
    Label(    'guangqi-chuanqi-GS4-2015',         57,      'SUV',          2),
    Label(        'jianghuai-ruifeng-S3',         58,      'SUV',          2),
    Label(                  'jili-boyue',         59,      'SUV',          2),
    Label(                      'jipu-3',         60,      'SUV',          2),
    Label(                  'linken-SUV',         61,      'SUV',          2),
    Label(                   'lufeng-X8',         62,      'SUV',          2),
    Label(                 'qirui-ruihu',         63,      'SUV',          2),
    Label(                 'rongwei-RX5',         64,      'SUV',          2),
    Label(             'sanling-oulande',         65,      'SUV',          2),
    Label(                  'sikeda-SUV',         66,      'SUV',          2),
    Label(            'Skoda_Fabia-2011',         67,      'SUV',          2),
    Label(            'xiandai-i25-2016',         68,      'SUV',          2),
    Label(            'yingfeinidi-qx80',         69,      'SUV',          2),
    Label(             'yingfeinidi-SUV',         70,      'SUV',          2),
    Label(                  'benchi-SUR',         71,      'SUV',          2),
    Label(                 'biyadi-tang',         72,      'SUV',          2),
    Label(           'changan-CS35-2012',         73,      'SUV',          2),
    Label(                 'changan-cs5',         74,      'SUV',          2),
    Label(          'changcheng-H6-2016',         75,      'SUV',          2),
    Label(                 'dazhong-SUV',         76,      'SUV',          2),
    Label(     'dongfeng-fengguang-S560',         77,      'SUV',          2),
    Label(       'dongfeng-fengxing-SX6',         78,      'SUV',          2)

]

# name to label object
car_name2id = {label.name: label for label in models}
car_id2name = {label.id: label for label in models}




train = pd.read_csv('D:\Dataset\\pku-autonomous-driving\\train.csv')


# From camera.zip
camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

# train.head()

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




def imread(path, fast_mode=False):
    cv2.destroyAllWindows()
    img = cv2.imread(path)
    # print('shape',np.shape(img))
    img = cv2.resize(img,(640,480))
    # print('shape',np.shape(img))

    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img


index = 1260



# 3D Visualization
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
color = (255, 0, 0)

def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), 8, (0, 255, 0), -1)
#         if p_x > image.shape[1] or p_y > image.shape[0]:
#             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image





def visualize(img, Rt,coor_3d,line=True,point_show=True):
    # You will also need functions from the previous cells
    # x_l = 1.02
    # y_l = 0.80
    # z_l = 2.31

    x_l,y_l,z_l  = coor_3d

    img = img.copy()
    print('RT',Rt)
    for point in Rt:
        # Get values
        print('point',point)
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
        #print('?', img_cor_points[:, 0])
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        if line:
            img = draw_line(img, img_cor_points)
        if point_show:
            img = draw_points(img, img_cor_points[-1:])

    return img




def get_3D_box(model_path,all=False):
    with open(model_path) as json_file:
        data = json.load(json_file)
        if all :
            coor = data['vertices']
        else:
            coor = np.max(data['vertices'], axis=0)


        return coor




def show_3D_model(path):
    with open(path) as json_file:
        data = json.load(json_file)
        vertices = np.array(data['vertices'])
        triangles = np.array(data['faces']) - 1
        plt.figure(figsize=(20,10))
        ax = plt.axes(projection='3d')
        ax.set_title('car_type: '+data['car_type'])
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([0, 3])
        ax.plot_trisurf(vertices[:,0], vertices[:,2], triangles, -vertices[:,1], shade=True, color='grey')

    plt.show()


fig, axes = plt.subplots(1, 2, figsize=(20,20))
# img = imread('car.jpg')
#axes[0].imshow(img)


car_model_dir = 'car_models_json'
img_dir = 'D:\\jieya\\pku_test\\train_images'

index =8

str_mess = str2coords(train['PredictionString'][index])
id_img = train['ImageId'][index]

id = str_mess[0]['id']
car_model_name = car_id2name[id].name
# car_model_name = 'aodi-Q7-SUV'
# print('car_model_name',car_model_name)
name = os.path.join(car_model_dir,car_model_name)
img_name = os.path.join(img_dir,'{}.jpg'.format(id_img))
# print('imgname',img_name)
img = imread(img_name)


#x= np.max(str_mess[:]['x'])

# print('x')

point_cloud = get_3D_box(name + '.json',all=True)

def visualize_all(img, image_mess, coor_3d_box,line_show=False,point_show = False):
    point_map = np.zeros((len(coor_3d_box),2))
    #point_map = np.zeros((3,2))

    img = img.copy()
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

            P = np.array([coor_3d_box[i][0],coor_3d_box[i][1],coor_3d_box[i][2],1])

            img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
            img_cor_points = img_cor_points.T

            img_cor_points[0] /= img_cor_points[2]
            img_cor_points[1] /= img_cor_points[2]
            img_cor_points = img_cor_points.astype(int)
            point_map[i][0] = img_cor_points[0]
            point_map[i][1] = img_cor_points[1]
            # 应该已经是int了啊
            cornet_get.append(point_map[i][:2])
            print('point',point_map[i][:2])
            if point_show:
                cv2.circle(img, (point_map[i][0].astype(int), point_map[i][1].astype(int)), 12, (0, 255, 0), -1)

        # print('corn',cornet_get)
        x_max, y_max = np.max(cornet_get, axis=0)
        x_min, y_min = np.min(cornet_get, axis=0)

        y_max = (y_max * 480 / 2710).astype(int)
        y_min = (y_min * 480 / 2710).astype(int)
        x_min = (x_min * 640 / 3384).astype(int)
        x_max = (x_max * 640 / 3384).astype(int)


        if line_show:
            cv2.line(img, (x_max, y_max), (x_max, y_min), color, 1)
            cv2.line(img, (x_max, y_min), (x_min, y_min), color, 1)
            cv2.line(img, (x_min, y_min), (x_min, y_max), color, 1)
            cv2.line(img, (x_min, y_max), (x_max, y_max), color, 1)

    return img




with open('car_models_json/linken-SUV.json') as json_file:
    data = json.load(json_file)

    vertices = np.array(data['vertices'])
    # print('data',data['vertices'])
    coor = np.max(data['vertices'], axis=0)
    coor = np.array([0.9,0.8,2.31])
    # print('coor',coor)

    # print('coor',np.shape(coor))


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


# print('e',eight_point)
# print('str mess',str_mess)
img = visualize_all(img,str_mess,eight_point,line_show=True,point_show=False)


axes[0].imshow(img)
plt.show()
