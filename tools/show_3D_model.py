import json
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

#
# model_root =  'D:\\jieya\\pku_test\\car_models_json'
# list_car = os.listdir(model_root)
# list_max = []
# for i in range(len(list_car)):
#     model_path = os.path.join(model_root,list_car[i])
#     with open(model_path) as json_file:
#         data = json.load(json_file)
#         # 明天算平均值
#         vertices = np.array(data['vertices'])
#         list_max.append(np.max(vertices,axis=0))
#         if i<4:
#             print('vert', np.max(vertices, axis=0))
#
#         # print('list max',np.mean(list_max,axis=1))
#
#
#
# print('list max mean', np.mean(list_max, axis=0))


with open('car_models_json/linken-SUV.json') as json_file:
    data = json.load(json_file)

    vertices = np.array(data['vertices'])
    print('vert', np.max(vertices, axis=0))

    #faces 和外观相关，三角形表示的车的外观？
    triangles = np.array(data['faces']) - 1
    plt.figure(figsize=(20, 10))
    ax = plt.axes(projection='3d')
    ax.set_title('car_type: ' + data['car_type'])
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 3])
    ax.plot_trisurf(vertices[:, 0], vertices[:, 2], triangles, -vertices[:, 1], shade=True, color='grey')

    plt.show()