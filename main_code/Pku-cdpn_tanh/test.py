import torch
import os
import sys
import cv2
import time
import ref
import csv 
import numpy as np
from progress.bar import Bar
from math import sin, cos
import matplotlib.pyplot as plt
# models_info 是 一些size，坐标，直径信息

def visualize_all(img, R,t,camera_matrix, coor_3d_box,line_show_2D=False,line_show_3D=False,point_show = False):
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


    cornet_get = []
    for i in range(len(coor_3d_box)):

        # Get values
        # x, y, z = point['x'], point['y'], point['z']
        # yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        # t = np.array([x, y, z])
        Rt[:3, 3] = t.T
        Rt[:3, :3] = R
        # Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]

        P = np.array([coor_3d_box[i][0],coor_3d_box[i][1],coor_3d_box[i][2],1])

        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T

        img_cor_points[ 0] /= img_cor_points[2]
        img_cor_points[1] /= img_cor_points[2]
        img_cor_points = img_cor_points.astype(int)
        point_map[i][0] = img_cor_points[0]
        point_map[i][1] = img_cor_points[1]
        # 应该已经是int了啊

        cornet_get.append(point_map[i][:2])
        #print('point',point_map[i][:2])

        if point_show:
            cv2.circle(img, (point_map[i][0].astype(int), point_map[i][1].astype(int)), 12, (0, 255, 0), -1)

    print('corn',cornet_get)
    x_max, y_max = np.max(cornet_get, axis=0).astype(int)
    x_min, y_min = np.min(cornet_get, axis=0).astype(int)

    if line_show_2D:
        cv2.line(img, (x_max, y_max), (x_max, y_min), (255,0,0), 2)
        cv2.line(img, (x_max, y_min), (x_min, y_min), (255,0,0), 2)
        cv2.line(img, (x_min, y_min), (x_min, y_max), (255,0,0), 2)
        cv2.line(img, (x_min, y_max), (x_max, y_max), (255,0,0), 2)

    for i in range(0,8):

        cornet_get[i] = (cornet_get[i][0].astype(int),cornet_get[i][1].astype(int))
    print('cornet_get[i]',cornet_get[0])
    if line_show_3D:

        for i in range(0,8):
            if (i+1)%2 != 0:

                cv2.line(img,cornet_get[i],cornet_get[i+1],(0,0,255), 2)
            if i <4:

                cv2.line(img,cornet_get[i],cornet_get[i+4],(255,0,0), 2)

        cv2.line(img, cornet_get[0], cornet_get[2], (0, 255, 0), 2)
        cv2.line(img, cornet_get[1], cornet_get[3], (0, 255, 0), 2)
        cv2.line(img, cornet_get[4], cornet_get[6], (0, 255, 0), 2)
        cv2.line(img, cornet_get[5], cornet_get[7], (0, 255, 0), 2)

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


def test(cfg, dataLoader, model, models_info=None, models_vtx=None):

    model.eval()
    use_mask = True
    if cfg.pytorch.exp_mode == 'val':
        from eval import Evaluation
        Eval = Evaluation(cfg.pytorch, models_info, models_vtx)     
    elif cfg.pytorch.exp_mode == 'test':
        csv_file = open(cfg.pytorch.save_csv_path, 'w')
        fieldnames = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        rst_collect = []

    preds = {}
    nIters = len(dataLoader)
    bar = Bar('{}_{}'.format(cfg.pytorch.dataset, cfg.pytorch.object), max=nIters)
    wall_time = 0
    # bbox是2D框，一开始全是猫的2D框
    # center是2D框的中心
    # size是h or w中最大的一个乘1.5
    for i, (input, pose, bbox, center, size, clsIdx, imgPath, scene_id, image_id, score,scale) in enumerate(dataLoader):
        # 这个input就是crop出来的 128x128
        input_var       = input.cuda(cfg.pytorch.gpu, async=True).float().cuda(cfg.pytorch.gpu)
        batch_size = len(input)
        # time begin
        T_begin = time.time()
        # 开始了，开始inference了
        output_conf, output_coor_x, output_coor_y, output_coor_z = model(input_var)
        output_coor_x = output_coor_x.data.cpu().numpy().copy()
        output_coor_y = output_coor_y.data.cpu().numpy().copy()
        output_coor_z = output_coor_z.data.cpu().numpy().copy()
        outConf = output_conf.data.cpu().numpy().copy()
        output_trans = np.zeros(batch_size)
        # collect存放网络的输出，还有测试集的信息
        collector = list(zip(clsIdx.numpy(), output_coor_x, output_coor_y, output_coor_z, outConf,
                                pose.numpy(), bbox.numpy(), center.numpy(), size.numpy(), input.numpy(), scene_id.numpy(), image_id.numpy(), score.numpy()))
        colLen = len(collector)
        for idx in range(colLen):
            # 这个size_怎么来的？ nus
            # scene_id_是啥？render相关？
            # 这个size和bbox应该是已经做了DZI前期处理后的产物，检测那一块已经帮着做完了
            # 这之后的操作就是得出一个框后resize到128x128
            clsIdx_, output_coor_x_, output_coor_y_, output_coor_z_, output_conf_, pose_gt, bbox_, center_, size_, input_, scene_id_, image_id_, score_ = collector[idx]                
            if cfg.pytorch.dataset.lower()  == 'lmo':
                cls = ref.lmo_id2obj[int(clsIdx_)]
            elif cfg.pytorch.dataset.lower() == 'tless':
                cls = ref.tless_id2obj[int(clsIdx_)]
            elif cfg.pytorch.dataset.lower() == 'ycbv':
                cls = ref.ycbv_id2obj[int(clsIdx_)]
            elif cfg.pytorch.dataset.lower() == 'tudl':
                cls = ref.tudl_id2obj[int(clsIdx_)]
            elif cfg.pytorch.dataset.lower() == 'hb':
                cls = ref.hb_id2obj[int(clsIdx_)]
            elif cfg.pytorch.dataset.lower() == 'icbin':
                cls = ref.icbin_id2obj[clsIdx_]
            elif cfg.pytorch.dataset.lower() == 'itodd':
                cls = ref.itodd_id2obj[int(clsIdx_)]

            # cls代表类的名字
            select_pts_2d = []
            select_pts_3d = []

            center_h = center_[0]
            center_w = center_[1]

            size_ = int(size_)
            output_coor_x_ = output_coor_x_.squeeze()
            output_coor_y_ = output_coor_y_.squeeze()
            output_coor_z_ = output_coor_z_.squeeze()
            # 全都是64啥意思啊

            # 找到属于的bin
            output_coor_ = np.stack([np.argmax(output_coor_x_, axis=0),
                                     np.argmax(output_coor_y_, axis=0),
                                     np.argmax(output_coor_z_, axis=0)], axis=2)
            # 结果都是0 不过我感觉
            # cfg.network.coor_bin = 64
            # 把最后一个当成和index = 0相同的
            # print("coor_bin",cfg.network.coor_bin)
            # 最大63，最小0
            output_coor_[output_coor_ == cfg.network.coor_bin] = 0
            # output x before (65, 128, 128)
            #
            # output x after (65, 128, 128)
            # output all (128, 128, 3)
            # 这里的128应该是说裁剪出来后统一 resize成 (128,128)

            # 这是什么操作
            # 这里可能是关于bin的后续操作，或许每个类都有个范围
            # 这个clsIdx就是需要的先验的类，同时还需要先验的min_x,min_y,min_z信息

            # 所以说，是不是提前就知道了某类物体3D模型的点云位置信息
            # 然后预测的xyz其实是比例，就是在点云中的位置，由此得到2D点在3D空间的位置


            output_coor_ = 2.0 * output_coor_ / float(cfg.network.coor_bin - 1) - 1.0      # [-1,1]
            #
            # print('bili ? max',np.max(output_coor_))
            # print('bili ? min',np.min(output_coor_))
            # print('min_x',models_info[clsIdx_]['min_x'])
            # 这个min还不知道怎么搞
            print('cls',clsIdx_)
            print('mdels',models_info)
            print('output',output_coor_[:, :, 0])

            output_coor_[:, :, 0] = output_coor_[:, :, 0] * abs(models_info[clsIdx_]['min_x'])
            output_coor_[:, :, 1] = output_coor_[:, :, 1] * abs(models_info[clsIdx_]['min_y'])
            output_coor_[:, :, 2] = output_coor_[:, :, 2] * abs(models_info[clsIdx_]['min_z'])
            # print('output coor',output_coor_[:, :,)
            # 绝壁写错了
            #print('output_conf before',np.shape(output_conf_))
            #print('output_conf 1 ',output_conf_[:,0,0])
            # if np.abs(output_conf_[0,0,0]) < np.abs(output_conf_[1,0,0]):
            #     print('1 > 0')
            output_conf_ = np.argmax(output_conf_, axis=0)
            #print('output_conf 2 ',output_conf_)
            #print('output_conf 2 ',)


            # print('output_conf size',np.shape(output_conf_))
            # print('output_conf',np.shape(np.nonzero(output_conf_)))
            # 这都把类找出来了，还用归一化操作? 拿着类的index做归一化？
            # 感觉这一步没什么用
            output_conf_ = (output_conf_ - output_conf_.min()) / (output_conf_.max() - output_conf_.min())
            # print('out put ',output_conf_)
            # print('output_conf max ',output_conf_.max())
            # print('output_conf min ',output_conf_.min())

            #print('coor',output_coor_[:, :, 2])
            min_x = 0.001 * abs(models_info[clsIdx_]['min_x'])
            min_y = 0.001 * abs(models_info[clsIdx_]['min_y'])
            min_z = 0.001 * abs(models_info[clsIdx_]['min_z'])
            #print('min_x',min_x)
            # resize到128x128的操作
            # 找到crop图像左下角坐标
            coor = np.array((0,0,0))
            coor[0] = abs(models_info[clsIdx_]['min_x'])
            coor[1] = abs(models_info[clsIdx_]['min_y'])
            coor[2] = abs(models_info[clsIdx_]['min_z'])

            eight_point = np.zeros((8,3))
            for i in range(0, 8):
                coor[2] = -coor[2]
                if (i ) % 4 == 0:
                    coor[0] = -coor[0]
                if (i ) % 2 == 0:
                    coor[1] = -coor[1]
                # print('eight_point',np.shape(eight_point))

                eight_point[i, 0] = coor[0]
                eight_point[i, 1] = coor[1]
                eight_point[i, 2] = coor[2]




            w_begin = center_w - size_ / 2.
            h_begin = center_h - size_ / 2.
            # 比例
            # 不过有比例的话，要怎么对应呢？
            w_unit = size_ * 1.0 / cfg.dataiter.rot_output_res
            h_unit = size_ * 1.0 / cfg.dataiter.rot_output_res
            output_conf_ = output_conf_.tolist()
            output_coor_ = output_coor_.tolist()
            # 遍历resize后的每个像素
            # 这个mask阈值是0.5 都变成index了，这个阈值有啥用？
            for x in range(cfg.dataiter.rot_output_res):
                for y in range(cfg.dataiter.rot_output_res):
                    # 把下面的去掉，应该就是全部的像素都参与计算了
                    # 这里还是按着128遍历的，所mask变成128x128之后，就不用管了
                    if use_mask:
                        if output_conf_[x][y] < cfg.test.mask_threshold:
                            continue
                    # 控制 xyz 在一个范围，如果出范围了就不要
                    # 所以出范围是不是就是错了，就是object外边的点？ nus
                    # 是否就说明坐标预测的并不好
                    # 这里应该是和0做比较，如果是原点，就不要这个点了？
                    if abs(output_coor_[x][y][0]) < min_x  and abs(output_coor_[x][y][1]) < min_y  and \
                        abs(output_coor_[x][y][2]) < min_z:
                        continue
                    # 现在知道了，这些w，x，y，h的变换都是为了让被crop出来的图像位置能对应上原图位置
                    # 不是整数的像素要怎么算，插值？


                    # 跟论文不一样，这里的w_begin应该不是C_x吧？还少了很多东西啊 T_z/f_x ？、
                    # 就是把框变成了128x128，中心点还是一样的，边界变了，把原本的像素缩放 不如说w_unit = 0.35，那就是原图的0.35个像素做128x128图像中的1个像素
                    # 这里好像也没看到保证长宽比的意思？
                    # print('output_coor_[x][y]',output_coor_[x][y])
                    select_pts_2d.append([w_begin + y * w_unit, h_begin + x * h_unit])
                    select_pts_3d.append(output_coor_[x][y])


            # 有了T_G，并没有预测T_S
            # 这个代码里并没有使用SITE算法。
            # 现在有了网络输出的2D点对应的3D点

            #这里加mask判断，可以想


            model_points = np.asarray(select_pts_3d, dtype=np.float32)
            image_points = np.asarray(select_pts_2d, dtype=np.float32)


            try:
                # 这个inlier干啥的？

                _, R_vector, T_vector, inliers = cv2.solvePnPRansac(model_points, image_points,
                                        cfg.pytorch.camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
                cur_wall_time = time.time() - T_begin
                wall_time += cur_wall_time
                R_matrix = cv2.Rodrigues(R_vector, jacobian=0)[0]       
                if R_matrix[0,0] == 1.0: 
                    continue         
                if cfg.pytorch.exp_mode == 'val':       
                    pose_est = np.concatenate((R_matrix, np.asarray(T_vector).reshape(3, 1)), axis=1)         
                    Eval.pose_est_all[cls].append(pose_est)
                    Eval.pose_gt_all[cls].append(pose_gt)
                    Eval.num[cls] += 1
                    Eval.numAll += 1
                elif cfg.pytorch.exp_mode == 'test': 
                    rst = {'scene_id': int(scene_id_), 'im_id': int(image_id_), 'R': R_matrix.reshape(-1).tolist(), 't': T_vector.reshape(-1).tolist(),
                           'score': float(score_), 'obj_id': int(clsIdx), 'time': cur_wall_time}
                    rst_collect.append(rst)
            except:
                if cfg.pytorch.exp_mode == 'val':
                    Eval.num[cls] += 1
                    Eval.numAll += 1                
        Bar.suffix = '{0} [{1}/{2}]| Total: {total:} | ETA: {eta:}'.format(cfg.pytorch.exp_mode, i, nIters, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()

    scale = scale.numpy()
    filepath = "../results_save/mask_{}_scale_{}.jpg".format(use_mask,scale)
    img = imread('D:\jieya\pku_test\cdpn\dataset\\test\\rgb\\000003.png')

    img = visualize_all(img,R_matrix, np.asarray(T_vector).reshape(3, 1),ref.lmo_camera_matrix ,eight_point,line_show_3D=True)
    #fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    #axes[0].imshow(img)
    img = np.array(img[:, :, ::-1])
    #cv2.imwrite(filepath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imshow('mask:{},scale={}'.format(use_mask,scale),img)
    cv2.waitKey(200000)
    cv2.destroyAllWindows()

    if cfg.pytorch.exp_mode == 'val':
        Eval.evaluate_pose()
    elif cfg.pytorch.exp_mode == 'test':
        for item in rst_collect:
            csv_writer.writerow(item)
        csv_file.close()
    print("Wall time of object {}: total {} seconds for {} samples".format(cfg.pytorch.object, wall_time, nIters))
    bar.finish()
