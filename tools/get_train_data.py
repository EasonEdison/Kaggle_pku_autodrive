import os
import csv
import shutil
import numpy as np
import cv2

def show_bbox(img_path,bbox_gt,bbox_pd):
    print('img_path',img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img,(640,480))
    cv2.rectangle(img,(bbox_gt[0],bbox_gt[1]),(bbox_gt[2],bbox_gt[3]),(0,255,0))
    cv2.rectangle(img,(bbox_pd[0],bbox_pd[1]),(bbox_pd[2],bbox_pd[3]),(0,0,255))
    cv2.imshow('aa',img)
    cv2.waitKey(0)
def get_iou(bbox_gt,bbox_predict):

    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox_gt[0], bbox_gt[1], bbox_gt[2], bbox_gt[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox_predict[0], bbox_predict[1], bbox_predict[2], bbox_predict[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0,640) * np.clip(
        inter_rect_y2 - inter_rect_y1 + 1, 0,640)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


dir_gt_bbox = 'D:\\jieya\\pku_test\\train_bbox'
dir_gt_rt = 'D:\\jieya\\pku_test\\train_rt'
dir_pd_bbox = 'D:\\jieya\\pku_test\\cdpn\\dataset\\save_box_mask\\bbox'
dir_pd_mask = 'D:\\jieya\\pku_test\\cdpn\\dataset\\save_box_mask\\mask'

# 获取图像名称
list_img = os.listdir(dir_gt_bbox)
# print('list',list_img[0])

headers = ['ID_image', 'else']
rows = []

# 对每个照片进行遍历
for i in range(len(list_img)):
    try:
        print('{} / {}'.format(i,len(list_img)))
        bbox_gt_data = np.load(os.path.join(dir_gt_bbox,list_img[i]))
        # print('paht',os.path.join(dir_pd_mask,list_img[i]))
        bbox_pd_data = np.load(os.path.join(dir_pd_bbox,list_img[i]))
        # print('dd',bbox_pd_data[0][:4].astype(int))
        ID_name = list_img[i].split('_')[-1].split('.')[0]

        # 对当前图片的每个bbox进行遍历
        item = []
        for j in range(len(bbox_gt_data)):
            max_iou = 0
            max_j = 0
            max_k = 0
            # 对当前图片的每个预测bbox进行遍历
            for k in range(len(bbox_pd_data)):
                curr_iou = get_iou(bbox_gt_data[j],bbox_pd_data[k][:4])
                if curr_iou > max_iou:
                    max_iou = curr_iou
                    max_j = j
                    max_k = k

            if max_iou > 0.3:
                # 获取对应的mask的名字
                # print('ID',ID_name)
                mask_path = 'mask_ID_{}_{}.jpg'.format(ID_name,max_k)
                # print('mask_path',os.path.join(dir_pd_mask,mask_path))
                # a = cv2.imread(os.path.join(dir_pd_mask,mask_path))
                # cv2.imshow('ing',a)
                # cv2.waitKey(0)

                # 获取保存对应的rt的值
                rt_path = os.path.join(dir_gt_rt,'rt_ID_{}.npy'.format(ID_name))
                rt = np.load(rt_path)[j]
                #print('rt',rt)

                # 预测提供mask和bbox(k),真实提供rt(j)
                # b_m_tt = [bbox_pd_data[max_k],mask_path,rt]
                # b_m_tt = bbox_pd_data[max_k].append(rt)
                d = np.zeros((10))
                # print('shape',np.shape(bbox_pd_data[max_k][:4]))
                # print('shape',np.shape(d[:4]))
                d[:4] = bbox_pd_data[max_k][:4]
                d[4:] = rt
                #item.append(b_m_tt)
                # print('b_m_tt',d)

                img_id = 'ID_{}+{}'.format(ID_name,max_k)
                rows.append((img_id,d.tolist()))
    except:
        continue
with open('trian_data.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

# Iou展示代码
# gt 绿色 pd 红色
# if (max_iou < 0.4) & (max_iou > 0.3):
# # if max_iou>0.25:
#     img_path = os.path.join('D:\\jieya\\pku_test\\train_images','ID_{}.jpg'.format(list_img[i].split('_')[-1].split('.')[0]))
#     print('max',max_iou)
#
#     show_bbox(img_path,bbox_gt_data[max_j],bbox_pd_data[max_k])

