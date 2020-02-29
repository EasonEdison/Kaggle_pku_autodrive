import time
import torch
import numpy as np

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

# 感觉还是要学一些cfg怎么用
# 要使用backward() 要一直用torch进行计算
def train(optimizer,cfg,model,dataloader,epoch,history=None):
    # model已处理
    model.train()
    model_min_x = -1.023
    model_min_y = -0.8
    model_min_z = -2.31
    # 读取数据集 为什么kaggle kernel要用tqdm，我还不清楚
    for i,(input, Rt, bbox, center,size, img_path,scale,mask) in enumerate(dataloader):


        # 应该是双卡处理操作，不过为什么要双cuda？
        # 这里已经是0-1的范围了
        input_var = input.cuda(cfg.pytorch.gpu, async=True).float().cuda(cfg.pytorch.gpu)

        batch_size = len(input)
        T_begin = time.time()
        output_coor_x, output_coor_y, output_coor_z = model(input_var)

        # 这里搞错了，上面的input_var是用来输入的，用过一次model推导之后就没用了
        # 非list才能numpy，所以在.numpy()之前要 np.array()来变成矩阵

        # 12.22晚上要不要重新搞一下

        collector = list(zip(output_coor_x, output_coor_y, output_coor_z, mask.cuda(cfg.pytorch.gpu),
                                Rt.numpy(), bbox.cuda(cfg.pytorch.gpu), center.numpy(), size.numpy(), input.cuda(cfg.pytorch.gpu)))
        colLen = len(collector)
        # print('colLen',colLen)
        for idx in range(colLen):
            output_coor_x_, output_coor_y_, output_coor_z_, input_mask, Rt, bbox_, center_, size_, input_= collector[idx]
            # select_pts_2d = []
            # select_pts_3d = []

            center_h = center_[0]
            center_w = center_[1]

            size_ = int(size_)
            output_coor_x_ = output_coor_x_.squeeze()
            output_coor_y_ = output_coor_y_.squeeze()
            output_coor_z_ = output_coor_z_.squeeze()
            # temp2 = torch.max(output_coor_x_, axis=0)[1].float()
            # temp2.requires_grad = True
            # temp = torch.argmax()
            # shape is 128 128 3
            output_coor_ = torch.stack([output_coor_x_,
                                     output_coor_y_,
                                     output_coor_z_], axis=2)

            # 应该是这个有问题？
            output_coor_t = torch.tanh(output_coor_)
            # 这里要做修改了
            output_coor_[:, :, 0] = output_coor_t[:, :, 0] * abs(model_min_x)
            output_coor_[:, :, 1] = output_coor_t[:, :, 1] * abs(model_min_y)
            output_coor_[:, :, 2] = output_coor_t[:, :, 2] * abs(model_min_z)
            # print('output coor',output_coor_[:, :,:])

            min_x = 0.001 * abs(model_min_x)
            min_y = 0.001 * abs(model_min_y)
            min_z = 0.001 * abs(model_min_z)


            # 这些用来可视化

            coor = np.array((0,0,0))
            coor[0] = abs(model_min_x)
            coor[1] = abs(model_min_y)
            coor[2] = abs(model_min_z)

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
            i = 0
            no_init = True
            for x in range(cfg.dataiter.rot_output_res):
                for y in range(cfg.dataiter.rot_output_res):
                    # 根据mask筛点
                    if input_mask[x][y][0] != 255:
                        continue
                    # 控制 xyz 在一个范围，如果出范围了就不要
                    # 所以出范围是不是就是错了，就是object外边的点？ nus
                    # 是否就说明坐标预测的并不好
                    # 这里应该是和0做比较，如果是原点，就不要这个点了？
                    if abs(output_coor_[x][y][0]) < min_x  and abs(output_coor_[x][y][1]) < min_y  and \
                        abs(output_coor_[x][y][2]) < min_z:
                        continue
                    # 现在知道了，这些w，x，y，h的变换都是为了让被crop出来的图像位置能对应上原图位置
                    # 不是整数的像素要怎么算，插值？ 答：不需要像素信息，只需要位置信息

                    # 就是把框变成了128x128，中心点还是一样的，边界变了，把原本的像素缩放 不如说w_unit = 0.35，那就是原图的0.35个像素做128x128图像中的1个像素
                    if no_init:
                        no_init = False
                        select_pts_2d = torch.Tensor([w_begin + y * w_unit, h_begin + x * h_unit])
                        select_pts_3d = output_coor_[x][y]
                        # print('2d ',select_pts_2d)
                        # print('3d ',select_pts_3d)
                    else:
                        select_pts_2d = torch.cat((select_pts_2d,torch.Tensor([w_begin + y * w_unit, h_begin + x * h_unit])),dim=0)

                        select_pts_3d = torch.cat((select_pts_3d,output_coor_[x][y]),dim=0)
                    # select_pts_2d.append(torch.Tensor([w_begin + y * w_unit, h_begin + x * h_unit]))
                    # print('list to tensro',torch.Tensor([w_begin + y * w_unit, h_begin + x * h_unit]))
                    # select_pts_3d.append(output_coor_[x][y])
                    # print('out_coor',output_coor_[x][y])

            # print('select 2d',select_pts_2d.view(-1,2).size())
            # print('select 3d',select_pts_3d.view(-1,3).size())
            # 这个在算loss的时候用不到，但是可以每epoch多少次之后，输出一个图片看看效果


            # 这里没有改动，用的还是torch的list
            model_points = select_pts_3d.view(-1,3)
            # 使用了tensor.Tensor之后，image_point 变成了tensor([518.0912, 326.2367])

            image_points = select_pts_2d.view(-1,2)
            # print('img_points',image_points)

            # 两种算loss方法,作者的loss还没加平均值
            # 用真实RT算出来的3D点
            is_init = True
            for i in range(len(image_points)):
                # print('rt 逆矩阵',np.shape(np.linalg.pinv(Rt)))
                # # 这里要改成自己的内参，搞错了
                # print('point',np.shape(image_points[i]))
                # print('point',image_points[i])
                # 因为原本映射出来的就是缩放的，所以现在这个也要缩放后再做PnP
                # 可能是这里有问题？

                # world_point 已经是tensor了 tensor([9.3667e+02, 1.7712e+03, 1.0000e+00])
                world_point = torch.tensor([image_points[i][0].item() * 3384 / 640, image_points[i][1].item() * 2710 / 480, 1]).view(3,1)
                # print('world',world_point)


                # 把这里都换tensor试试
                RT_inv = torch.tensor(np.linalg.pinv(Rt),dtype=torch.float32)
                # print('RT_inv',RT_inv.size())
                Cam_Mat = torch.tensor(np.linalg.pinv(camera_matrix),dtype=torch.float32)

                # 这里二维点的最后一个维度没有加1
                coor_point = RT_inv.mm(Cam_Mat).mm(world_point)
                #coor_point = torch.mm(torch.mm(RT_inv，Cam_Mat)， world_point）
                # print('coor_point',coor_point)
                #coor_point = np.dot(np.dot(np.linalg.pinv(Rt),np.linalg.pinv(camera_matrix)),world_point)
                # xianzai wentishi
                # print('yuanben',coor_point.tolist())
                # 把算出来的点的第四个数变成1
                coor_point[0] /= coor_point[3]
                coor_point[1] /= coor_point[3]
                coor_point[2] /= coor_point[3]
                # print('what type?',coor_point[:3])

                if is_init:
                    cal_gt_point = torch.Tensor(coor_point[:3]).cuda(cfg.pytorch.gpu)
                    is_init = False
                else:

                    cal_gt_point = torch.cat((cal_gt_point,torch.Tensor(coor_point[:3]).cuda(cfg.pytorch.gpu)),dim=0)

            # 这里立马用到了，copy
            #cal_gt_point = np.asarray(model_gt_points,dtype=np.float32).copy()
            #print('after asarray',cal_gt_point)
            cal_gt_point = cal_gt_point.view(-1,3)
            cal_pd_point = model_points
            # 现在是啥啥都不准，list 里面的是tensor 和tensor 的list
            # 预测出来的3D坐标也都是整数
            # 明天看看numpy版本的是select3d是啥样的，是不是也是整数

            # print('cal_gt_point',cal_gt_point)
            # print('model_gt_points',model_gt_points)
            # tensor([ 1,  0, -2], device='cuda:0')
            # print('cal_pd_point',cal_pd_point)

            # print('len is ',len(cal_gt_points))


            # print('output_coor_x', output_coor_x.size())

            # 证实了，就是tensor出了问题
            # loss = output_coor_x[0][0][0][0]
            # print('cal_pd_point',cal_pd_point)
            # loss = torch.sum(cal_pd_point) / len(cal_pd_point)
            loss = torch.sum(abs(cal_gt_point - cal_pd_point)) / len(cal_pd_point)
            # loss = torch.sum(abs(cal_pd_point)) / len(cal_pd_point)
            # 挨着个试，看那个出问题了

            if history is not None:
                history.loc[epoch,'trian_loss'] = loss.item()

            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
            #print('pred',model_points[0])
            # for i in range(len(cal_gt_point)):
            #     print('xiangjian ',cal_gt_point - model_gt_points)
            if epoch==0:
                print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
                    epoch,
                    optimizer.state_dict()['param_groups'][0]['lr'],
                    loss.item()))

    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.item()))

def eval(optimizer,cfg,model,dataloader,epoch,history=None):
    # model已处理
    model.eval()
    loss_mean = 0
    with torch.no_grad():
        model_min_x = -1.023
        model_min_y = -0.8
        model_min_z = -2.31
        # 读取数据集 为什么kaggle kernel要用tqdm，我还不清楚
        for i,(input, Rt, bbox, center,size, img_path,scale,mask) in enumerate(dataloader):


            # 应该是双卡处理操作，不过为什么要双cuda？
            # 这里已经是0-1的范围了
            input_var = input.cuda(cfg.pytorch.gpu, async=True).float().cuda(cfg.pytorch.gpu)

            batch_size = len(input)
            T_begin = time.time()
            output_coor_x, output_coor_y, output_coor_z = model(input_var)

            # 这里搞错了，上面的input_var是用来输入的，用过一次model推导之后就没用了
            # 非list才能numpy，所以在.numpy()之前要 np.array()来变成矩阵

            # 12.22晚上要不要重新搞一下

            collector = list(zip(output_coor_x, output_coor_y, output_coor_z, mask.cuda(cfg.pytorch.gpu),
                                    Rt.numpy(), bbox.cuda(cfg.pytorch.gpu), center.numpy(), size.numpy(), input.cuda(cfg.pytorch.gpu)))
            colLen = len(collector)
            # print('colLen',colLen)
            for idx in range(colLen):
                output_coor_x_, output_coor_y_, output_coor_z_, input_mask, Rt, bbox_, center_, size_, input_= collector[idx]
                # select_pts_2d = []
                # select_pts_3d = []

                center_h = center_[0]
                center_w = center_[1]

                size_ = int(size_)
                output_coor_x_ = output_coor_x_.squeeze()
                output_coor_y_ = output_coor_y_.squeeze()
                output_coor_z_ = output_coor_z_.squeeze()
                # temp2 = torch.max(output_coor_x_, axis=0)[1].float()
                # temp2.requires_grad = True
                # temp = torch.argmax()
                test_x = torch.argmax(output_coor_x_, axis=0).float()
                test_y = torch.argmax(output_coor_y_, axis=0).float()
                test_z = torch.argmax(output_coor_z_, axis=0).float()
                test_x.requires_grad = True
                test_y.requires_grad = True
                test_z.requires_grad = True
                output_coor_ = torch.stack([test_x,
                                         test_y,
                                         test_z], axis=2)
                output_coor_[output_coor_ == cfg.network.coor_bin] = 0
                output_coor_ = 2.0 * output_coor_.float() / 63.0 - 1.0      # [-1,1]

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

                # tensor 的int型 / python的float 不会变成float型

                # print('output 0 ',output_coor_.float())

                # 这里要做修改了
                output_coor = torch.zeros_like(output_coor_)
                output_coor[:, :, 0] = output_coor_[:, :, 0] * abs(model_min_x)
                output_coor[:, :, 1] = output_coor_[:, :, 1] * abs(model_min_y)
                output_coor[:, :, 2] = output_coor_[:, :, 2] * abs(model_min_z)
                # print('output coor',output_coor_[:, :,:])

                min_x = 0.001 * abs(model_min_x)
                min_y = 0.001 * abs(model_min_y)
                min_z = 0.001 * abs(model_min_z)


                # 这些用来可视化

                coor = np.array((0,0,0))
                coor[0] = abs(model_min_x)
                coor[1] = abs(model_min_y)
                coor[2] = abs(model_min_z)

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
                i = 0
                no_init = True
                for x in range(cfg.dataiter.rot_output_res):
                    for y in range(cfg.dataiter.rot_output_res):
                        # 根据mask筛点
                        if input_mask[x][y][0] != 255:
                            continue
                        # 控制 xyz 在一个范围，如果出范围了就不要
                        # 所以出范围是不是就是错了，就是object外边的点？ nus
                        # 是否就说明坐标预测的并不好
                        # 这里应该是和0做比较，如果是原点，就不要这个点了？
                        if abs(output_coor[x][y][0]) < min_x  and abs(output_coor[x][y][1]) < min_y  and \
                            abs(output_coor[x][y][2]) < min_z:
                            continue
                        # 现在知道了，这些w，x，y，h的变换都是为了让被crop出来的图像位置能对应上原图位置
                        # 不是整数的像素要怎么算，插值？ 答：不需要像素信息，只需要位置信息

                        # 就是把框变成了128x128，中心点还是一样的，边界变了，把原本的像素缩放 不如说w_unit = 0.35，那就是原图的0.35个像素做128x128图像中的1个像素
                        if no_init:
                            no_init = False
                            select_pts_2d = torch.Tensor([w_begin + y * w_unit, h_begin + x * h_unit])
                            select_pts_3d = output_coor[x][y]
                            # print('2d ',select_pts_2d)
                            # print('3d ',select_pts_3d)
                        else:
                            select_pts_2d = torch.cat((select_pts_2d,torch.Tensor([w_begin + y * w_unit, h_begin + x * h_unit])),dim=0)

                            select_pts_3d = torch.cat((select_pts_3d,output_coor[x][y]),dim=0)
                        # select_pts_2d.append(torch.Tensor([w_begin + y * w_unit, h_begin + x * h_unit]))
                        # print('list to tensro',torch.Tensor([w_begin + y * w_unit, h_begin + x * h_unit]))
                        # select_pts_3d.append(output_coor_[x][y])
                        # print('out_coor',output_coor_[x][y])

                # print('select 2d',select_pts_2d.view(-1,2).size())
                # print('select 3d',select_pts_3d.view(-1,3).size())
                # 这个在算loss的时候用不到，但是可以每epoch多少次之后，输出一个图片看看效果


                # 这里没有改动，用的还是torch的list
                model_points = select_pts_3d.view(-1,3)
                # 使用了tensor.Tensor之后，image_point 变成了tensor([518.0912, 326.2367])

                image_points = select_pts_2d.view(-1,2)
                # print('img_points',image_points)

                # 两种算loss方法,作者的loss还没加平均值
                # 用真实RT算出来的3D点
                is_init = True
                for i in range(len(image_points)):
                    # print('rt 逆矩阵',np.shape(np.linalg.pinv(Rt)))
                    # # 这里要改成自己的内参，搞错了
                    # print('point',np.shape(image_points[i]))
                    # print('point',image_points[i])
                    # 因为原本映射出来的就是缩放的，所以现在这个也要缩放后再做PnP
                    # 可能是这里有问题？

                    # world_point 已经是tensor了 tensor([9.3667e+02, 1.7712e+03, 1.0000e+00])
                    world_point = torch.tensor([image_points[i][0].item() * 3384 / 640, image_points[i][1].item() * 2710 / 480, 1]).view(3,1)
                    # print('world',world_point)


                    # 把这里都换tensor试试
                    RT_inv = torch.tensor(np.linalg.pinv(Rt),dtype=torch.float32)
                    # print('RT_inv',RT_inv.size())
                    Cam_Mat = torch.tensor(np.linalg.pinv(camera_matrix),dtype=torch.float32)

                    # 这里二维点的最后一个维度没有加1
                    coor_point = RT_inv.mm(Cam_Mat).mm(world_point)
                    #coor_point = torch.mm(torch.mm(RT_inv，Cam_Mat)， world_point）
                    # print('coor_point',coor_point)
                    #coor_point = np.dot(np.dot(np.linalg.pinv(Rt),np.linalg.pinv(camera_matrix)),world_point)
                    # xianzai wentishi
                    # print('yuanben',coor_point.tolist())
                    # 把算出来的点的第四个数变成1
                    coor_point[0] /= coor_point[3]
                    coor_point[1] /= coor_point[3]
                    coor_point[2] /= coor_point[3]
                    # print('what type?',coor_point[:3])

                    if is_init:
                        cal_gt_point = torch.Tensor(coor_point[:3]).cuda(cfg.pytorch.gpu)
                        is_init = False
                    else:

                        cal_gt_point = torch.cat((cal_gt_point,torch.Tensor(coor_point[:3]).cuda(cfg.pytorch.gpu)),dim=0)

                # 这里立马用到了，copy
                #cal_gt_point = np.asarray(model_gt_points,dtype=np.float32).copy()
                #print('after asarray',cal_gt_point)
                cal_gt_point = cal_gt_point.view(-1,3)
                cal_pd_point = model_points
                # 现在是啥啥都不准，list 里面的是tensor 和tensor 的list
                # 预测出来的3D坐标也都是整数
                # 明天看看numpy版本的是select3d是啥样的，是不是也是整数

                # print('cal_gt_point',cal_gt_point)
                # print('model_gt_points',model_gt_points)
                # tensor([ 1,  0, -2], device='cuda:0')
                # print('cal_pd_point',cal_pd_point)

                # print('len is ',len(cal_gt_points))


                # print('output_coor_x', output_coor_x.size())

                # 证实了，就是tensor出了问题
                # loss = output_coor_x[0][0][0][0]
                # print('cal_pd_point',cal_pd_point)
                # loss = torch.sum(cal_pd_point) / len(cal_pd_point)
                loss = torch.sum(abs(cal_gt_point - cal_pd_point)) / len(cal_pd_point)
                print('loss',loss_mean)
                loss_mean = loss_mean + loss.item()
                # loss = torch.sum(abs(cal_pd_point)) / len(cal_pd_point)
                # 挨着个试，看那个出问题了

    loss_mean = loss_mean / len(dataloader)
    if history is not None:
        history.loc[epoch,'eval_loss'] = loss_mean

    print('Dev loss: {:.4f}'.format(loss_mean))

