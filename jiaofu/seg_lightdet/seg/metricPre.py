import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
# 超参数，类别数量
class_num = 2

####################
# 计算各种评价指标  #
####################
def fast_hist(a, b, n):
    """
    生成混淆矩阵
    a 是形状为(HxW,)的预测值
    b 是形状为(HxW,)的真实值
    n 是类别数
    """
    # 确保a和b在0~n-1的范围内，k是(HxW,)的True和False数列
    a = torch.softmax(a, dim=1)
    _, a = torch.max(a, dim=1)
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    k = (a >= 0) & (a < n)
    # 横坐标是预测的类别，纵坐标是真实的类别
    hist = np.bincount(a[k].astype(int) + n * b[k].astype(int), minlength=n ** 2).reshape(n, n)
    # print(hist[20])
    return hist


def per_class_acc(hist):
    """
    :param hist: 混淆矩阵
    :return: 没类的acc和平均的acc
    """
    np.seterr(divide="ignore", invalid="ignore")
    acc_cls = np.diag(hist) / hist.sum(1)
    np.seterr(divide="warn", invalid="warn")
    acc_cls[np.isnan(acc_cls)] = 0.
    return acc_cls


# 使用这个函数计算模型的各种性能指标
# 输入网络的输出值和标签值，得到计算结果
def get_pre(mask_name,predict,hist):
    """
    :param pred: 预测向量
    :param label: 真实标签值
    :return: 准确率
    """
    image_mask = cv2.imread(mask_name, 0)
    image_mask = cv2.resize(image_mask, (512, 512))
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask, (576, 576))
    height = predict.shape[0]
    weight = predict.shape[1]
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  # 由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:  # 由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1
            if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                o += 1
    predict = predict.astype(np.int16)

    hist = hist + fast_hist(predict, image_mask, class_num)
    # print(hist[20])
    # 准确率
    acc = np.diag(hist).sum() / hist.sum()
    return acc, hist
