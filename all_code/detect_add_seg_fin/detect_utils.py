from utils.augmentations import letterbox
import torch
import numpy as np
import math
import json

def process_data(img,img_size,device):
    img = letterbox(img, new_shape=img_size,stride=32,auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # contiguous
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    tensor_img = img
    if tensor_img.ndimension() == 3: tensor_img = tensor_img.unsqueeze(0)
    return tensor_img


def getDist_P2P(PointA,PointB):
    distance = math.pow((PointA[0]-PointB[0]),2) + math.pow((PointA[1] - PointB[1]),2)
    distance = math.sqrt(distance)
    return distance

def getDist_P2PS(Point,Points):
    Dises = []
    for point in Points:
        dis = getDist_P2P(Point,point)
        Dises.append(dis)
    return min(Dises)

def getDist_P2L(PointP,Pointa,Pointb):
    A = Pointa[1] - Pointb[1]
    B = Pointb[0] - Pointa[0]
    C = Pointa[0]*Pointb[1] - Pointa[1]*Pointb[0]
    distance = abs(A*PointP[0]+B*PointP[1]+C)/math.sqrt(A*A+B*B)
    return distance

def load_jit_model(device,weights):
    extra_files = {'config.txt': ''}  # model metadata
    net = torch.jit.load(weights, _extra_files=extra_files, map_location=device)
    net.float()
    if extra_files['config.txt']:  # load metadata dict
        d = json.loads(extra_files['config.txt'],object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                              for k, v in d.items()})
    return net