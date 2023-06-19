import os
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
import time
from .models.unetpp import NestedUNet
import torch.nn as nn

class RoadSeg:
    def __init__(self, modelpath=None, size=None, device=None, deepsupervision=True, conf_thres=0.5): #必须要有一个self参数，
        self.modelpath = modelpath
        self.img_size = size
        self.conf_thres = conf_thres
        self.device = device
        self.deepsupervision = deepsupervision 
        self.x_transforms = transforms.Compose([
            transforms.ToTensor(),  # -> [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
        ])

    def modelinit(self):
        print("1. init seg model ......")
        self.model = NestedUNet(self.deepsupervision,3,1).to(self.device)
        self.model.load_state_dict(torch.load(self.modelpath, map_location='cuda'))  # 载入训练好的模型
        self.model = self.model.to(self.device)
        self.model.eval()

    def modelwarmup(self):
        print("2. set seg model warmup ......")
        blob = torch.rand(1, 3, self.img_size, self.img_size).to(self.device)
        for i in range(2):
           _ = self.model_inter(blob)  

    def preprocess(self, im0):
        imgx,imgy=(self.img_size, self.img_size)
        pic = cv2.resize(im0,(imgx,imgy))
        pic = pic.astype('float32') / 255
        pic = self.x_transforms(pic)
        pic = pic.unsqueeze(0)
        pic = pic.to(self.device)
        return pic

    def postprocess(self, predict):
        # 模型推理
        if self.deepsupervision:
            # predict = predict[-1].detach().cpu().numpy()[0]
            predict = predict[-1][0]
        else:
            # predict = predict.detach().cpu().numpy()[0]
            predict = predict[0]
            
        return predict

    def model_inter(self, input_data):
        output = self.model(input_data)
        return output

    def inter(self, image):
        blob = self.preprocess(image)
        output = self.model_inter(blob)
        predict = self.postprocess(output)
        return predict
        


if __name__=="__main__":
    device = torch.device('cuda')
    modelpath = r"/root/project/Modules/TrackAnomalyTask/all_code/TrackProject/pretrains/seg/unet++_4_roadseg_200.pth"
    model = RoadSeg(modelpath=modelpath, size=576, device=device)
    model.modelinit()
    model.modelwarmup()

    imagepath = r"/root/project/Datas/otherDatas/dataset_n/detect/images/018-2.jpg"
    image = cv2.imread(imagepath)
    h_, w_ = image.shape[:2]

    predict = model.inter(image)
    
    predict = predict.transpose(1,2,0)
    predict = np.repeat(predict, 3, axis=2)
    predict[predict>=0.5] = 255
    predict[predict<0.5] = 0
    predict = predict.astype("uint8")
    # print(predict.shape)
    predict[:,:,0][predict[:,:,0]==255] = 0
    predict[:,:,1][predict[:,:,1]==255] = 255
    predict[:,:,2][predict[:,:,2]==255] = 0
    predict = cv2.resize(predict,(w_,h_))
    
    
    # # 添加透明显示
    alpha = 0.4
    beta = 0.8
    pt_image = cv2.addWeighted(predict, alpha, image , beta, 0)
    cv2.imwrite("mask_pt.jpg", pt_image)
    
    
#   # load FP32 model






