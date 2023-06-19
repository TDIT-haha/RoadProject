import os
import cv2
import numpy as np
import torch
from .models.yolo import Model
from .models.experimental import attempt_load
from .utils.augmentations import letterbox
from .utils.general import non_max_suppression, scale_coords
import time
import torch.nn as nn

class AnimalDetect:
    def __init__(self, modelpath=None, className=None, size=None, device=None, conf_thres=0.5, iou_thres=0.5): #必须要有一个self参数，
        self.modelpath = modelpath
        self.className = className
        self.img_size = size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

    def modelinit(self):
        print("1. init detect model ......")
        self.model = attempt_load(self.modelpath, map_location=self.device, inplace=True, fuse=True)

    def modelwarmup(self):
        print("2. set detect model warmup ......")
        blob = torch.rand(1, 3, self.img_size, self.img_size).to(self.device)
        for i in range(2):
           _ = self.model_inter(blob)  

    def preprocess(self, im0):
        im, ratio, (dw, dh) = letterbox(im0, self.img_size, stride=32, auto=False)  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = np.expand_dims(im, 0).astype(np.float32)
        im = im/255.0
        im = torch.from_numpy(im).to(self.device)
        return im

    def postprocess(self, outpout):
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        classes = None
        agnostic_nms = False
        max_det = 1000

        pred = non_max_suppression(outpout, conf_thres, iou_thres, classes, agnostic_nms)

        results = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords([self.img_size, self.img_size], det[:, :4], self.im.shape[:2]).round()

            for *xyxy, conf, cls in reversed(det):
                x1,y1,x2,y2 = xyxy
                clsName = self.className[int(cls)]
                results.append([[x1,y1,x2,y2], conf, clsName])

        return results

    def model_inter(self, input_data):
        with torch.no_grad():
            output = self.model(input_data)[0]
        return output

    def inter(self, image):
        self.im = image.copy()
        blob = self.preprocess(image)
        output = self.model_inter(blob)
        results = self.postprocess(output)
        return results
        


if __name__=="__main__":
    device = torch.device('cuda')
    modelpath = r"/root/project/Modules/TrackAnomalyTask/all_code/TrackProject/pretrains/detect/yolov5lEfficientLite/best.pt"
    className = ["bird", "cat", "dog", "horse", "sheep", "cow", "fox", "hog", "wolf", "paguma"]
    model = AnimalDetect(modelpath=modelpath, className=className, size=416, device=device)
    model.modelinit()
    model.modelwarmup()

    imagepath = r"/root/project/Datas/otherDatas/dataset_n/detect/images/018-2.jpg"
    image = cv2.imread(imagepath)

    dets = model.inter(image)
    for det_, conf_, name_ in dets:
        x1,y1,x2,y2 = det_
        cv2.putText(image, "name:{}".format(name_), (int(x1), int(y1)), 1, 1, (255, 255, 255), 2)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,255,0))
    
    cv2.imwrite("tmp.jpg", image)
    
    
#   # load FP32 model






