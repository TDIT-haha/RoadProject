from detect.detect_demo import AnimalDetect
from seg.seg_demo import RoadSeg

import torch
import numpy as np
import cv2
import os
import time


device = torch.device('cuda')
# 初始化检测模型
detect_modelpath = r"/root/project/Modules/TrackAnomalyTask/jiaofu/seg_lightdet/runs/train/yolov5l/exp/weights/best.pt"
className = ["bird", "cat", "dog", "horse", "sheep", "cow", "fox", "hog", "wolf", "paguma"]
detect_model = AnimalDetect(modelpath=detect_modelpath, className=className, size=640, device=device, conf_thres=0.5)
detect_model.modelinit()
detect_model.modelwarmup()

# 初始化分割模型
seg_modelpath = r"/root/project/Modules/TrackAnomalyTask/all_code/TrackProject/pretrains/seg/unet++_4_roadseg_200.pth"
seg_model = RoadSeg(modelpath=seg_modelpath, size=576, device=device)
seg_model.modelinit()
seg_model.modelwarmup()


# videopath = r"/root/project/Datas/otherDatas/TEST_INPUT/1-2-1.mp4"
# videopath = r"/root/project/Datas/otherDatas/TEST_INPUT/1-2-2.mp4"
# videopath = r"/root/project/Datas/otherDatas/TEST_INPUT/1-2-3.mp4"
# videopath = r"/root/project/Datas/otherDatas/TEST_INPUT/1-2-4.mp4"
videopath = r"/root/project/Datas/otherDatas/TEST_INPUT/NIGHT-1.mp4"
videopath = r"/root/project/Datas/otherDatas/TEST_INPUT/NIGHT-2.mp4"
videopath = r"/root/project/Datas/otherDatas/TEST_INPUT/RAIN.mp4"
cap = cv2.VideoCapture(videopath)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = 'mp4v'  # output video codec
save_path = "results/videos/predict_{}".format(os.path.basename(videopath))
vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

setdraw = False
while True:
    ret, image = cap.read()
    if not ret:
        break

    h_, w_ = image.shape[:2]
    status = "None"
    
    t0 = time.time()
    predict = seg_model.inter(image)
    predict = predict.detach().cpu().numpy()
    predict = predict.transpose(1,2,0)
    predict = np.repeat(predict, 3, axis=2)
    predict[predict>=0.5] = 255
    predict[predict<0.5] = 0
    predict = predict.astype("uint8")
    predict[:,:,0][predict[:,:,0]==255] = 1
    predict[:,:,1][predict[:,:,1]==255] = 1
    predict[:,:,2][predict[:,:,2]==255] = 1
    predict = cv2.resize(predict,(w_,h_))
    
    dets = detect_model.inter(image)
    if len(dets)>0:
        image01 = image.copy()*0
        for det_, conf_, name_ in dets:
            x1,y1,x2,y2 = det_
            # if setdraw:
            cv2.putText(image, "Anomalies", (int(x1), int(y1)), 1, 1, (255, 255, 255), 2)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,255,0))
            cv2.rectangle(image01, (int(x1), int(y1)), (int(x2), int(y2)), color=(1,1,1), thickness=-1)
            
        intersection = np.logical_and(image01,predict)
        if intersection.any():
            status = "Intrude"
        else:
            status = "Alarm"
            
    t1 = time.time()
    FPS = 1/(t1-t0)
    
    if status == "Intrude":
        status_color = (255,255,0)
        cv2.putText(image, "{}".format(status), (100, 100), 1, 5, status_color, 2)
    elif status == "Alarm":
        status_color = (0,255,255)
        cv2.putText(image, "{}".format(status), (100, 100), 1, 5, status_color, 2)
    else:
        status_color = (255,255,255)
        
    
    cv2.putText(image, "FPS={:.2f}".format(FPS), (w_-300, 100), 1, 2, (255,255,255), 2)
    # predict[:,:,0][predict[:,:,0]==1] = 0
    # predict[:,:,1][predict[:,:,1]==1] = 255
    # predict[:,:,2][predict[:,:,2]==1] = 0
    # # # 添加透明显示
    # alpha = 0.4
    # beta = 0.8
    # image = cv2.addWeighted(predict, alpha, image , beta, 0)
    cv2.imwrite("haha.jpg", image)
    vid_writer.write(image)
    print("now status: {}".format(status))  
    print("FPS={:.2f}".format(FPS))











