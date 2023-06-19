import copy
import os
import cv2
import torch
from utils.general import non_max_suppression,scale_boxes
import argparse
import numpy as np
from utils.augmentations import classify_transforms
import torch.nn.functional as F
from detect_utils import process_data,getDist_P2PS,getDist_P2L,load_jit_model
from utils.metrics import box_iou
import json
from utils.augmentations import letterbox
from demo_seg import *
import time
from tqdm import tqdm

names = ['Anomalies']
def draw_box(im,box,label,color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(im,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

def detect_imgs(det_net,seg_net,video_path,img_size,device,count):
    # thres
    conf_thres = 0.3
    iou_thres = 0.3

    #video
    fourcc = 'mp4v'  # output video codec
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_path = os.path.join('output', os.path.basename(video_path))
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    while True:
        count +=1
        ret_val, orgimg = cap.read()
        if orgimg is None:continue
        
        t0 = time.time()
        # seg
        seg_orgimg = copy.deepcopy(orgimg)
        contours_list,draw_mask = predict_img(net=seg_net,full_img=seg_orgimg,scale_factor=0.5,out_threshold=0.5,device=device)
        # seg

        fps+=1
        if fps >=length:break
        tensor_img = process_data(orgimg, img_size, device)
        with torch.no_grad(): 
            pred = det_net(tensor_img)
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        is_intrude = False
        for i, det in enumerate(pred):  # detections per image
            if det == None: continue
            if len(det):
                det[:, :4] = scale_boxes(tensor_img.shape[2:], det[:, :4], orgimg.shape).round()
                for j in range(det.size()[0]):
                    class_num = int(det[j, 5].cpu().numpy())
                    xyxy = det[j, :4]
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])
                    box = [x1, y1, x2, y2, class_num]
                    cent = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    Distances = [y1, x1, (w - x2), (h - y2)]
                    min_dis = min(Distances)
                    draw_box(orgimg,box,names[0])

                    tmp_contours = np.array([(x1,y1), (x2,y1), (x2, y2), (x1, y2)])
                    tmp_contours = np.expand_dims(tmp_contours, axis=1)
                    for contours in contours_list:
                        intersection = contourIntersect(draw_mask, contours, tmp_contours,cent)
                        if intersection: is_intrude = True

        t1 = time.time()
        FPS = 1/(t1-t0)
        print("FPS:{:.3f}".format(FPS))
                    
        if is_intrude:cv2.putText(orgimg,'Intrude',(int(w/3),50),0,1,(0,0,255),1)
        elif len(pred[0]):cv2.putText(orgimg,'Alarm',(int(w/3),100),0,1,(0,0,255),1)
        
        cv2.putText(orgimg,'FPS:{:.3f}'.format(FPS),(w-500,100),0,1,(0,0,255),1)

        
        vid_writer.write(orgimg)

    return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/last.torchscript', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    det_net = load_jit_model(device,opt.weights)
    det_net.eval()

    #Load seg
    args = get_args()
    in_files = args.input
    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nn.DataParallel(net)
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()


    all_path = r'/root/project/Datas/otherDatas/TEST_INPUT'
    count = 0
    for video_name in os.listdir(all_path):
        print("video_name:{}".format(video_name))
        paths = os.path.join(all_path, video_name)
        detect_imgs(det_net,net,paths, opt.img_size, device,count)
        
        
