import argparse
import os
import sys
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import glob
import copy
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

from seg.model.unetpp import NestedUNet
from seg.segpredict import predict,contourIntersect
from img_simility import  image_similarity_vectors_via_numpy

videonames = ['Anomalies']
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


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',segweights = ROOT / 'unet++_8_liver_15.pth',source=ROOT / 'data/images',
        refpic=ROOT / 'ref',imgsz=640,conf_thres=0.25,iou_thres=0.45,max_det=1000,device='', view_img=False,
        save_txt=False, save_conf=False, save_crop=False,nosave=False,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,augment=False,visualize=False,update=False,  # update all models
        project=ROOT / 'runs/detect',name='exp',exist_ok=False,
        half=False):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA


    # Load seg model
    seg_model = NestedUNet(3, 1).to(device)
    seg_model.load_state_dict(torch.load(segweights, map_location='cpu'))  # 载入训练好的模型
    seg_model.eval()

    # Load  det model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    t_all = []
    for path, img, im0s, vid_cap in dataset:
        # seg
        seg_orgimg = copy.deepcopy(im0s)
        h,w = im0s.shape[0],im0s.shape[1]
        try:
            contours_list, draw_mask = predict(seg_model, seg_orgimg, device)
        except:
            continue
        # seg

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        t1 = time.time()
        pred = model(img, augment=augment, visualize=visualize)[0]
        t2 = time.time()
        t_all.append(t2 - t1)

        # FPS for inference
        print('average time:', np.mean(t_all) / 1)
        print('average fps:', 1 / np.mean(t_all))

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        is_intrude = True
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            s += '%gx%g ' % img.shape[2:]
            if not len(det): is_intrude = False
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for j in range(det.size()[0]):
                    class_num = int(det[j, 5].cpu().numpy())
                    if class_num ==5:
                        print('fox')
                    xyxy = det[j, :5]
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])
                    score = xyxy[4]
                    flag = False
                    if  score<=0.7:
                        bundingbox = [x1, y1, x2, y2]
                        result_box_img = im0[bundingbox[1]:bundingbox[3], bundingbox[0]:bundingbox[2], :]
                        result_box_img = Image.fromarray(cv2.cvtColor(result_box_img, cv2.COLOR_BGR2RGB))
                        for refpath in glob.glob(str(refpic) + '/'+ str(class_num)+'*.png'):
                            refimg = Image.open(refpath).convert('RGB')
                            sim = image_similarity_vectors_via_numpy(result_box_img, refimg)
                            if sim > 0.62:
                                flag = True
                                break
                            else:
                                continue
                    if not flag:
                        is_intrude = False
                        break
                    box = [x1, y1, x2, y2, class_num]
                    cent = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    Distances = [y1, x1, (w - x2), (h - y2)]
                    min_dis = min(Distances)
                    draw_box(im0, box, videonames[0])

                    tmp_contours = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                    tmp_contours = np.expand_dims(tmp_contours, axis=1)
                    for contours in contours_list:
                        intersection = contourIntersect(draw_mask, contours, tmp_contours, cent)
                        if intersection: is_intrude = True
            if is_intrude:
                cv2.putText(im0, 'Intrude', (int(w / 3), 50), 0, 1, (0, 0, 255), 1)
            # elif len(pred[0]):
            else:
                cv2.putText(im0, 'Alarm', (int(w / 3), 100), 0, 1, (0, 0, 255), 1)


            # Save results (image with detections)
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--segweights', nargs='+', type=str, default=ROOT / 'unet++_8_liver_15.pth', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'testimgs', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--refpic', type=str, default=ROOT / 'ref', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
