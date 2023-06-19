import argparse
import copy
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
from utils_seg.data_loading import BasicDataset
from unet import UNet
import time


colormap = [(0, 0, 0), (255, 255, 255), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]
ndcolormap = np.array(colormap)


def contourIntersect(org_img,contour1,contour2,cent):
    is_ok = False
    contours = [contour1,contour2]
    blank = np.zeros(org_img.shape[0:2])
    image1 = cv2.drawContours(blank.copy(),contours,0,1)
    image2 = cv2.drawContours(blank.copy(),contours,1,1)
    intersection = np.logical_and(image1,image2)
    in_inter = cv2.pointPolygonTest(contour1,cent,1)
    if in_inter > 0 or intersection.any():is_ok = True
    return is_ok

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    h,w,_ = full_img.shape
    img = cv2.resize(full_img,(512,512))
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # contiguous
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    tensor_img = img
    if tensor_img.ndimension() == 3: tensor_img = tensor_img.unsqueeze(0)

    with torch.no_grad():
        output = net(tensor_img)[0]
        pr = F.softmax(output.permute(1,2,0),dim = -1).cpu().numpy()
        pr = pr.argmax(axis=-1)
        full_mask = ndcolormap[pr].astype(np.uint8)
        full_mask = cv2.resize(full_mask,(w,h))
        draw_mask = copy.deepcopy(full_mask)
        full_mask = cv2.cvtColor(full_mask,cv2.COLOR_BGR2GRAY)
        contours = cv2.findContours(full_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


        contours = contours[1] if len(contours)==3 else contours[0]
        contours_list = []
        for contour in contours:
            contours_list.append(contour)
            cv2.drawContours(draw_mask,[contour],0,(0,255,0),2)

        # tmp_contours = np.array([(546,440),(571,440),(571,466),(546,466)])
        # tmp_contours = np.expand_dims(tmp_contours,axis=1)
        # cv2.drawContours(draw_mask, [tmp_contours], 0, (0, 255, 0), 2)
        # intersection = contourIntersect(draw_mask,contours_list[0],tmp_contours)
        # print(intersection)

        # cv2.imshow('test',draw_mask)
        # cv2.waitKey()
        return contours_list,draw_mask

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='weights/checkpoint_epoch110.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='/media/root/fc6cf683-9c07-46ff-a325-9115b0701844/root/LIKE/夜间轨道/video_test/img',metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', default='output',metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))



if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nn.DataParallel(net)
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))


    for i, filename in enumerate(os.listdir(in_files)):
        img_path = os.path.join(in_files,filename)
        save_path = os.path.join(args.output,filename)
        img = Image.open(img_path)
        img = img.resize((512, 512), resample=Image.BICUBIC)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        cv2.imwrite(save_path,mask)
