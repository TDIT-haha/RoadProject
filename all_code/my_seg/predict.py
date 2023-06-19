import argparse
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


colormap = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]
ndcolormap = np.array(colormap)

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    n_classes = 6
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)[0]

        pr = F.softmax(output.permute(1,2,0),dim = -1).cpu().numpy()
        pr = pr.argmax(axis=-1)
        full_mask = ndcolormap[pr].astype(np.uint8)
        return full_mask

    #     if n_classes > 1:
    #         probs = F.softmax(output, dim=1)[0]
    #     else:
    #         probs = torch.sigmoid(output)[0]
    #
    #     tf = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((full_img.size[1], full_img.size[0])),
    #         transforms.ToTensor()
    #     ])
    #
    #     full_mask = tf(probs.cpu()).squeeze()
    #
    # if n_classes == 1:
    #     return (full_mask > out_threshold).numpy()
    # else:
    #     return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints/checkpoint_epoch480.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='/media/root/fc6cf683-9c07-46ff-a325-9115b0701844/root/LIKE/电路板定位/img',metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', default='output',metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
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
        img.resize((512, 512), resample=Image.BICUBIC)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        cv2.imwrite(save_path,mask)