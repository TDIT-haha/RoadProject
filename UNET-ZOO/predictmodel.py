from UNet import Unet,resnet34_unet
from attention_unet import AttU_Net
from channel_unet import myChannelUnet
from unet.unet_model import UNet as unet_cbam
from unetpp_cbam import NestedUNet as NestedUNet_Cbam
from r2unet import R2U_Net
from segnet import SegNet
from unetpp import NestedUNet
from fcn import get_fcn8s

import torch
from torchvision.transforms import transforms
from torchsummary import summary
import argparse
import time
import cv2
import numpy as np


parse = argparse.ArgumentParser()
parse.add_argument('--deepsupervision', default=0)
args = parse.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# UNet
# model = Unet(3, 1).to(device)
# modelpath = r"/root/project/Modules/TrackAnomalyTask/UNET-ZOO/saved_model/UNet_8_roadseg_200.pth"
# model.load_state_dict(torch.load(modelpath, map_location='cuda'))  # 载入训练好的模型
# model.eval()

# # Unet++
args.deepsupervision = True
model = NestedUNet(args,3,1).to(device)
modelpath = r"/root/project/Modules/TrackAnomalyTask/UNET-ZOO/saved_model/unet++_4_roadseg_200.pth"
model.load_state_dict(torch.load(modelpath, map_location='cuda'))  # 载入训练好的模型
model.eval()

# unet+SE+CAM
# model = unet_cbam(3, 1).to(device)
# modelpath = r"/root/project/Modules/TrackAnomalyTask/UNET-ZOO/saved_model/UNet_Cbam_8_roadseg_200.pth"
# model.load_state_dict(torch.load(modelpath, map_location='cuda'))  # 载入训练好的模型

#  Unet++ +CBAM
# args.deepsupervision = False
# model = NestedUNet_Cbam(3, 1).to(device)
# modelpath = r""
# model.load_state_dict(torch.load(modelpath, map_location='cuda'))  # 载入训练好的模型


imagepath = r"/root/project/Datas/otherDatas/dataset/images/11_319.jpg"
maskpath = r"/root/project/Datas/otherDatas/dataset/mask/11_319_mask.png"
image = cv2.imread(imagepath)
h_, w_ = image.shape[:2]
print(image.shape)
# 数据预处理
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])
imgx,imgy=(576,576)
pic = cv2.resize(image,(imgx,imgy))
pic = pic.astype('float32') / 255
pic = x_transforms(pic)
pic = pic.unsqueeze(0)
print(pic.shape)


# 模型推理
predict = model(pic.to(device))
if args.deepsupervision:
    predict = predict[-1].detach().cpu().numpy()[0]
else:
    predict = predict.detach().cpu().numpy()[0]
predict = predict.transpose(1,2,0)
predict = np.repeat(predict, 3, axis=2)
predict[predict>=0.5] = 255
predict[predict<0.5] = 0
predict = predict.astype("uint8")
print(predict.shape)
predict[:,:,0][predict[:,:,0]==255] = 0
predict[:,:,1][predict[:,:,1]==255] = 255
predict[:,:,2][predict[:,:,2]==255] = 0
predict = cv2.resize(predict,(w_,h_))


# # 添加透明显示
alpha = 0.4
beta = 0.8
pt_image = cv2.addWeighted(predict, alpha, image , beta, 0)
cv2.imwrite("mask_pt.jpg", pt_image)

# gt的mask
gt_mask = cv2.imread(maskpath)
print(gt_mask.shape)
print(np.unique(gt_mask))
gt_mask[:,:,0][gt_mask[:,:,0]==255] = 0
gt_mask[:,:,1][gt_mask[:,:,1]==255] = 0
gt_mask[:,:,2][gt_mask[:,:,2]==255] = 255

# # 添加透明显示
alpha = 0.4
beta = 0.8
gt_image = cv2.addWeighted(gt_mask, alpha, image , beta, 0)
cv2.imwrite("mask_gt.jpg", gt_image)

