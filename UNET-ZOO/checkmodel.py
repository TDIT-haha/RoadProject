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
from torchsummary import summary
import argparse
import time

parse = argparse.ArgumentParser()
parse.add_argument('--deepsupervision', default=0)
args = parse.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# UNet
# model = Unet(3, 1).to(device)

# # Unet++
# args.deepsupervision = True
# model = NestedUNet(args,3,1).to(device)

# unet+SE+CAM
# model = unet_cbam(3, 1).to(device)

#  Unet++ +CBAM
# args.deepsupervision = True
# model = NestedUNet_Cbam(3, 1).to(device)


# 模型参数
summary(model, input_size=(3, 576, 576))

# # 模型推理速度
# for i in range(10):
#     input_ = torch.rand(1, 3, 576, 576).to(device)
#     model(input_)
    
# nums = 100
# totals = 0
# for i in range(nums):
#     input_ = torch.rand(1, 3, 576, 576).to(device)
#     time0 = time.time()
#     model(input_)
#     time1 = time.time()
#     totals += (time1-time0)

# meantime = totals/nums
# print("mean time :{:.4f}".format(meantime))
# print("FPS :{:.4f}".format(1/meantime))





