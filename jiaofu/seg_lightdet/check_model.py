import os
import numpy as np
import yaml
import torch
from models.yolo import Model
from torchsummary import summary


nc = 3
cfg = r"./models/lightModels/yolov5lEfficientLite.yaml"
hyp = r"./data/hyps/hyp.scratch-low.yaml"

with open(hyp) as ff:
    hyp = yaml.safe_load(ff)
model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors'))

device = torch.device('cuda')
model.to(device)
input_ = torch.randn(1,3,416,416).to(device)
output = model(input_)

summary(model, (3, 416, 416), batch_size=-1)



# # 定义总参数量、可训练参数量及非可训练参数量变量
# Total_params = 0
# Trainable_params = 0
# NonTrainable_params = 0

# # 遍历model.parameters()返回的全局参数列表
# for param in model.parameters():
#     mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
#     Total_params += mulValue  # 总参数量
#     if param.requires_grad:
#         Trainable_params += mulValue  # 可训练参数量
#     else:
#         NonTrainable_params += mulValue  # 非可训练参数量

# print(f'Total params: {Total_params}')
# print(f'Trainable params: {Trainable_params}')
# print(f'Non-trainable params: {NonTrainable_params}')




