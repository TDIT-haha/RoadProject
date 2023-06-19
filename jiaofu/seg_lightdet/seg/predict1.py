import torch
from torch.utils.data import DataLoader
from torch import autograd, optim
from model.unetpp import NestedUNet
from dataset import *
from torchvision.transforms import transforms
import numpy as np
import cv2

def getDataset(imgpath,x_transforms):
    test_dataset = LiverDataset(imgpath,transform=x_transforms)
    test_dataloaders = DataLoader(test_dataset)
    return test_dataloaders


def postprocess(predict,imgpath):
    img = cv2.imread(imgpath,0)
    raw_height,raw_weight = img.shape[0],img.shape[1]
    height = predict.shape[0]
    weight = predict.shape[1]
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1
    predict = predict.astype(np.float32)
    cv2.imwrite('2.png',predict*255)
    predict1 = np.resize(predict,(raw_height,raw_weight))
    return predict1

def pretest(model,val_dataloaders,device):
    with torch.no_grad():
        for pic in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            result = torch.squeeze(predict[-1]).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
    return result

def predict(imgpath):
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = NestedUNet(3, 1).to(device)
    model.load_state_dict(torch.load('unet++_8_liver_15.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()
    test_dataloaders = getDataset(imgpath,x_transforms)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    result = pretest(model,test_dataloaders,device)
    result = postprocess(result,imgpath)
    return result

imgpath = 'seg/test/1.jpg'
predict = predict(imgpath)
cv2.imwrite('1.png',predict*255)
