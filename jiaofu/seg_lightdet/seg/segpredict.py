import torch
from torch.utils.data import DataLoader
from torch import autograd, optim
# from model.unetpp import NestedUNet
from seg.dataset import *
from torchvision.transforms import transforms
import numpy as np
import cv2

x_transforms = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])

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


def getDataset(img,x_transforms):
    test_dataset = LiverDataset(img,transform=x_transforms)
    test_dataloaders = DataLoader(test_dataset)
    return test_dataloaders


def postprocess(predict,img):
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
    predict = predict.astype(np.uint8)
    predict1 = cv2.resize(predict, (raw_weight,raw_height), interpolation=cv2.INTER_NEAREST)
    predict2 = predict1 * 255
    cv2.imwrite('1.png', predict2)  # for test
    contours, hierarchy = cv2.findContours(predict2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valuelist = []
    for i in range(len(contours)):
        num = contours[i].shape[0]
        valuelist.append(num)
    if not valuelist: return
    value =  max(valuelist)
    id = valuelist.index(value)
    contours = contours[id]
    # numlist = []
    # for i in range(len(contours)):
    #     if contours[i].shape[0] > 0.5 * value:
    #         numlist.append(contours[i])

    # contours = contours[1] if len(contours) == 3 else contours[0]
    contours_list = []
    for contour in contours:
        contours_list.append(contour)
        cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)

    cv2.imwrite('3.jpg',img)

    return contours_list,img

def test(model,val_dataloaders,device):
    with torch.no_grad():
        for pic in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            result = torch.squeeze(predict[-1]).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
    return result

def predict(model,img,device):
    test_dataloaders = getDataset(img,x_transforms)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    result = test(model,test_dataloaders,device)

    contours_list,img = postprocess(result,img)
    return contours_list,img

# imgpath = 'test/1.jpg'
# predict = predict(imgpath)
# cv2.imwrite('1.png',predict*255)

