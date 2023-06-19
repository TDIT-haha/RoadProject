import math
import numpy as np
import torch
from unet import UNet
import cv2

colormap = [[0, 0, 0], [0, 255, 0]]
ndcolormap = np.array(colormap)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r'weight/net.pth'
shape = (32,128)

def getDist_P2P(PointA,PointB):
    distance = math.pow((PointA[0]-PointB[0]),2) + math.pow((PointA[1] - PointB[1]),2)
    distance = math.sqrt(distance)
    return distance

def getDist_P2L(PointP,Pointa,Pointb):
    A = Pointa[1] - Pointb[1]
    B = Pointb[0] - Pointa[0]
    C = Pointa[0]*Pointb[1] - Pointa[1]*Pointb[0]
    distance = (A*PointP[0]+B*PointP[1]+C)/math.sqrt(A*A+B*B)
    return distance

def seg_demo(net,image,danger):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,shape)
    image = image.astype(np.float32)
    image = image.transpose((2,0,1))
    image = torch.from_numpy(image)/255.
    image = image.to(device, dtype=torch.float32)
    image = image.unsqueeze(0)


    net.eval()
    with torch.no_grad():
        output = net(image)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        full_mask = probs.squeeze().cpu().numpy()
        full_mask[full_mask>0.5] = 255
        mask = full_mask.astype(np.uint8)

    all_coord = []
    new_mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

    contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if contours != []:
        idx, maxLencontours = max(enumerate(contours),key=lambda x:len(x[1]))
        back_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype='uint8')
        cv2.fillPoly(back_mask, [maxLencontours], (255))

        y_indexs,x_indexs = np.where(back_mask==255)[0],np.where(back_mask==255)[1]
        sampl_interval = [30,40,50,60,70,80,90,100]

        for inter in sampl_interval:
            if len(np.where(y_indexs==inter)[0]) == 0:continue
            tmp_index = np.where(y_indexs==inter)[0][0]
            tmp_coord = (x_indexs[tmp_index],y_indexs[tmp_index])
            all_coord.append(tmp_coord)
            cv2.circle(new_mask, tmp_coord, 2, (0,0,255), -1)

        for coord in all_coord:
            dis = math.fabs(getDist_P2L(coord,all_coord[0],all_coord[-1]))
            if dis > 3.:danger = True

    return all_coord,danger,new_mask