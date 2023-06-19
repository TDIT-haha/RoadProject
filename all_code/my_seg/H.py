import cv2
import os

path = r'train_data/masks'
for img_name in os.listdir(path):
    img_path = os.path.join(path,img_name)
    img = cv2.imread(img_path,0)
    print(img[img>0])
    # img[img==255] = 1
    # cv2.imwrite(img_path,img)