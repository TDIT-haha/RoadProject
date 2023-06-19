import os
import xml.etree.ElementTree as ET
import shutil

# Path
# 源数据集的位置
ann_filepath = 'E:/projects/datasets/PASCAL2012/VOC2012/Annotations/'
img_filepath = 'E:/projects/datasets/PASCAL2012/VOC2012/JPEGImages/'
# 新建保存数据集的位置
img_savepath = 'E:/projects/datasets/PASCAL2012/VOC2012_1/JPEGImages/'
ann_savepath = 'E:/projects/datasets/PASCAL2012/VOC2012_1/Annotations/'

if not os.path.exists(img_savepath):
    os.mkdir(img_savepath)

if not os.path.exists(ann_savepath):
    os.mkdir(ann_savepath)

# VOC class information
# 需要的标签
classes = ['cat', 'cow', 'sheep','dog']  # The classes needed

# classes = ['sheep', 'sofa', 'train', 'person','tvmonitor']
def save_annotation(file):
    tree = ET.parse(ann_filepath + '/' + file)
    root = tree.getroot()
    result = root.findall("object")
    bool_num = 0
    for obj in result:
        if obj.find("name").text not in classes:
            root.remove(obj)
        else:
            bool_num = 1
    if bool_num:
        tree.write(ann_savepath + file)
        return True
    else:
        return False


def save_images(file):
    name_img = img_filepath + os.path.splitext(file)[0] + ".jpg"

    shutil.copy(name_img, img_savepath)
    # 图片名称txt保存的位置
    with open('list.txt', 'a') as file_txt:
        file_txt.write(os.path.splitext(file)[0])
        file_txt.write("\n")
    return True


if __name__ == '__main__':
    for f in os.listdir(ann_filepath):
        if save_annotation(f):
            save_images(f)




