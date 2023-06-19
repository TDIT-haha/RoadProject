import torch.utils.data as data
import PIL.Image as Image


class LiverDataset(data.Dataset):
    def __init__(self, img, transform=None):
        self.pics = []
        self.pics.append(img)
        self.transform = transform

    def __getitem__(self, index):
        x_img = self.pics[0]
        origin_x = Image.fromarray(x_img).convert('RGB').resize((224, 224))
        # origin_x = Image.open(x_path).convert('RGB').resize((224,224))

        if self.transform is not None:
            img_x = self.transform(origin_x)
        return img_x

    def __len__(self):
        return len(self.pics)

