import os
import os.path
import torch.utils.data as data
from PIL import Image
import numpy as np


def make_train_data(data_path):
    print('INFO: Processing Train Data')
    data_path = os.path.join(data_path, 'train')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_path, 'image')) if f.endswith('.jpg')]
    return [
        (os.path.join(data_path, 'image', img_name + '.jpg'),
         os.path.join(data_path, 'glass', img_name + '.png'),
         os.path.join(data_path, 'ghost', img_name + '.png'),
         os.path.join(data_path, 'r1', img_name + '.png'),
         os.path.join(data_path, 'r2', img_name + '.png'),
         os.path.join(data_path, 'h_mask', img_name + '.npy'),
         os.path.join(data_path, 'w_mask', img_name + '.npy')
         )
        for img_name in img_list]


def make_test_data(data_path):
    print('INFO: Processing Test Data')
    data_path = os.path.join(data_path, 'test')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_path, 'image')) if f.endswith('.jpg')]
    return [
        (os.path.join(data_path, 'image', img_name + '.jpg'),
         os.path.join(data_path, 'glass', img_name + '.png'),
         os.path.join(data_path, 'ghost', img_name + '.png'),
         os.path.join(data_path, 'r1', img_name + '.png'),
         os.path.join(data_path, 'r2', img_name + '.png'),
         os.path.join(data_path, 'h_mask', img_name + '.npy'),
         os.path.join(data_path, 'w_mask', img_name + '.npy')
         )
        for img_name in img_list]




class make_dataSet(data.Dataset):
    def __init__(self, data_path, train=True, rgb_transform=None, grey_transform=None, npy_transform=None):
        self.train = train
        self.data_path = data_path
        self.rgb_transform = rgb_transform
        self.grey_transform = grey_transform
        self.npy_transform = npy_transform

        if self.train:
            self.images = make_train_data(data_path)
        else:
            self.images = make_test_data(data_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.train:
            image_path, glass_path, ghost_path, r1_path, r2_path, h_path, w_path = self.images[index]
        else:
            image_path, glass_path, ghost_path, r1_path, r2_path, h_path, w_path = self.images[index]

        # image_path, ghost_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        glass = Image.open(glass_path)
        ghost = Image.open(ghost_path)
        r1 = Image.open(r1_path)
        r2 = Image.open(r2_path)
        h = np.load(h_path)
        h = Image.fromarray(h)
        w = np.load(w_path)
        w = Image.fromarray(w)

        if self.rgb_transform is not None:
            image = self.rgb_transform(image)

        if self.grey_transform is not None:
            glass = self.grey_transform(glass)
            ghost = self.grey_transform(ghost)
            r1 = self.grey_transform(r1)
            r2 = self.grey_transform(r2)
            h = self.grey_transform(h)
            w = self.grey_transform(w)

        return image, glass, ghost, r1, r2, h, w
        # return image, ghosts
