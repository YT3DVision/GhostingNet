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
         )
        for img_name in img_list]


class make_dataSet(data.Dataset):
    def __init__(self, data_path, train=True, rgb_transform=None, grey_transform=None):
        self.train = train
        self.data_path = data_path
        self.rgb_transform = rgb_transform
        self.grey_transform = grey_transform

        if self.train:
            self.images = make_train_data(data_path)
        else:
            self.images = make_test_data(data_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        print(index)
        print(len(self.images[index]))
        print('-----')
        if self.train:
            image_path, glass_path, ghost_path, r1_path, r2_path = self.images[index]
            r1 = Image.open(r1_path).convert('L')
            r2 = Image.open(r2_path).convert('L')
        else:
            image_path, glass_path, ghost_path = self.images[index]

        # image_path, ghost_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        glass = Image.open(glass_path).convert('L')
        ghost = Image.open(ghost_path).convert('L')

        if self.rgb_transform is not None:
            image = self.rgb_transform(image)

        if self.grey_transform is not None:
            glass = self.grey_transform(glass)
            ghost = self.grey_transform(ghost)
            if self.train:
                r1 = self.grey_transform(r1)
                r2 = self.grey_transform(r2)
        if self.train:
            return image, glass, ghost, r1, r2
        else:
            return image, glass, ghost
        # return image, ghosts