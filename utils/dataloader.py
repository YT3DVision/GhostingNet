import os
import os.path
import torch.utils.data as data
import numpy as np
import torch

from PIL import Image


def make_train_data(data_path):
    print('INFO: Processing Train Data')
    data_path = os.path.join(data_path, 'train')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_path, 'r1')) if f.endswith('.png')]
    return [
        (os.path.join(data_path, 'r1', img_name + '.png'),
         os.path.join(data_path, 'r2', img_name + '.png'),
         os.path.join(data_path, 'h_mask', img_name + '.npy'),
         os.path.join(data_path, 'w_mask', img_name + '.npy')
         )
        for img_name in img_list]

def make_test_data(data_path):
    print('INFO: Processing Test Data')
    data_path = os.path.join(data_path, 'test')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_path, 'r1')) if f.endswith('.png')]
    return [
        (os.path.join(data_path, 'r1', img_name + '.png'),
         os.path.join(data_path, 'r2', img_name + '.png'),
         os.path.join(data_path, 'h_mask', img_name + '.npy'),
         os.path.join(data_path, 'w_mask', img_name + '.npy')
         )
        for img_name in img_list]


class make_dataSet(data.Dataset):
    def __init__(self, data_path, train=True, grey_transform=None):
        self.train = train
        self.data_path = data_path
        self.grey_transform = grey_transform

        if self.train:
            self.images = make_train_data(data_path)
        else:
            self.images = make_test_data(data_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        r1_path, r2_path, h_path, w_path = self.images[index]
        r1 = Image.open(r1_path).convert('L')
        r2 = Image.open(r2_path).convert('L')
        h = np.load(h_path)
        h = Image.fromarray(h).convert('L')
        w = np.load(w_path)
        w = Image.fromarray(w).convert('L')

        if self.grey_transform is not None:
            r1 = self.grey_transform(r1)
            r2 = self.grey_transform(r2)
            h = self.grey_transform(h)
            w = self.grey_transform(w)
            
        # h = h.unsqueeze(0)
        # w = w.unsqueeze(0)

        return r1, r2, h, w














