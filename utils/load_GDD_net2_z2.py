import os
import os.path
import torch.utils.data as data
import numpy as np
from PIL import Image

def make_train_data(data_path):
    print('INFO: Processing Train Data')
    data_path = os.path.join(data_path, 'train')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_path, 'image')) if f.endswith('.jpg')]
    return [
        (os.path.join(data_path, 'image', img_name + '.jpg'),
         os.path.join(data_path, 'mask', img_name + '.png'),
         os.path.join(data_path, 'ghostnpy_raw', img_name + '.npy'),
         os.path.join(data_path, 'gedfusionnpy_new2', img_name + '.npy'),

         )
        for img_name in img_list]


def make_test_data(data_path):
    print('INFO: Processing Test Data')
    data_path = os.path.join(data_path, 'test')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_path, 'image')) if f.endswith('.jpg')]
    return [
        (os.path.join(data_path, 'image', img_name + '.jpg'),
         os.path.join(data_path, 'mask', img_name + '.png'),
         os.path.join(data_path, 'ghostnpy_raw', img_name + '.npy'),
         os.path.join(data_path, 'gedfusionnpy_new2', img_name + '.npy')
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
        if self.train:
            image_path, glass_path, ghost_path, ged_fusion_path = self.images[index]
        else:
            image_path, glass_path, ghost_path, ged_fusion_path = self.images[index]

        image = Image.open(image_path).convert('RGB')
        glass_mask = Image.open(glass_path).convert('L')
        ghost = Image.fromarray(np.load(ghost_path).squeeze(0))
        ged_fusion = np.load(ged_fusion_path)

        if self.rgb_transform is not None:
            image = self.rgb_transform(image)

        if self.grey_transform is not None:
            glass_mask = self.grey_transform(glass_mask)
            ghost = self.grey_transform(ghost)

        if self.train:
            return image, glass_mask, ghost, ged_fusion
        else:
            return image, glass_mask, ghost, ged_fusion


if __name__ == '__main__':
    ghost = np.load(r"E:\download\WeChat Files\wxid_qmpjv5wq2o6z22\FileStorage\File\2023-06\999.npy")
    ghost = ghost.squeeze(0)
    ghost = 65535*(ghost-0)/(ghost.max() - ghost.min())
    ghost = Image.fromarray(ghost)
    ghost.show()



