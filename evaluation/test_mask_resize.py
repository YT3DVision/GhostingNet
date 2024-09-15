# import torch
# from torch.utils.data import Dataset
import numpy as np
import os
import os.path as osp
import cv2
root = r'F:\LF\LF_code\LFM\LFM_V2\Test'
lfdir = osp.join(root, 'test_mask')
new_lfdir = osp.join(root,'test_mask_res/')
if not os.path.isdir(new_lfdir):
    os.makedirs(new_lfdir)

img_list = [img_name for img_name in os.listdir(lfdir)]
w,h=256,256
for idx, img_name in enumerate(img_list):
    # print(img_name)
    print(idx)

    lf = cv2.imread(osp.join(lfdir, img_name))

    b, g, r = cv2.split(lf)  #三通道分别显示

    target_lf = cv2.resize(b, (w, h))
    print(target_lf.shape)
    newpath=new_lfdir+img_name
    print(newpath)
    cv2.imwrite(newpath, target_lf)

    # u, v = get_uv(target_lf, angular_size=9)
    # b, hu, wv, c = u.shape
    # u = np.concatenate(u, axis=0)
    # v = np.concatenate(v, axis=1)

    # cv2.imwrite(os.path.join(savedir_U, img_name), u)
    # cv2.imwrite(os.path.join(savedir_V, img_name), v)