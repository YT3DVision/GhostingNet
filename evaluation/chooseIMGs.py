import cv2
import os
import shutil
import matplotlib.pyplot as plt

img_list2 = {
'IMG_7648.jpg', 'IMG_7896.jpg', 'IMG_7959.jpg', 'IMG_8007.jpg', 'IMG_8020.jpg',
'IMG_8023.jpg', 'IMG_8043.jpg', 'IMG_8745.jpg', 'IMG_8830.jpg', 'IMG_8868.jpg',
'IMG_8874.jpg', 'IMG_8911.jpg', 'IMG_8924.jpg', 'IMG_9144.jpg',

'IMG_8064.jpg', 'IMG_7684.jpg', 'IMG_7869.jpg', 'IMG_8846.jpg',

'IMG_8192.jpg', 'IMG_8830.jpg', 'IMG_8068.jpg', 'IMG_9320.jpg',
'IMG_7684.jpg', 'IMG_7802.jpg', 'IMG_7830.jpg', 'IMG_7869.jpg',
'IMG_8504.jpg', 'IMG_8745.jpg'
}

img_list = {'IMG_8246.jpg'}

img_path = 'F:\\LFM\\LFM_V2\\Test\\test_cv' # image
depth_path = 'F:\\LFM\\LFM_V2\\Test\\test_depth' # depth
# DCENet = 'E:\\Compare\\TCSVT2021_DCENet-main\\results\\DCENet\\mask'
MINet = 'F:\\Compare\\MINet\\results\\MINet\\mask' # MINet
BBSNet = 'F:\\Compare\\BBS-Net-master\\results\\BBSNet\\mask'
OBGNet = 'F:\\Compare\\OBGNet-main\code\\results\OBGNet\\mask'
MirrorNet = 'F:\\Compare\\MirrorNet-master\\results\\MirrorNet\\mask'
PMD = 'F:\\Compare\\PMDNet-master\\results\\PMDNet\\mask'
PDNet = 'F:\\ablation_code\\results\\pdnet_lfm\\pdnet_lfm'
Ours = 'F:\\ablation_code\\Ours\\ablation\\Ours-3f\\mask'
mask_path = 'F:\\LFM\\LFM_V2\\Test\\test_mask' # mask
save_path = 'F:\\Compare\\failure_case'

img_save_path = os.path.join(save_path, 'cv')
depth_save_path = os.path.join(save_path, 'depth')
minet_save_path = os.path.join(save_path, 'MINet')
bbs_save_path = os.path.join(save_path, 'BBSNet')
obgnet_save_path = os.path.join(save_path, 'OBGNet')
mirror_save_path = os.path.join(save_path, 'MirrorNet')
pmd_save_path = os.path.join(save_path, 'PMD')
pdnet_save_path = os.path.join(save_path, 'PDNet')
ours_save_path = os.path.join(save_path, 'Ours')
mask_save_path = os.path.join(save_path, 'mask')

if not os.path.isdir(img_save_path):
    os.makedirs(img_save_path)

if not os.path.isdir(depth_save_path):
    os.makedirs(depth_save_path)

if not os.path.isdir(minet_save_path):
    os.makedirs(minet_save_path)

if not os.path.isdir(bbs_save_path):
    os.makedirs(bbs_save_path)

if not os.path.isdir(obgnet_save_path):
    os.makedirs(obgnet_save_path)

if not os.path.isdir(mirror_save_path):
    os.makedirs(mirror_save_path)

if not os.path.isdir(pmd_save_path):
    os.makedirs(pmd_save_path)

if not os.path.isdir(pdnet_save_path):
    os.makedirs(pdnet_save_path)

if not os.path.isdir(ours_save_path):
    os.makedirs(ours_save_path)

if not os.path.isdir(mask_save_path):
    os.makedirs(mask_save_path)

img_num = len(os.listdir(img_path))  # compute num
print(img_num)

img_list = [f for f in os.listdir(img_path) if f in img_list]
img_list.sort()
print(img_list)
print(len(img_list))

for idx, img_name in enumerate(img_list):
    print(img_name)
    cvs_path = os.path.join(img_path, img_name)
    cvs_save_path = os.path.join(img_save_path, img_name)
    shutil.copyfile(cvs_path, cvs_save_path)

    _depth_path = os.path.join(depth_path, img_name[:-4] + '.png')
    _depth_save_path = os.path.join(depth_save_path, img_name[:-4] + '.png')
    shutil.copyfile(_depth_path, _depth_save_path)

    _MINet = os.path.join(MINet, img_name)
    _minet_save_path = os.path.join(minet_save_path, img_name[:-4] + '.png')
    shutil.copyfile(_MINet, _minet_save_path)

    _BBSNet = os.path.join(BBSNet, img_name)
    _bbs_save_path = os.path.join(bbs_save_path, img_name[:-4] + '.png')
    shutil.copyfile(_BBSNet, _bbs_save_path)

    _OBGNet = os.path.join(OBGNet, img_name[:-4] + '.png')
    _obgnet_save_path = os.path.join(obgnet_save_path, img_name[:-4] + '.png')
    shutil.copyfile(_OBGNet, _obgnet_save_path)

    _MirrorNet = os.path.join(MirrorNet, img_name)
    _mirror_save_path = os.path.join(mirror_save_path, img_name[:-4] + '.png')
    shutil.copyfile(_MirrorNet, _mirror_save_path)

    _PMD = os.path.join(PMD, img_name)
    _pmd_save_path = os.path.join(pmd_save_path, img_name[:-4] + '.png')
    shutil.copyfile(_PMD, _pmd_save_path)

    _PDNet = os.path.join(PDNet, img_name[:-4] + '.png')
    _pdnet_save_path = os.path.join(pdnet_save_path, img_name[:-4] + '.png')
    shutil.copyfile(_PDNet, _pdnet_save_path)

    _Ours = os.path.join(Ours, img_name[:-4] + '.png')
    _ours_save_path = os.path.join(ours_save_path, img_name[:-4] + '.png')
    shutil.copyfile(_Ours, _ours_save_path)

    _mask_path = os.path.join(mask_path, img_name[:-4] + '.png')
    _mask_save_path = os.path.join(mask_save_path, img_name[:-4] + '.png')
    shutil.copyfile(_mask_path, _mask_save_path)


