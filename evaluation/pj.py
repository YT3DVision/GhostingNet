import cv2
import os
import matplotlib.pyplot as plt

# img_path = 'E:\\0906\\95realdata'  # 图像路径
# re_path = 'E:\\0906\\New folder1\\GSD384_95\\glass'  # 图像路径
# re2_path = 'E:\\0906\\New folder1\\glass'
# re3_path = 'E:\\0906\\New folder1\\ST384_3\\glass'
# save_path = 'E:\\0906\\New folder1\\results'

# img_path = 'E:\\0916\\real\\test\\image'  # image
# re_path = 'E:\\0916\\real\\test\\glass'  # glass GT
# re2_path = 'E:\\0916\\real\\test\\ghost'  # ghost
# re3_path = 'E:\\0924\\0924\\Unet_E_real\\edge'  # edge
# re4_path = 'E:\\GlassResults_Ours\\GSD' # gsd
# re5_path = 'E:\\GlassResults_Ours\\gdnet' # gdd
# re6_path = 'E:\\GlassResults_Ours\\Unet_GSD_real126' # Unet
# re7_path = 'E:\\GlassResults_Ours\\UnetE' # Unet + E
# save_path = 'E:\\0924\\glass_results'

# img_path = 'E:\\GlassResults_Ours\\real_all\\image'  # image
# re_path = 'E:\\GlassResults_Ours\\real_all\\mask'  # glass GT
# re2_path = 'E:\\GlassResults_Ours\\real_all\\ghost'  # ghost
# re3_path = 'E:\\GlassResults_Ours\\1012\\Ours'  # our glass
# re4_path = 'E:\\GlassResults_Ours\\1012\\GSD' # gsd
# re5_path = 'E:\\GlassResults_Ours\\1012\\GDNet' # gdd
# # re6_path = 'E:\\GSD_GSD_Results\\1007\\Translab' # translab
# re7_path = 'E:\\GlassResults_Ours\\1012\\Trans2Seg' # tran2seg
# save_path = 'E:\\GlassResults_Ours\\1012\\pj'


img_path = 'F:\\LFM\\LFM_V2\\Test\\test_cv' # image
depth_path = 'F:\\LFM\\LFM_V2\\Test\\test_depth' # depth
# DCENet = 'E:\\Compare\\TCSVT2021_DCENet-main\\results\\DCENet\\mask'
MINet = 'F:\\Compare\\MINet\\results\\MINet\\mask' # MINet
HIDANet = 'F:\\Compare\\HIDANet-main\\test_maps\\HiDANet\\LFM_V2'
BBSNet = 'F:\\Compare\\BBS-Net-master\\results\\BBSNet\\mask'
OBGNet = 'F:\\Compare\\OBGNet-main\code\\results\OBGNet\\mask'
MirrorNet = 'F:\\Compare\\MirrorNet-master\\results\\MirrorNet\\mask'
PMD = 'F:\\Compare\\PMDNet-master\\results\\PMDNet\\mask'
PDNet = 'F:\\ablation_code\\results\\pdnet_lfm\\pdnet_lfm'
Ours = 'F:\\ablation_code\\Ours\\ablation\\Ours-3f\\mask'
mask_path = 'F:\\LFM\\LFM_V2\\Test\\test_mask' # mask
save_path = 'F:\\Compare\\Visual-Results-88'



if not os.path.isdir(save_path):
    os.makedirs(save_path)

img_num = len(os.listdir(img_path))  # compute num
print(img_num)

img_list = [os.path.splitext(f)[0] for f in os.listdir(img_path) if f.endswith('jpg')]
print(img_list)

count = 0
for img_name in img_list:
    print(count)
    count += 1
    img = cv2.imread(os.path.join(img_path, img_name + '.jpg'))
    depth = cv2.imread(os.path.join(depth_path, img_name + '.png'))
    dce = cv2.imread(os.path.join(MINet, img_name + '.jpg'))
    bbs = cv2.imread(os.path.join(BBSNet, img_name + '.jpg'))
    hidanet = cv2.imread(os.path.join(BBSNet, img_name + '.png'))
    obg = cv2.imread(os.path.join(OBGNet, img_name + '.png'))
    mn = cv2.imread(os.path.join(MirrorNet, img_name + '.jpg'))
    pmdnet = cv2.imread(os.path.join(PMD, img_name + '.jpg'))
    pd = cv2.imread(os.path.join(PDNet, img_name + '.png'))
    our = cv2.imread(os.path.join(Ours, img_name + '.png'))
    mask = cv2.imread(os.path.join(mask_path, img_name + '.png'))
    # H,W,C = img.shape

    b, g, r = cv2.split(img)
    img_merge = cv2.merge([r, g, b])

    plt.figure(facecolor="#F3F3F3")

    #
    plt.subplot(2, 5, 1)
    plt.imshow(img_merge)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.title('image', fontsize=8)

    #
    plt.subplot(2, 5, 2)
    plt.imshow(depth)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.title('depth', fontsize=8)

    #
    plt.subplot(2, 5, 3)
    plt.imshow(dce)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.title('MINet',fontsize=8)

    #
    plt.subplot(2, 5, 4)
    plt.imshow(hidanet)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.title('HIDANet',fontsize=8)

    #
    plt.subplot(2, 5, 5)
    plt.imshow(obg)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.title('OBGNet',fontsize=8)

    #
    plt.subplot(2, 5, 6)
    plt.imshow(mn)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.title('MirrorNet',fontsize=8)

    #
    plt.subplot(2, 5, 7)
    plt.imshow(pmdnet)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.title('PMDNet',fontsize=8)

    #
    plt.subplot(2, 5, 8)
    plt.imshow(pd)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.title('PDNet',fontsize=8)

    #
    plt.subplot(2, 5, 9)
    plt.imshow(our)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.title('Ours',fontsize=8)

    #
    plt.subplot(2, 5, 10)
    plt.imshow(mask)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.title('GT',fontsize=8)

    # plt.show()
    fig = plt.gcf()
    plt.rcParams['savefig.dpi'] = 240 # 图片像素
    plt.savefig(os.path.join(save_path, img_name + '.png'), bbox_inches='tight')
    plt.clf()
    plt.close()

