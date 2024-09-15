import numpy as np
import os
from PIL import Image
import skimage.io
import skimage.transform
import xlwt


# import pydensecrf.densecrf as dcrf


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


################################################################
######################## Evaluation ############################
################################################################
def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(sheetname="sheet1", cell_overwrite_ok=True)

    j = 0
    for data in datas:
        for i in range(len(data)):
            sheet1.write(i, j, data[i])
        j = j + 1

    f.save(file_path)


# def get_gt_mask(imgname, MASK_DIR):
#     # filestr = imgname.split(".")[0]
#     filestr = imgname[:-4]
#     mask_folder = MASK_DIR
#     mask_path = mask_folder + "/" + filestr + ".png"
#     mask = skimage.io.imread(mask_path)
#     mask = np.where(mask == 255, 1, 0).astype(np.float32)
#
#     return mask


def get_gt_mask(imgname, MASK_DIR):
    # 获取文件名（不带扩展名）
    filestr = imgname[:-4]
    mask_path = os.path.join(MASK_DIR, filestr + ".png")

    # 使用 PIL 读取图像并处理调色板问题
    pil_image = Image.open(mask_path)

    # 如果图像是调色板模式 (P mode)，将其转换为灰度图像
    if pil_image.mode == "P":
        pil_image = pil_image.convert("L")

    # 将图像转换为 NumPy 数组
    mask = np.array(pil_image)

    # 将像素值为 255 的区域设为 1，其他区域设为 0，并转换为 float32 类型
    mask = np.where(mask == 255, 1, 0).astype(np.float32)

    return mask


def get_normalized_predict_mask(imgname, PREDICT_MASK_DIR):
    """Get mask by specified single image name"""
    # filestr = imgname.split(".")[0]
    filestr = imgname[:-4]
    mask_folder = PREDICT_MASK_DIR
    mask_path = mask_folder + "/" + filestr + ".png"
    # mask_path = mask_folder + "/" + filestr + ".jpg"
    if not os.path.exists(mask_path):
        print("{} has no label8.png")
    mask = skimage.io.imread(mask_path)
    mask = mask.astype(np.float32)
    if np.max(mask) - np.min(mask) == 0:
        mask = mask / 256
    elif np.max(mask) > 0:
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    mask = mask.astype(np.float32)

    return mask


def get_binary_predict_mask(imgname, PREDICT_MASK_DIR):
    """Get mask by specified single image name"""
    # filestr = imgname.split(".")[0]
    filestr = imgname[:-4]
    mask_folder = PREDICT_MASK_DIR
    mask_path = mask_folder + "/" + filestr + ".png"
    # mask_path = mask_folder + "/" + filestr + ".jpg"
    if not os.path.exists(mask_path):
        print("{} has no label8.png")
    mask = skimage.io.imread(mask_path)
    mask = np.where(mask >= 127.5, 1, 0).astype(np.float32)

    return mask


def accuracy_mirror(predict_mask, gt_mask):
    """
    sum_i(n_ii) / sum_i(t_i)
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    if N_p == 0:
        print('warning NP=0...')
        return 0
    N_n = np.sum(np.logical_not(gt_mask))
    # if N_p + N_n != 640 * 512:
    #     raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    # print(TP)
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))
    # print(TN)

    accuracy_ = TP / N_p

    return accuracy_


def accuracy_image(predict_mask, gt_mask):
    """
    sum_i(n_ii) / sum_i(t_i)
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    # if N_p + N_n != 640 * 512:
    #     raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    accuracy_ = (TP + TN) / (N_p + N_n)

    return accuracy_


def compute_iou(predict_mask, gt_mask):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    Here, n_cl = 1 as we have only one class (mirror).
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    if np.sum(predict_mask) == 0 or np.sum(gt_mask) == 0:
        return 0

    n_ii = np.sum(np.logical_and(predict_mask, gt_mask))
    t_i = np.sum(gt_mask)
    n_ij = np.sum(predict_mask)

    iou_ = n_ii / (t_i + n_ij - n_ii)

    return iou_


def cal_precision_recall(predict_mask, gt_mask):
    # input should be np array with data type uint8
    assert predict_mask.dtype == np.uint8
    assert gt_mask.dtype == np.uint8
    assert predict_mask.shape == gt_mask.shape

    if np.sum(predict_mask) == 0:
        return 0, 0

    eps = 1e-4

    prediction = predict_mask / 255.
    gt = gt_mask / 255.

    # mae = np.mean(np.abs(prediction - gt))

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt).astype(np.float32)

    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt).astype(np.float32)
        p = np.sum(hard_prediction).astype(np.float32)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall


def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure


def compute_mae(predict_mask, gt_mask):
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    # if N_p + N_n != 640 * 512:
    #     raise Exception("Check if mask shape is correct!")

    mae_ = np.mean(abs(predict_mask - gt_mask)).item()

    return mae_


def compute_ber(predict_mask, gt_mask):
    """
    BER: balance error rate.
    :param predict_mask:
    :param gt_mask:
    :return:
    """
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    if N_p == 0:
        print('warning NP=0....')
        return 0
    N_n = np.sum(np.logical_not(gt_mask))
    # if N_p + N_n != 640 * 512:
    #     raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    ber_ = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

    return ber_


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
