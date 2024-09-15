from metrics import *
import os
import numpy as np
from tqdm import tqdm

# pred_dir = 'E:/GlassResults_GSD/'
# gt_dir = "E:/nf/GSD/test/mask"
# sub_dir = 'GSD'

# pred_dir = 'E:/GlassResults_Ours/'
# gt_dir = "E:/GlassResults_Ours/glass"
# sub_dir = 'UnetE'

# ghost
# pred_dir = 'E:/1230/3600_to_test'
# gt_dir = 'E:/DataSet/GEGD_syn/test/ghost'
# pred_dir= 'E:/1230/eval/20211111'


# pred_dir = 'E:/Ghost_Results/TEST3600'
# pred_dir = 'E:/1230/eval/datagjh21010_ghostGPGN_4Cresults1028TEST3600' 1028
# pred_dir = 'E:/1230/eval/datagjh21104GhostNetresults1104'
# pred_dir = 'E:/1230/glass_eval/3'
# gt_dir = 'E:/1230/glass_eval/gt/3'

# glass:  mae: 0.037795, ber: 0.043685, acc: 0.935928, iou: 0.912782, f_measure: 0.953901

# GEGD GLASS
# gt_dir = 'D:/GhostDataSet/GEGD_eval'
# pred_dir = 'F:/TPAMI/revision_results/net2_sem2gsd/glass_resize'

# GEGD GHOST
# gt_dir = 'D:/GhostDataSet/GEGD_eval_ghost'
# pred_dir = 'G:/contrast_ghost2/wait'

# GDD
# gt_dir = 'G:/experiments/GDD/mask'
# pred_dir = 'G:/contrast/GDD_Results/'

# GSD
# gt_dir = 'G:/experiments/GSD/test/mask'
# pred_dir = 'G:/experiments/Glass_Results/eval'

# Response
# gt_dir = 'F:/TPAMI/split/gt_ghost/r'
# pred_dir = 'F:/TPAMI/split/pred_ghost/r'

# gt_dir = 'F:\LF\LF_code\LFM\LFM_V2\Test\\test_mask'
# pred_dir='F:\LF\LF_code\code\MACNet\results\MirrorNet\mask'
# gt_dir = 'F:\\LF\\LF_code\\LFM\\LFM_V2\\Test\\test_mask_res'
# gt_dir = 'E:\\LFM\\LFM_V2\\Test\\test_mask'
# pred_dir='E:\\ablation_code\\MS\\ablation\\RGB+Macro2\\mask'
# pred_dir = 'F:\LF\LF_code\code\DLSD\Saliency maps'
# pred_dir = 'F:\\Compare\\OBGNet-main\\code\\results\\OBGNet\\mask'

# gt_dir = 'D:/GhostDataSet/GEGD_eval_ghost'
# pred_dir = 'G:/GEGD_ghost_eval/wait'
# pred_dir = 'F:/TPAMI/revision_results/GED_Ours_ghost_best/ghost_resize'

# gt_dir = 'F:/data/GDD/test/mask'
# pred_dir = 'F:/TPAMI/New folder1/1101/woGhost'
#
# gt_dir = 'F:/data/GSD/test/mask'
# pred_dir = 'F:/TPAMI/Ablation/New folder1/glass'
#
"""
请优先采取此测试方案
"""

item = "glass"
gt_dir = os.path.join(r"E:\PycharmProjects\GDD\test\mask")
pred_dir = os.path.join(r"E:\PycharmProjects\GhostNet2\results\GDD_hh", item)



iou_l = []
acc_l = []
mae_l = []
ber_l = []
f_measure_p_l = []
f_measure_r_l = []
precision_record, recall_record = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
# print(precision_record,recall_record)
pbar = tqdm(os.listdir(os.path.join(gt_dir)))
for name in (pbar):
    pbar.set_description(f"item: {item}")
    pbar.set_postfix(img_name=name)
    # print(name)
    gt = get_gt_mask(name, gt_dir)
    # print(gt.shape)
    normalized_pred = get_normalized_predict_mask(name, os.path.join(pred_dir))
    binary_pred = get_binary_predict_mask(name, os.path.join(pred_dir))

    # normalized_pred = get_normalized_predict_mask(name[:-4]+'.jpg', os.path.join(pred_dir))
    # binary_pred = get_binary_predict_mask(name[:-4]+'.jpg', os.path.join(pred_dir))

    if normalized_pred.ndim == 3:
        normalized_pred = normalized_pred[:, :, 0]
    if binary_pred.ndim == 3:
        binary_pred = binary_pred[:, :, 0]

    if binary_pred.shape == gt.shape:
        print(f"success: pred_shape {binary_pred.shape} == gt_shape {gt.shape}")

    try:
        check_size(binary_pred, gt)
    except EvalSegErr:
        print(f"error: pred_shape {binary_pred.shape} != gt_shape {gt.shape}")
        continue

    acc_l.append(accuracy_mirror(binary_pred, gt))
    iou_l.append(compute_iou(binary_pred, gt))
    mae_l.append(compute_mae(normalized_pred, gt))
    ber_l.append(compute_ber(binary_pred, gt))

    pred = (255 * normalized_pred).astype(np.uint8)
    gt = (255 * gt).astype(np.uint8)
    p, r = cal_precision_recall(pred, gt)

    if p != 0 and r != 0:
        for idx, data in enumerate(zip(p, r)):
            p, r = data
            precision_record[idx].update(p)
            recall_record[idx].update(r)
    else:
        for idx in range(256):
            precision_record[idx].update(0)
            recall_record[idx].update(0)

print('%s:  mae: %3f, ber: %3f, acc: %3f, iou: %3f, f_measure: %3f' % (pred_dir, np.mean(mae_l), np.mean(ber_l), np.mean(acc_l), np.mean(iou_l),
  cal_fmeasure([precord.avg for precord in precision_record], [rrecord.avg for rrecord in recall_record])))






