from metrics import *
import os
import numpy as np

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
# gt_dir = 'D:/GhostDataSet/GEGD_eval_ghost'
# pred_dir = 'G:/GEGD_ghost_eval/wait'


# GEGD GHOST
# gt_dir = 'D:/GhostDataSet/GEGD_eval_ghost'
# pred_dir = 'G:/contrast_ghost2/wait'

gt_dir = r'E:\PycharmProjects\GhosetNetV3\GEGD\test'
pred_dir = r'E:\PycharmProjects\GhosetNetV3\Run'


# GDD
# gt_dir = 'G:/experiments/GDD/mask'
# pred_dir = 'G:/contrast/GDD_Results/'

# GSD
# gt_dir = 'G:/experiments/GSD/test/mask'
# pred_dir = 'G:/experiments/Glass_Results/eval'

# Response
# gt_dir = 'G:\LFM\LFM_V2\Train\\train_mask'
# pred_dir = 'G:\\code\\evaluation\\evaluation\\mask'

count = 0

for sub_dir in os.listdir(pred_dir):
    print(pred_dir)
    print(sub_dir)
    if 0 == len(os.listdir(os.path.join(pred_dir, sub_dir))):
        continue
    iou_l = []
    acc_l = []
    mae_l = []
    ber_l = []
    f_measure_p_l = []
    f_measure_r_l = []
    precision_record, recall_record = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
    for name in os.listdir(os.path.join(pred_dir, sub_dir)):
        print(name)
        gt = get_gt_mask(os.path.join(sub_dir, name), gt_dir)
        normalized_pred = get_normalized_predict_mask(name, os.path.join(pred_dir, sub_dir))
        binary_pred = get_binary_predict_mask(name, os.path.join(pred_dir, sub_dir))

        if normalized_pred.ndim == 3:
            normalized_pred = normalized_pred[:, :, 0]
        if binary_pred.ndim == 3:
            binary_pred = binary_pred[:, :, 0]

        acc_l.append(accuracy_mirror(binary_pred, gt))
        iou_l.append(compute_iou(binary_pred, gt))
        mae_l.append(compute_mae(normalized_pred, gt))
        count = count + compute_mae(normalized_pred, gt)
        print(count)
        ber_l.append(compute_ber(binary_pred, gt))


        pred = (255 * normalized_pred).astype(np.uint8)
        gt = (255 * gt).astype(np.uint8)
        p, r = cal_precision_recall(pred, gt)
        for idx, data in enumerate(zip(p, r)):
            p, r = data
            precision_record[idx].update(p)
            recall_record[idx].update(r)
    print('%s:  mae: %3f, ber: %3f, acc: %3f, iou: %3f, f_measure: %3f' %
          (sub_dir, np.mean(mae_l), np.mean(ber_l), np.mean(acc_l), np.mean(iou_l),
           cal_fmeasure([precord.avg for precord in precision_record], [rrecord.avg for rrecord in recall_record])))
