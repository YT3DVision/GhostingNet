import metrics
import cv2

path1 = 'D:/2022/revise/misc/IMG_8367_0.5.jpg'
path2 = 'D:/2022/revise/misc/prova_1_r_0.5.jpg'
gt = 'D:/2022/revise/misc/IMG_8367_mask.jpg'

a = cv2.imread(gt)
cv2.imshow('img', a)
b = cv2.imread(path2)
gt_img = cv2.imread(gt)

temp = metrics.accuracy_mirror(gt_img,a)
print(temp)