import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

lena = mpimg.imread('D:\DataSet\label\IMG_8606_json\label.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
#(512, 512, 3)
lena = lena[:,:,0]
# print(lena[100,100,3])

plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()