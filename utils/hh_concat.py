from PIL import Image
import os


def img_concat(img1_path, img2_path, name):
    # 打开图片
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    # 获取图片尺寸
    width1, height1 = img1.size
    width2, height2 = img2.size
    # 计算拼接后的图片尺寸
    new_width = width1 + width2
    new_height = max(height1, height2)
    # 创建新的空白图片
    img3 = Image.new("RGB", (new_width, new_height))
    # 将 img1 和 img2 拼接到 img3 上
    img3.paste(img1, (0, 0))
    img3.paste(img2, (width1, 0))
    # 保存拼接后的图片
    name = name.split(".")[0] + ".png"
    path = os.path.join(r"E:\shift_concat", name)
    img3.save(path)
    # 显示拼接后的图片
    # img3.show()
    # print(123)


if __name__ == '__main__':
    imgs1 = os.listdir(r"F:\GEGD_shift\h_mask_pic")
    imgs2 = os.listdir(r"F:\GEGD_shift\w_mask_pic")

    imgs1_path = r"F:\GEGD_shift\h_mask_pic"
    imgs2_path = r"F:\GEGD_shift\w_mask_pic"
    for idx, img1_name in enumerate(imgs1):
        # img_name = os.path.splitext(img1_path)
        img2_path = os.path.join(imgs2_path, img1_name)
        img1_path = os.path.join(imgs1_path, img1_name)
        img_concat(img1_path, img2_path, img1_name)