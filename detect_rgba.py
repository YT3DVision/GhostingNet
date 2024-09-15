from PIL import Image
import os


def is_image_rgba(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                return True
            else: return False
    except IOError:
        return False



if __name__ == '__main__':
    img_paths = os.listdir(r"E:\PycharmProjects\GEGD2\train\glass")
    count = 0
    for img_path in img_paths:
        count+=1
        img_path = os.path.join(r"E:\PycharmProjects\GEGD2\train\glass", img_path)
        if is_image_rgba(img_path) is True:
            print(img_path)
    print(count)


    # img = Image.open(r"E:\PycharmProjects\GEGD2\train\glass\DSCF1475.png").convert("L")
    # img.save("E:\PycharmProjects\GhosetNetV3\dection_rgba2rgb\DSCF1475.png")