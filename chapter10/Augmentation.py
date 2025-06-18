import PIL.Image as Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import warnings
import numpy as np
import random

def imshow(img_path, transform):
    """
    Function to show data augmentation
    Param img_path: path of the image
    Param transform: data augmentation technique to apply
    """
    img = Image.open(img_path)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].set_title(f'Original image {img.size}')
    ax[0].imshow(img)
    img = transform(img)
    ax[1].set_title(f'Transformed image {img.size}')
    ax[1].imshow(img)
    plt.show()


def cutout(image, mask_size=30, num_masks=5):
    image_np = np.array(image).copy()
    h, w = image_np.shape[0], image_np.shape[1]

    for _ in range(num_masks):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        y1 = max(0, y - mask_size // 2)
        y2 = min(h, y + mask_size // 2)
        x1 = max(0, x - mask_size // 2)
        x2 = min(w, x + mask_size // 2)

        # 遮挡区域设置为黑色
        image_np[y1:y2, x1:x2, :] = 0

    # 转回 PIL 图像
    return Image.fromarray(image_np)

path = r"E:\电子书\RethinkFun深度学习\data\PetImages\Cat\6039.jpg"
#transform = transforms.RandomRotation(degrees=30)

#transform = transforms.RandomHorizontalFlip(p=1.0)  # p=1.0 表示总是翻转，p是翻转的概率值。

#transform = transforms.RandomCrop(size=(120, 120))

# transform = transforms.RandomPerspective(
#     distortion_scale=0.5,  # 控制变形强度，0~1，越大越扭曲
#     p=1.0,                 # 应用该变换的概率
#     interpolation=transforms.InterpolationMode.BILINEAR
# )

# transform = transforms.ColorJitter(
#     brightness=0.5,
#     contrast=0.5,
#     saturation=0.5,
#     hue=0.1
# )


#transform = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)) # 对图像进行高斯模糊，kernel size 为 5，sigma 可调节模糊强度


imshow(path, cutout)