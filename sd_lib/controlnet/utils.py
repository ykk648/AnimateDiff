# -- coding: utf-8 --
# @Time : 2023/10/23
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
import torch
from cv2box import CVImage


def make_inpaint_condition(image_p, image_mask_p):
    image = CVImage(image_p).pillow()
    image_mask = CVImage(image_mask_p).pillow()
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose((0, 3, 1, 2))
    image = torch.from_numpy(image)
    return image
