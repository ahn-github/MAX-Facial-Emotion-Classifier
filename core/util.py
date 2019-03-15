import cv2
import numpy as np

def img_resize(input_data):
    # ratio = 1
    img_h, img_w, _ = np.shape(input_data)
    if img_w > 1024:
        ratio = 1024/img_w
        input_data = cv2.resize(input_data, (int(ratio*img_w), int(ratio*img_h)))
    elif img_h > 1024:
        ratio = 1024/img_h
        input_data = cv2.resize(input_data, (int(ratio*img_w), int(ratio*img_h)))
    return input_data #, ratio
