import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
def bgr2grey(input_path, output_path):
    filelist = os.listdir(input_path)
    for file in filelist:
        raw_img = cv2.imread(input_path+file)
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2Lab)[:, :, 0]
        img = cv2.merge((img, img, img))
        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(output_path + file, img)

if __name__ == "__main__":
    bgr2grey('E:/work/Code/ColorAnything/examples/', 'E:/work/Code/ColorAnything/examples_grey/')