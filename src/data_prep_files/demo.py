import cv2

import os
path = "/"
dir_list = os.listdir("D:/LT/dataset/train/train/")
#print(dir_list)
for i in dir_list:
    img = cv2.imread("D:/LT/dataset/train/train/"+i)
    height, width, channels = img.shape
    print(height, width, channels)