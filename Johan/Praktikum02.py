# 30.03.2023
import cv2 as cv2
import numpy as np
import matplotlib as plt

imgCol = cv2.imread('Utils\LenaJPEG.jpg')

width, height, depth = imgCol.shape

def getHistogramm(img):
    histogram=[0]*256
    for y in range(height):
        for x in range(width):
            histogram[img[y][x]] += 1
    return histogram


plt.plot(getHistogramm(imgCol))
cv2.imshow("img", imgCol)
cv2.waitKey(0)
cv2.destroyAllWindows()