# 30.03.2023
import cv2 as cv2
import numpy as np

imgCol = cv2.imread('Utils\LenaJPEG.jpg')

width, height, depth = imgCol.shape



cv2.imshow("img", imgCol)
cv2.waitKey(0)
cv2.destroyAllWindows()