import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

print("start")

Path_MADN_01 = '../../Utils/MenschAergere01.jpg'
Path_MADN_02 = '../../Utils/MenschAergere02.jpg'
Path_MADN_03 = '../../Utils/MenschAergere03.jpg'

MADN_01 = cv2.imread(Path_MADN_01)
MADN_02 = cv2.imread(Path_MADN_02)
MADN_03 = cv2.imread(Path_MADN_03)

height, width, _ = MADN_01.shape

MADN_01_resize = cv2.resize(MADN_01, (int(width/5), int(height/5)))
MADN_02_resize = cv2.resize(MADN_02, (int(width/5), int(height/5)))
MADN_03_resize = cv2.resize(MADN_03, (int(width/5), int(height/5)))

hsv_img = cv2.cvtColor(MADN_01_resize, cv2.COLOR_BGR2HSV)

hsv_onlyRed = np.zeros((height, width,3))
hsv_onlyBlue = np.zeros((height, width,3))
hsv_onlyGreen = np.zeros((height, width,3))

def seperateHSV(image, mode):
    hsv_seperated = np.zeros_like(image)
    if mode == 1 or mode == 2 : #green or blue
        if mode == 1 :
            min = 40
            max = 75
        else:
            min = 110
            max = 130

    for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if min < image[y][x][0] < max:
                    hsv_seperated[y][x][:] = [255,255,255]

    return hsv_seperated

cv2.imshow('HSV image', seperateHSV(hsv_img,2))
#cv2.imshow('HSV image', hsv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("end")