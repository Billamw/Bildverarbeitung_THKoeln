import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

print("start")

Path_MADN_01 = 'Utils/MenschAergere01.jpg'
Path_MADN_02 = 'Utils/MenschAergere02.jpg'
Path_MADN_03 = 'Utils/MenschAergere03.jpg'

MADN_01 = cv2.imread(Path_MADN_01)
MADN_02 = cv2.imread(Path_MADN_02)
MADN_03 = cv2.imread(Path_MADN_03)

height, width, _ = MADN_01.shape

MADN_01_resize = cv2.resize(MADN_01, (int(width/5), int(height/5)))
MADN_02_resize = cv2.resize(MADN_02, (int(width/5), int(height/5)))
MADN_03_resize = cv2.resize(MADN_03, (int(width/5), int(height/5)))

hsv_img = cv2.cvtColor(MADN_01_resize, cv2.COLOR_BGR2HSV)

hsv_onlyRed = empty_array = np.zeros((height, width))
hsv_onlyBlue = empty_array = np.zeros((height, width))
hsv_onlyGreen = empty_array = np.zeros((height, width))

def seperateHSV(image, mode):
    hsv_seperated= np.zeros((image.shape[0], image.shape[1]))
    if(mode == 1 or mode == 2): #green or blue
        if mode == 1 :
            min = 40
            max = 70
        else:
            min = 100
            max = 130
        
        for y in range(height):
            for x in range(width):
                if(min < image[y][x][0] and image[y][x][0] < max):
                    hsv_seperated[y][x][0] = image[y][x][0]
    
    return hsv_seperated[y][x][0]

cv2.imshow('HSV image', seperateHSV(hsv_img,1))
cv2.waitKey(0)
cv2.destroyAllWindows()

print("end")