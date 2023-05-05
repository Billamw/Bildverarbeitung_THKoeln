import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

imgCol = cv2.imread('../../Utils/LennaCol.png')
height, width, depth = imgCol.shape

def getGrayImage(image):
    height, width, depth = image.shape
    grayImg = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            grayImg[y][x] = 0.2125 * image[y][x][0] + 0.7154 * image[y][x][1] + 0.072 * image[y][x][2]

    cv2.imshow('Mein Bild', grayImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

getGrayImage(imgCol)