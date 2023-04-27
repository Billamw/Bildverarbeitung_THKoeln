import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def Binarization(image, threshhold):
    imgBin = np.zeros((image.shape[0], image.shape[1]))
    for y in range(height):
        for x in range(width):
            if 0.2125 * image[y][x][0] + 0.7154 * image[y][x][1] + 0.072 * image[y][x][2] < threshhold:
                imgBin[y][x] = 0
            else:
                imgBin[y][x] = 255
    return imgBin

imgCol = cv2.imread('../../Utils/LenaJPEG.jpg')
height, width, depth = imgCol.shape

# Anzeigen des Bildes in einem Fenster
cv2.imshow('Mein Bild', Binarization(imgCol, 100))

# Warten auf eine Taste, um das Fenster zu schließen
cv2.waitKey(0)

# Schließen des Fensters
cv2.destroyAllWindows()
