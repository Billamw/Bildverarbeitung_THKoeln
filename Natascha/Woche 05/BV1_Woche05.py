import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

imgCol = cv2.imread('../../Utils/Beispiel_Tilman.jpg')
height, width, depth = imgCol.shape

def getGrayImage(image):
    height, width, depth = image.shape
    grayImg = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            grayImg[y][x] = 0.2125 * image[y][x][0] + 0.7154 * image[y][x][1] + 0.072 * image[y][x][2]

    return grayImg

# Anzeigen des Bildes in einem Fenster
cv2.imshow('Mein Bild', imgCol)

# Warten auf eine Taste, um das Fenster zu schließen
cv2.waitKey(0)

# Schließen des Fensters
cv2.destroyAllWindows()

# Anzeigen des Bildes in einem Fenster
cv2.imshow('Mein Bild', getGrayImage(imgCol))

# Warten auf eine Taste, um das Fenster zu schließen
cv2.waitKey(0)

# Schließen des Fensters
cv2.destroyAllWindows()