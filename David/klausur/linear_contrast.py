import numpy as np
import cv2

def linearContrastStretch(t0, t1, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = np.zeros_like(gray_image)
    height, width = gray_image.shape

    for x in range(width):
        for y in range(height):
            if gray_image[y][x] <= t0:
                output[y][x] = 0
            elif gray_image[y][x] < t1:
                output[y][x] = (255/(t1-t0))*(gray_image[y][x] - t0)
            else:
                output[y][x] = 255

    return output

image = cv2.imread("Utils/Uebung 10/Rugby.png")
output = linearContrastStretch(-100,300 ,image)

cv2.imshow('displaymywindows', output)
cv2.waitKey(0)   #wait for a keyboard input
cv2.destroyAllWindows()
