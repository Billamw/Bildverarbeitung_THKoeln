import numpy as np
import cv2

def linearContrastStretch(q, amin, amax, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = np.zeros_like(gray_image)
    height, width = gray_image.shape

    cumhistogram = np.cumsum(np.histogram(gray_image))
    t0 = 0
    t1 = 255
    for a in range(len(cumhistogram)):
        if 255*q > cumhistogram[a]:
            t0 = a
        if (255 - 255*q) < cumhistogram[a]:
            t1 = a

    for x in range(width):
        for y in range(height):
            if gray_image[y][x] <= t0:
                output[y][x] = amin
            elif gray_image[y][x] < t1:
                output[y][x] = amin + (((amax - amin)/(t1-t0))*(gray_image[y][x] - t0))
            else:
                output[y][x] = amax

    return output

image = cv2.imread("Utils/Uebung 10/Rugby.png")
output = linearContrastStretch(0.1, 100, 200,image)

cv2.imshow('displaymywindows', output)
cv2.waitKey(0)   #wait for a keyboard input
cv2.destroyAllWindows()
