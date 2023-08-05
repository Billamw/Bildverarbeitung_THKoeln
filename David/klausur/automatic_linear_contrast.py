import numpy as np
import cv2

def linearContrastStretch(q, amin, amax, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = np.zeros_like(gray_image)
    height, width = gray_image.shape

    histogram, bins = np.histogram(gray_image, bins=256, range=(0, 256))
    cummulative_histogram = np.cumsum(histogram)
    t0 = 0
    t1 = 255
    for a in range(len(cummulative_histogram)):
        if 255*q > cummulative_histogram[a]:
            t0 = a
        if (255 - 255*q) < cummulative_histogram[a]:
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
