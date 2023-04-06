# 30.03.2023
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

imgCol = cv2.imread('Utils\LenaJPEG.jpg')

width, height, depth = imgCol.shape

def getHistogramm(img=[]):
    histogram=np.zeros(256)
    
    for y in range(height):
        for x in range(width):
            if img.ndim == 3:
                i = 0.2125 * img[y][x][0] + 0.7154 * img[y][x][1] + 0.072 * img[y][x][2]
            else:
                i = img[y][x]

            histogram[int(i)] += 1

    return histogram

def getNormHistogramm(img=[]):
    return getHistogramm(img) / (width*height)

def getCumulatedHistogramm(img):
    H = np.zeros(256)
    h = getNormHistogramm(img)
    for i in range(len(h)):
        for j in range(i): 
            H[i] += h[j]
    return H

def linearContrast(img, t0, t1):
    toRet = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            a = 0.2125 * img[y][x][0] + 0.7154 * img[y][x][1] + 0.072 * img[y][x][2]
            # a = img[y][x][0]
            if a < t0:
                toRet[y][x][0] = 0
                toRet[y][x][1] = 0
                toRet[y][x][2] = 0
            if t0 <= a < t1:
                toRet[y][x][0] = 255/(t1-t0) * (a-t0)
                toRet[y][x][1] = 255/(t1-t0) * (a-t0)
                toRet[y][x][2] = 255/(t1-t0) * (a-t0)
            if a >= t1:
                toRet[y][x][0] = 255
                toRet[y][x][1] = 255
                toRet[y][x][2] = 255
    return toRet

def autoContrast(img, schwellwert):
    a1_low = getPercentage(img, schwellwert)
    a1_high = getPercentage(img, 1-schwellwert)

    return linearContrast(img, a1_low, a1_high)

def getPercentage(img, schwellwert):
    H = getCumulatedHistogramm(img)
    for i in range(len(H)):
        if H[i] >= schwellwert:
            return i


def getMedianOfHistogramm(img):
    H = getCumulatedHistogramm(img)
    for i in range(len(H)):
        if H[i] >= 0.5:
            return getHistogramm(img)[i]


def getBinary(img=[], schwellwert=125):
    toRet = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            a = 0.2125 * img[y][x][0] + 0.7154 * img[y][x][1] + 0.072 * img[y][x][2]
            if a < schwellwert:
                toRet[y][x] = 0
            else:
                toRet[y][x] = 1
    return toRet

    
# print(getMedianOfHistogramm(imgCol))

# plt.plot(getHistogramm(imgCol))
# plt.plot(getCumulatedHistogramm(imgCol))
# plt.show()

# img_con = linearContrast(imgCol, 5, 250)
# img_autoCon = autoContrast(imgCol, 0.05)
# plt.plot(getHistogramm(imgCol))
# plt.plot(getHistogramm(img_con))
# plt.plot(getHistogramm(img_autoCon))

# plt.plot(getHistogramm(img_autoCon))
# plt.show()

cv2.imshow("img", getBinary(imgCol, getMedianOfHistogramm(imgCol)))

# cv2.imshow("img", imgCol)
# cv2.imshow("img", linearContrast(imgCol, 0, 255))
cv2.waitKey(0)
cv2.destroyAllWindows()