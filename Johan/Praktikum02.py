# 30.03.2023
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

imgCol = cv2.imread('Utils\LenaJPEG.jpg')

width, height, depth = imgCol.shape

def getHistogramm(img):
    histogram=np.zeros(256)
    
    for y in range(height):
        for x in range(width):
            i = 0.2125 * img[y][x][0] + 0.7154 * img[y][x][1] + 0.072 * img[y][x][2]
            # i = img[y][x][0]
            histogram[int(i)] += 1
    return histogram

def getNormHistogramm(img):
    histogram=np.zeros(256)
    
    for y in range(height):
        for x in range(width):
            i = 0.2125 * img[y][x][0] + 0.7154 * img[y][x][1] + 0.072 * img[y][x][2]
            # i = img[y][x][0]
            histogram[int(i)] += 1
    histogram/=(width*height)
    return histogram

def getCumulatedHistogramm(img):
    H = np.zeros(256)
    h = getNormHistogramm(img)
    for i in range(len(h)):
        for j in range(i): 
            H[i] += h[j]
    return H

def linearContrast(img, t0, t1):
    toRet = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            a = 0.2125 * img[y][x][0] + 0.7154 * img[y][x][1] + 0.072 * img[y][x][2]
            if a < t0:
                toRet[y][x] = 0
            if t0 <= a <= t1:
                toRet[y][x] = 255/(t1-t0) * (a-t0)
            if a >= t1:
                toRet[y][x] = 255
    return toRet

def autoContrast(img):
    toRet = np.zeros((height, width))
    tmp = getHistogramm(img)[getPercentage(img, 0.05):getPercentage(img, 1-0.05)]
    for y in range(height):
        for x in range(width):
            a = 2

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


def getBinary(img, schwellwert):
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
# plt.plot(getCumulatedHistogram(imgCol))
plt.plot(getCumulatedHistogramm(imgCol))
plt.show()
# cv2.imshow("img", getBinary(imgCol, 125))
# cv2.waitKey(0)
# cv2.destroyAllWindows()