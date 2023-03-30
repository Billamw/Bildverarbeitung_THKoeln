import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

# print("Hell0 World!")

imgCol = cv2.imread('Utils\LenaJPEG.jpg')

# cv2.imshow("img", imgCol)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def getHistogram(imgCol):

    width, height, debth = imgCol.shape

    histogram = np.zeros(256)

    for y in range(height):
        for x in range(width):
            tmp = int(imgCol[x][y][0])
            histogram[tmp]+=1

    histogram/=width*height
    return histogram

def getCumulatedHistogram(histogram):

    result = np.zeros(256)

    for i in range(len(histogram)):
        for j in range(i):
            result[i] += histogram[j]

    return result

def linearContrastSpread(imgCol, t0, t1):

    width, height, debth = imgCol.shape

    image = np.zeros((width, height))

    for y in range(height):
        for x in range(width):
            if(imgCol[x][y][0] <= t0):
                image[x][y] = 0
            elif(t0 < imgCol[x][y][0] < t1):
                image[x][y] = 256/(t0-t1)*(imgCol[x][y][0] - t0)
            elif(imgCol[x][y][0] >= t1):
                image[x][y] = 256
    return image

def getMeanGreyValue(cumulatedHistogram, histogram):
    median = 0
    for i in range(len(cumulatedHistogram)):
        if(cumulatedHistogram[i] > 0.5):
            median = histogram[i]
            break
    return median

# cv2.imshow("img", linearContrastSpread(imgCol, 10, 180))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

width, height, debth = imgCol.shape

print(getMeanGreyValue(getCumulatedHistogram(getHistogram(imgCol)), getHistogram(imgCol))*width*height)

# plt.plot(getCumulatedHistogram(getHistogram(imgCol)))
# plt.show()