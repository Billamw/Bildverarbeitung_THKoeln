from __future__ import print_function
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import argparse

# print("Hell0 World!")

# imgCol = cv2.cvtColor(cv2.imread('Utils/bookPage.png'), cv2.COLOR_BGR2GRAY)
original = cv2.imread('Utils/IMG_1737.jpg')

imgCol = cv2.cvtColor(cv2.imread('Utils/Beispiel_Tilman2.jpg' ), cv2.COLOR_BGR2GRAY)

width, height = imgCol.shape

scale_percent = 15 # percent of original size
width = int(imgCol.shape[1] * scale_percent / 100)
height = int(imgCol.shape[0] * scale_percent / 100)
dim = (width, height)
imgCol = cv2.resize(imgCol, dim, interpolation = cv2.INTER_AREA)
original = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)

colors = [
  [255, 0, 0],   # Rot
  [0, 255, 0],   # Grün
  [0, 0, 255],   # Blau
  [255, 255, 0], # Gelb
  [255, 0, 255], # Magenta
  [0, 255, 255], # Cyan
  [128, 0, 0],   # Dunkelrot
  [0, 128, 0],   # Dunkelgrün
  [0, 0, 128],   # Dunkelblau
  [128, 128, 0], # Dunkelgelb
  [128, 0, 128], # Dunkelmagenta
  [0, 128, 128], # Dunkelcyan
  [192, 192, 192], # Hellgrau
  #[128, 128, 128], # Grau
  #[255, 255, 255], # Weiß
  #[0, 0, 0],     # Schwarz
  [255, 128, 128], # Hellrot
  [128, 255, 128], # Hellgrün
  [128, 128, 255], # Hellblau
  [255, 255, 128], # Hellgelb
  [255, 128, 255], # Hellmagenta
  [128, 255, 255], # Hellcyan
  [255, 0, 128],   # Pink
  [128, 255, 0],   # Limette
  [0, 128, 255],   # Himmelblau
  [255, 128, 0],   # Orange
  [128, 0, 255],   # Lila
  [0, 255, 128],   # Türkis
  [255, 128, 64],  # Pfirsich
  [64, 255, 128],  # Limonengrün
  [128, 64, 255],  # Violett
]

# cv2.imshow["img", imgCol)
# cv2.waitKey[0)
# cv2.destroyAllWindows[)

def showPicture(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getHistogram(imgCol):

    width, height = imgCol.shape

    histogram = np.zeros(256)

    for y in range(height):
        for x in range(width):
            tmp = int(imgCol[x][y])
            histogram[tmp]+=1

    #histogram/=width*height
    return histogram

def getCumulatedHistogram(histogram):

    result = np.zeros(256)

    for i in range(len(histogram)):
        for j in range(i):
            result[i] += histogram[j]

    return result

def linearContrastSpread(imgCol, t0, t1):

    width, height = imgCol.shape

    image = np.zeros((width, height))

    count1 = 0
    count2 = 0
    count3 = 0

    print(t0, t1)

    for y in range(height):
        for x in range(width):
            if(imgCol[x][y] <= t0):
                image[x][y] = 0
                count1+=1
            elif(t0 < imgCol[x][y] < t1):
                image[x][y] = int((255/(t1-t0))*(imgCol[x][y] - t0))/255
                count2+=1
                # print(imgCol[x][y], "==>", image[x][y])
            elif(imgCol[x][y] >= t1):
                image[x][y] = 1
                count3+=1
    print(count1, "+", count2, "+", count3, "=", count1+count2+count3, "Amount of Pixels:", height*width)
    return image

def getMeanGreyValue(cumulatedHistogram, histogram):
    median = 0
    for i in range(len(cumulatedHistogram)):
        if(cumulatedHistogram[i] > 0.5):
            median = histogram[i]
            break
    return median


def binarize(imgCol, t):
    width, height = imgCol.shape
    result = np.zeros((width, height))
    for y in range(height):
        for x in range(width):
            if(imgCol[x][y] > t):
                result[x][y] = 1
            else:
                result[x][y] = 0
    
    return result

def automizedBinarize(imgCol, cumulatedHistogram):
    tIsFound = False
    for x in range(len(cumulatedHistogram)):
        if(cumulatedHistogram[x] >= 0.05 and tIsFound == False):
            t = x
            tIsFound = True
            break
    return binarize(imgCol, t)


def automizedLinearContrastSpread(imgCol, cumulatedHistogram):
    #define t_0 and t_1: 
    t_0IsFound = False
    t_1IsFound = False
    for x in range(len(cumulatedHistogram)):
        if(cumulatedHistogram[x] >= 0.05 and t_0IsFound == False):
            t_0 = x
            t_0IsFound = True
        elif(cumulatedHistogram[x] >= 0.95 and t_1IsFound == False):
            t_1 = x
            t_1IsFound = True
        if(t_0IsFound and t_1IsFound):
            break
    print(t_0, t_1)
    return linearContrastSpread(imgCol, t_0, t_1)

def inverseImage(imgCol):
    image = np.zeros((width, height))
    for y in range(height):
        for x in range(width):
            image[x][y] = (255 - imgCol[x][y])/255
            # print("255 -", imgCol[x][y], "=", image[x][y])
    print("done!")
    return image

def exposure(imgCol, a):
    image = np.zeros((width, height))
    for y in range(height):
        for x in range(width):
            image[x][y] = (imgCol[x][y] + a)/255
            if(image[x][y]>255):
                image[x][y] = 1
            elif(image[x][y] == 0):
                image[x][y] = 0
    return image

def otsu_naive(histogram):
    result = np.zeros((width, height))

    tStar = -1
    varMax = -1

    for i in range(255):
        w_0 = np.sum(histogram[0:i]) / width*height
        w_1 = np.sum(histogram[i:256]) / width*height
        mu_0 = np.mean(histogram[0:i])
        mu_1 = np.mean(histogram[i:256])
        varBetween = w_0 * w_1 * (mu_0 - mu_1)**2
        if(varMax < varBetween):
            varMax = varBetween
            tStar = i
    print("naive otsu:", tStar)
    return tStar
    
def otsu_efficient(histogram):
    tStar = -1
    varMax = -1
    c_0 = 0
    sum_0 = 0
    sum_h = weightetHistogram(histogram)
    for i in range(255):
        c_0 = c_0 + histogram[i]
        c_1 = width*height - c_0
        sum_0 = sum_0 + i * histogram[i]
        mu_0 = sum_0 / c_0
        mu_1 = (sum_h - sum_0) / c_1
        varBetween = c_0 * c_1 * (mu_0 - mu_1)**2
        if(varMax < varBetween):
            varMax = varBetween
            tStar = i
    print("efficient otsu:", tStar)
    return tStar

def weightetHistogram(histogram):
    result = 0
    for i in range(len(histogram)):
        result += i * histogram[i]
    return result

def segmentation(img):
    width, height = img.shape
    result = np.array((width, height))
    label = 2
    print(width, height)
    for y in range(height):
        for x in range(width):
            # print(x, y)
            if(img[x][y] == 1):
                
                result = floodFill(img, x, y, label)
                label+=1

    print(label)
    return result

def floodFill(img, x, y, label):
    stack = []
    stack.append((x,y))
    while(stack):
        tmp = stack.pop()
        x = tmp[0]
        y = tmp[1]
        if(checkBounds(x,y, height, width) and img[x][y] == 1):
            img[x][y] = label
            stack.append((x+1, y  ))
            stack.append((x  , y+1))
            stack.append((x  , y-1))
            stack.append((x-1,   y))
        
    return img

def checkBounds(x, y, width, height):
    
    result = False
    if(0<x<width and 0<y<height):
        result = True
    return result

def drawSegmentedImage(img):

    width, height = img.shape
 
    result = np.zeros(((width, height, 3)))

    for y in range(height):
        for x in range(width):

            if(not img[x][y] == 0):
                # print(int(img[x][y]), colors[int(img[x][y])]) 
                result[x][y] = colors[int((img[x][y])%(len(colors)-1))]

            #if(result[x][y][0] >= 255 and result[x][y][1] >= 255 and result[x][y][2] >= 255):
                #print("error, label:", img[x][y])

    return result

def normalizeColors():
    for y in range(len(colors)):
        colors[y][0] /= 255
        colors[y][1] /= 255
        colors[y][2] /= 255

def findContours(img):
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    # contours = np.array(())
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

histogram = getHistogram(imgCol)
# cumulatedHistogram = getCumulatedHistogram(histogram)
# meanGreyValue = getMeanGreyValue(cumulatedHistogram, histogram*width*height)
# autoLinSpread = automizedLinearContrastSpread(imgCol, cumulatedHistogram)
# invImg = inverseImage(exposure(imgCol, 20)*255)

# print(meanGreyValue)

# plt.plot(histogram)
# plt.show()
# plt.plot(cumulatedHistogram)
# plt.show()

#contrast = linearContrastSpread(imgCol, 30, 225)

#showPicture("linearContrastSpread()", contrast)

#showPicture("Original, color: ", original)

#showPicture("Original, BW: ", imgCol)

# showPicture("contours: ", findContours(imgCol))

cardImage1 = binarize(imgCol, otsu_efficient(histogram))
#showPicture("binarize()", cardImage1)

kernel = np.ones((3,3), np.uint8)

cardImage1 = cv2.erode(cardImage1, kernel, iterations=1)
#cardImage1 = cv2.dilate(cardImage1, kernel, iterations=1)
#showPicture("binarized, dilate, erode", cardImage1)

#normalizeColors()

#segmentedImage = segmentation(cardImage1)

# showPicture("contours: ", findContours(segmentedImage))

#contours = findContours(cardImage1)

ret, thresh = cv2.threshold(imgCol, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

test = cv2.drawContours(imgCol, contours, -1, (255,255,255), 3)

showPicture("test", test)

# showPicture("automzedBinarize()", automizedBinarize(imgCol, cumulatedHistogram))

# showPicture("original", imgCol)

# showPicture("automizedLinearContrastSpread()", automizedLinearContrastSpread(imgCol, cumulatedHistogram))

# print(imgCol.max(), imgCol.min())

# inv = inverseImage(imgCol) * 255

# testPic1 = exposure(inv, 60)

# cv2.imshow("invertiert, überbelichtet gescannt", testPic1)
# cv2.imshow("verarbeitet, überbelichtet gescannt", inverseImage(testPic1*255))
# cv2.imshow("invertiert, optimaler belichtet", inv/255)
# cv2.imshow("verarbeitet, optimaler belichtet", inverseImage(inv))
# cv2.waitKey(0)
# cv2.destroyAllWindows()