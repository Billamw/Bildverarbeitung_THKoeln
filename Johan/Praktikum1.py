# 23.02.2023
import cv2 as cv2
import numpy as np

# print("Hello, World!")

def printNumberFrom1To100():
    for i in range(1, 101):
        print(i)

def fibunacci(n):
    a = [0]*(n+4)
    a[0] = 1
    a[1] = 1
    print(0)
    print(1)
    print(1)
    for i in range(2,n+3):
        a[i] = a[i-1]+a[i-2]
        print(a[i])

# fibunacci(10)

imgCol = cv2.imread('Utils\LenaJPEG.jpg')

width, height, depth = imgCol.shape

for y in range(height):
    for x in range(width):
        imgCol[y][x] = pow(imgCol[y][x]/255, 1/2) * 255


cv2.imshow("img", imgCol)
cv2.waitKey(0)
cv2.destroyAllWindows()