import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("start")

imageName = "cameraAnalogue01.jpg"

pathPre = "Utils/cameraAnalogue01.jpg"

imagePath = f"{pathPre} {imageName}"

def getImage(path):
    if os.path.exists(path):
        imagePath = path
    else:
        pathPre = "../../"
        imagePath = f"{pathPre} {path}"

    image = cv2.imread(imagePath)

    return image

def getResizedImage(image):
    height, width, _ = image.shape
    imageResized = cv2.resize(image, (int(width/5), int(height/5)))
    return imageResized

def HoughTransformation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Hough-Transformation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

HoughTransformation(getImage("Utils/TestImgLine.png"))

print("finish")