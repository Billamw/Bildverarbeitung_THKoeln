import numpy as np
import cv2
import matplotlib.pyplot as plt

def printHistogram(histogram, bins, title=""):
    plt.figure(figsize=(8, 6))
    plt.bar(bins[:-1], histogram, width=1, color='gray')
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

def printGraph(data_array, title="", x_label="", y_label=""):
    plt.figure(figsize=(8, 6))
    plt.plot(data_array)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

def linearHistogramEqualization(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = np.zeros_like(gray_image)
    height, width = gray_image.shape

    histogram, bins = np.histogram(gray_image, bins=256, range=(0, 256))
    cdf = np.cumsum(histogram)

    printHistogram(histogram, bins, title="hist1")
    printGraph(cdf, title="cdf1")

    for x in range(width):
        for y in range(height):
            a = gray_image[y][x]
            output[y][x] = (cdf[a]/(width*height))*(255-1)

    histogram, bins = np.histogram(output, bins=256, range=(0, 256))
    cdf = np.cumsum(histogram)
    
    printHistogram(histogram, bins, title="hist2")
    printGraph(cdf, title="cdf2")
    plt.show()

    return output

image = cv2.imread("Utils/Uebung 10/Rugby.png")
output = linearHistogramEqualization(image)

cv2.imshow('original', cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
cv2.imshow('output', output)
cv2.waitKey(0)   #wait for a keyboard input
cv2.destroyAllWindows()