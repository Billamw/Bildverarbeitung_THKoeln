import cv2
import numpy as np

print("Start...")

print("Aufgabe 1 ...", end="")

img = cv2.imread('../../Utils/Uebung 10/Rugby.png', cv2.IMREAD_GRAYSCALE)

img = cv2.GaussianBlur(img, (5, 5), 0)

edges = cv2.Canny(img, threshold1=30, threshold2=100)

cv2.imshow('Kantenbild', edges)

print("done")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Aufgabe 2 ...", end="")

integral = cv2.integral(edges);
integral_scaled = cv2.normalize(integral, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

cv2.imshow('Kantenbild', integral_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("done")

print("Aufgabe 3 ...", end="")

height, width = edges.shape
rec_height, rec_width = 50, 50

for y in range(0, height - rec_height):
    for x in range(0, width - rec_width):
        # Bestimmen Sie die Eckpunkte des Rechtecks
        top_left = integral_scaled[y, x]
        top_right = integral_scaled[y, x + rec_width]
        bottom_left = integral_scaled[y + rec_height, x]
        bottom_right = integral_scaled[y + rec_height, x + rec_width]

        sum_in_rec = bottom_right - top_right - bottom_left + top_left

print("done")

print("...End")