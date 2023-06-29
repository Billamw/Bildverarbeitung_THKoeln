import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

width = 1920
height = 1080

image = np.zeros((height, width, 3), dtype=np.uint8)

image[:] = (255, 255, 255)

image[0:height, 0] = (0, 0, 0)  # Linker Rand
image[0:height, width-1] = (0, 0, 0)  # Rechter Rand
image[0, 0:width] = (0, 0, 0)  # Oberer Rand
image[height-1, 0:width] = (0, 0, 0)  # Unterer Rand

width -= 1
height -= 1

startX = 1
startY = 1

colorStart = (255, 0, 0)
colorEnd = (0, 0, 255)

for i in range(10):
    image[:, i] = colorStart

for i in range(10):
    image[:, (width-1)-i] = colorEnd

#### Linear Interpolation

# for y in range(startY, height):
#     for x in range(startX, width):
#         red = int(colorStart[0] * (x / width) + colorEnd[0] * (1 - x / width))
#         green = int(colorStart[1] * (x / width) + colorEnd[1] * (1 - x / width))
#         blue = int(colorStart[2] * (x / width) + colorEnd[2] * (1 - x / width))

#         image[y, x] = (red, green, blue)

# color01 = (255, 0, 0)
# color02 = (0, 0, 255)
# color03 = (0, 255, 0)
# color04 = (0, 0, 0)

color01 = image[1,1] = (0, 255, 0) # oben links
color02 = image[1,width-1] = (0, 255, 0) # oben rechts
color03 = image[height-1,1] = (255, 0, 0) # unten links 
color04 = image[height-1,width-1] = (255, 0, 0) # unten rechts


for y in range(startY, height):
     for x in range(startX, width):
         red = int((color01[0] * (x / width) * color02[0] * (1 - x / width)))
         green = int((color01[1] * (x / width) + color02[1] * (1 - x / width)))
         blue = int((color01[2] * (x / width) + color02[2] * (1 - x / width)))
         

         red = max(0, min(red, 255))
         green = max(0, min(green, 255))
         blue = max(0, min(blue, 255))

         image[y, x] = (red, green, blue)

        
#  + (color03[0] * (y / height) + color04[0] * (1 - y / height))
#  + (color03[1] * (y / height) + color04[1] * (1 - y / height))
#  + (color03[2] * (y / height) + color04[2] * (1 - y / height))

plt.imshow(image)
plt.axis('off')  # Optionales Entfernen der Achsenbeschriftungen
plt.show()