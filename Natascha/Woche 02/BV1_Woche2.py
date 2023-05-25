import cv2 as cv2
import numpy as np

# Pfad zum Bild angeben
Path_Lenna = 'Utils/LenaJPEG.jpg'
Path_Ferrari = 'Utils/Ferrari_2004.jpg'
Path_Analogkamera = 'Utils/cameraAnalogue02.jpg'

# Bild einlesen
Lenna = cv2.imread(Path_Lenna)
Ferrari = cv2.imread(Path_Ferrari)
Analogkamera = cv2.imread(Path_Analogkamera)

# Bild anzeigen
# cv2.imshow('Bild', Analogkamera)

# Auf eine Taste warten, um das Fenster zu schließen
# cv2.waitKey(0)

# Alle geöffneten Fenster schließen
# cv2.destroyAllWindows()

# ----- Aufgabe 01 ------------------------------------------

def Histogram(image, mode=0):
    if mode == 1:
        r = 0.3;
        g = 0.59;
        b = 0.11;
    elif mode == 2:
        print("es gibt keinen mode 2")
    elif mode == 3:
        print("es gibt keinen mode 3")
    else:
        r = 1/3;
        g = 1/3;
        b = 1/3;

    print("r = " + str(r) + "\n" + "g = " + str(g) + "\n" + "b = " + str(b));

    # Höhe und Breite des Bildes abrufen
    height, width, channels = image.shape

    imageGray = np.empty((height, width), dtype=np.uint8)

    # Über die Pixel des Bildes iterieren
    for y in range(height):
        for x in range(width):
            imageGray[y][x] = int(r * image[y][x][2] + g * image[y][x][1] + b * image[y][x][0]);

    cv2.imshow('Bild', imageGray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


Histogram(Analogkamera,0)