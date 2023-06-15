import cv2
import numpy as np

# def template_matching(image, template):
#     # Konvertiere das Bild und die Vorlage in Graustufen
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#     # Führe das Template Matching durch
#     result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

#     # Finde die Position(en) des besten Matches
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     top_left = max_loc
#     bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

#     # Zeichne ein Rechteck um das beste Match
#     cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

#     # Gib das Bild mit dem markierten Match zurück
#     return image

def scale(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    result = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return result

def template_matching(image, template):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(grey_image, grey_template, cv2.TM_CCOEFF_NORMED)
    resNorm = cv2.normalize(res, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    cv2.imshow('adhg', resNorm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for i in range(50):
        #print('affe')
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

        xBegin = maxLoc[0] - 1
        yBegin = maxLoc[1] - 1

        xEnd = maxLoc[0] + template.shape[0] - 1
        yEnd = maxLoc[1] + template.shape[1] - 1

        cv2.rectangle(image, (xBegin, yBegin), (xEnd, yEnd), color=(255,255,0))

        for x in range(int(template.shape[0]/2)):
            for y in range(int(template.shape[1]/2)):
                if(resNorm[x][y] < maxVal):
                    resNorm[x][y] = 0

        return image



image = cv2.imread('Utils/Uebung08/bookPage.png')
template = cv2.imread('Utils/Uebung08/Template_a.png')

image = scale(image, 50)
template = scale(template, 50)

# Führe das Template Matching durch
result_image = template_matching(image, template)

# Zeige das Ergebnisbild
cv2.imshow('Template Matching', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
