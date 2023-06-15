import cv2
import numpy as np#


def scale(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    result = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return result

def template_matching(image, template):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # grey_image = image
    # grey_template = template

    res = cv2.matchTemplate(grey_image, grey_template, cv2.TM_CCOEFF_NORMED)
    resNorm = cv2.normalize(res, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # cv2.imshow('adhg', resNorm)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for i in range(50):
        # print(i)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        print(maxLoc)
        xBegin = maxLoc[0] - 1
        yBegin = maxLoc[1] - 1

        xEnd = maxLoc[0] + template.shape[0] + 1
        yEnd = maxLoc[1] + template.shape[1] + 1

        cv2.rectangle(image, (xBegin, yBegin), (xEnd, yEnd), color=(255,255,0))

        neighbors = 4
        for j in range(neighbors):
            for j1 in range(neighbors):
                res[int(maxLoc[1] + neighbors // 2 - j)][int(maxLoc[0] + neighbors // 2 - j1)] = 0


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
