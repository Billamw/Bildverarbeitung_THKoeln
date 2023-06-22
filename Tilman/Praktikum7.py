import cv2
import numpy as np
import matplotlib as plt


def scale(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    result = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return result

def template_matching(image, template, res):
    #grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #grey_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # grey_image = image
    # grey_template = template

    #res = cv2.matchTemplate(grey_image, grey_template, cv2.TM_CCOEFF_NORMED)
    #resNorm = cv2.normalize(res, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # cv2.imshow('adhg', resNorm)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for i in range(50):
        # print(i)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        xBegin = maxLoc[1] - 1
        yBegin = maxLoc[0] - 1

        xEnd = maxLoc[1] + template.shape[1] + 1
        yEnd = maxLoc[0] + template.shape[0] + 1

        cv2.rectangle(image, (xBegin, yBegin), (xEnd, yEnd), color=(255,255,0))

        neighbors = 4
        for j in range(neighbors):
            for j1 in range(neighbors):
                if(maxLoc[1] + neighbors // 2 - j < res.shape[1] and int(maxLoc[0] + neighbors // 2 - j1) < res.shape[0]):
                    res[int(maxLoc[1] + neighbors // 2 - j)][int(maxLoc[0] + neighbors // 2 - j1)] = 0


    return image

def rotate_image(mat, angle):
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=255)
    return rotated_mat

allResults = {}



image = cv2.imread('Utils/Uebung08/setGameNew.jpg')
template = cv2.imread('Utils/Uebung08/TemplateWelle.png')

image = scale(image, 15)
template = scale(template, 15)

for idxScale, scale in enumerate((0.5, 0.6, 0.7, 0.8)):
    templateCurrentScale = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
    
    for idxRot, rotation in enumerate(range(0, 350, 15)):
        templateCurrentRotation = rotate_image(templateCurrentScale, rotation)
        res = cv2.matchTemplate(image, templateCurrentRotation, cv2.TM_CCOEFF_NORMED)
        
        if idxScale not in allResults:
            allResults[idxScale] = {}
        
        allResults[idxScale][idxRot] = res

        
print( f"allResults: {len(allResults)} x {len(allResults[0])}" )

cv2.imshow("geht das?", template_matching(image, template, allResults[2][2]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Führe das Template Matching durch
result_image = template_matching(image, template)

# Zeige das Ergebnisbild
cv2.imshow('Template Matching', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
