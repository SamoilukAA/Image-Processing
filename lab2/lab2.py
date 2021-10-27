import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def NlMeans(img):
    dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    b, g, r = cv.split(dst) # get b, g, r
    rgb_dst = cv.merge([r, g, b]) # switch it to rgb
    return rgb_dst

def Median(img):
    result = cv.medianBlur(img, 5)
    b, g, r = cv.split(result) # get b, g, r
    rgb_res = cv.merge([r, g, b]) # switch it to rgb
    return rgb_res

def Gaussian(img):
    result = cv.GaussianBlur(img, (5,5), 0)
    b, g, r = cv.split(result) # get b, g, r
    rgb_res = cv.merge([r, g, b]) # switch it to rgb
    return rgb_res

def CreateMarkers(img, num):
    markers = np.zeros((img.shape[0], img.shape[1] ), dtype = "int32")
    if (num == 5):
        markers[90:140, 90:140] = 255
        markers[200:255, 0:55] = 1
        markers[0:20, 0:20] = 1
        markers[0:20, 236:255] = 1
        markers[236:255, 236:255] = 1
    elif (num == 8):
        markers[90:140, 90:140] = 255
        markers[236:255, 0:20] = 1
        markers[0:40, 0:40] = 1
        markers[0:20, 236:255] = 1
        markers[236:255, 236:255] = 1
    else:
        markers[90:140, 90:140] = 255
        markers[236:255, 0:20] = 1
        markers[0:20, 0:20] = 1
        markers[0:20, 236:255] = 1
        markers[236:255, 236:255] = 1
    return markers

def CalcOfDamageAndNonDamage(img, num):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    image_erode = cv.erode(img, kernel)
    
    hsv_img = cv.cvtColor(image_erode, cv.COLOR_BGR2HSV)
    
    markers = CreateMarkers(img, num)
    
    leafs_area_BGR = cv.watershed(image_erode, markers)    
    healthy_part = cv.inRange(hsv_img, (36, 25, 25), (86, 255, 255))
    ill_part = leafs_area_BGR - healthy_part 
    
    mask = np.zeros_like(img, np.uint8)
    mask[leafs_area_BGR > 1] = (255, 0, 255)
    mask[ill_part > 1] = (0, 0, 255)
    return mask

img = cv.imread("8.jpg")
num = 8
nlm_img = NlMeans(img)
m_img = Median(img)
g_img = Gaussian(img)
b, g, r = cv.split(img) # get b, g, r
rgb_img = cv.merge([r, g, b]) # switch it to rgb
mask_nlm = CalcOfDamageAndNonDamage(nlm_img, num)
mask_m = CalcOfDamageAndNonDamage(m_img, num)
mask_g = CalcOfDamageAndNonDamage(g_img, num)

plt.subplot(421), plt.imshow(rgb_img)
plt.subplot(423), plt.imshow(nlm_img)
plt.subplot(424), plt.imshow(mask_nlm)
plt.subplot(425), plt.imshow(m_img)
plt.subplot(426), plt.imshow(mask_m)
plt.subplot(427), plt.imshow(g_img)
plt.subplot(428), plt.imshow(mask_g)
plt.show()