import cv2 as cv
import numpy as np

#importing and reading the iamge
img = cv.imread("images/1.png")
cv.imshow('test_1', img)

#splitting the image into it's color channels
b,g,r = cv.split(img)
cv.imshow('blue',b)
cv.imshow('green',g)
cv.imshow('red',r)

#applying threshold, trying to distingush the houses, because saw from the seperate channel images

#for blue

threshold, thresh_house_b = cv.threshold(b, 230, 255, cv.THRESH_BINARY)
cv.imshow('thresh blue house', thresh_house_b)

#for red

threshold, thresh_house_r = cv.threshold(r, 230, 255, cv.THRESH_BINARY)
cv.imshow('thresh red house', thresh_house_r)

#trying to seperate out areas using threshold (fingers crossed)

#for green area
#trying different blurs
gauss = cv.bilateralFilter(g, 20, 20, 10)
threshold, thresh_area_green = cv.threshold(gauss, 60, 255, cv.THRESH_BINARY)
cv.imshow('green area gauss thresh', thresh_area_green)

#for brown area


threshold, thresh_area_green = cv.threshold(g, 55, 255, cv.THRESH_BINARY)
cv.imshow('green area thresh', thresh_area_green)

#creating a blank image with 640X640 dimensions, same as the imput image size

blank = np.zeros((640,640), dtype = 'uint8')
cv.imshow('blank', blank)

cv.waitKey(0)
cv.destroyAllWindows()