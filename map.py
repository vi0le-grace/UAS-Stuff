import cv2 as cv
import numpy as np

#importing and reading the iamge
img = cv.imread("images/5.png")
cv.imshow('test_image', img)



#splitting the image into it's color channels
b,g,r = cv.split(img)

#applying threshold, trying to distingush the houses, because saw from the seperate channel images

#for blue

threshold, thresh_house_b = cv.threshold(b, 230, 255, cv.THRESH_BINARY)

#for red

threshold, thresh_house_r = cv.threshold(r, 230, 255, cv.THRESH_BINARY)

map_houses = cv.bitwise_or(thresh_house_b, thresh_house_r)

map_houses_inverse = cv.bitwise_not(map_houses)

map_houses_inverse_color = cv.cvtColor(map_houses_inverse, cv.COLOR_GRAY2BGR)

map_wo_houses = cv.bitwise_and(img, map_houses_inverse_color)


b2, g2, r2 = cv.split(map_wo_houses)

gauss = cv.bilateralFilter(g2, 20, 20, 10)
threshold, green_area = cv.threshold(gauss, 60, 255, cv.THRESH_BINARY)


green_area_in = cv.bitwise_not(green_area)

blank = np.zeros((640,640), dtype = 'uint8')
blank_white = cv.bitwise_not(blank)



red_area = cv.bitwise_and(map_houses_inverse, green_area_in)


#blue area for segment
threshold, blue_part = cv.threshold(green_area, 250, 230, cv.THRESH_BINARY) 

b_need = cv.bitwise_or(thresh_house_b, blue_part)


threshold, red_part = cv.threshold(green_area, 250, 173, cv.THRESH_BINARY) 

r_need1 = cv.bitwise_or(thresh_house_r, red_area)
r_need2 = cv.bitwise_or(r_need1,red_part )


threshold, lmao_part = cv.threshold(green_area, 250, 216, cv.THRESH_BINARY) 
g_need = cv.bitwise_or(red_area, lmao_part)


segmented_img = cv.merge([b_need, g_need, r_need2])
cv.imshow('segmented', segmented_img)

cv.waitKey(0)
cv.destroyAllWindows()