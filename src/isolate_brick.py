import cv2
import os
import numpy as np

# This Python Script will take given brick color ranges and return isolated image of just that brick

#Variables
color_range = 10
HUE_BLUE = 113
HUE_RED_1 = 174
HUE_RED_2 = 5
HUE_GREEN = 90
HUE_YELLOW = 26
area_limit = 90000 # if sum is less than 70,000 pixels we can assume we do not see a full brick. TODO find correct area estimate
"""
NOTE Red is a special case since color goes past total range we have to split into 2 cases
Threshold one is from 169 - 179, Threshold 2 is from 0 -10. In this case our range from middle is only 5 not 10
Then we have to combine results of both thresholds using bitwise_or() operator
"""




#This function reads every image from the folder
#TODO: Delete when switching to live camera
def load_img_from_folder(folder_path):
    pictures = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            pictures.append(img)
    return pictures

#This function removes isolated pixels
def denoise_img(image):
    kernel = np.ones((5,5), np.uint8)
    #Step one erode
    er = cv2.erode(image, kernel, iterations=1)
    #Step 2 dilate and fill missing values we removed
    dil = cv2.dilate(er, kernel, iterations=1)
    return dil

file_path = '/home/cvdarbeloff/Documents/2120/2.12-final-project/src/brick_pictures' ; # folder to read photos from
images = load_img_from_folder(file_path)
cur_img = images[0]
hsv = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV) # converts photo from RGB to HSV
cv2.namedWindow('Color_Frame')
cv2.namedWindow('Hue_Frame')

#TODO: update with some sort of switch logic for each of 4 brick colors
lower_range = np.array([(HUE_BLUE - color_range), 100, 100], dtype=np.uint8)
upper_range = np.array([(HUE_BLUE + color_range), 255, 255], dtype=np.uint8)
threshold = cv2.inRange(hsv, lower_range, upper_range ) #Capture threshold of brick within set color range
clean_mask = denoise_img(threshold)
area = np.count_nonzero(clean_mask)
if area > area_limit:
    print('We found a brick!')
    print('Area is: ' + str(area))
    #This is where we would now update status and move to grabbing mode
while True:
    cv2.imshow('Color_Frame', cur_img)
    cv2.imshow('Hue_Frame', clean_mask)
    cv2.imshow('Before_denoise', threshold)
    key = cv2.waitKey(1)
    if key == 27 or key == ord("q"): 
        break

cv2.destroyAllWindows()