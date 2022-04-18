import cv2
import os
from cv2 import bitwise_and
import numpy as np

# This Python Script will take given brick color ranges and return isolated image of just that brick

#Variables
color_range = 10
HUE_BLUE = 113
HUE_RED_1 = 174
HUE_RED_2 = 5
HUE_GREEN = 90
HUE_YELLOW = 30
area_limit = 50000 # if sum is less than 50,000 pixels we can assume we do not see a full brick. TODO find correct area estimate
color_state = '' # Start off with empty color state
"""
NOTE Red is a special case since color goes past total range we have to split into 2 cases
Threshold one is from 169 - 179, Threshold 2 is from 0 - 10. In this case our range from middle is only 5 not 10
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
def denoise_img(image, num_steps):
    kernel = np.ones((3,3), np.uint8)
    #Step one erode
    er = cv2.erode(image, kernel, iterations=1)
    #Step 2 dilate and fill missing values we removed
    dil = cv2.dilate(er, kernel, iterations=1)
    
    dil2 = cv2.dilate(dil, kernel, iterations=1)
    er2 = cv2.erode(dil2, kernel, iterations=1)
    return er2

#This function returns the hue color range for specific brick
def brick_range(state):
    colors = {
         0: [HUE_YELLOW, 'y'],
         1: [HUE_BLUE , 'b'],
         2: [HUE_GREEN, 'g'],
    }
    cur_h, cur_color = colors.get(state)
    lower = np.array([(cur_h - color_range), 100, 100], dtype=np.uint8)
    upper = np.array([(cur_h + color_range), 255, 255], dtype=np.uint8)
    return lower, upper, cur_color
        

file_path = './brick_pictures' ; # folder to read photos from
images = load_img_from_folder(file_path)
cur_img = images[10]
hsv = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV) # converts photo from RGB to HSV
# cv2.namedWindow('Color_Frame')
# cv2.namedWindow('Hue_Frame')

#TODO: delete this comment section. This is logic to isolate single color

"""
lower_range = np.array([(HUE_YELLOW - color_range), 100, 100], dtype=np.uint8)
upper_range = np.array([(HUE_YELLOW + color_range), 255, 255], dtype=np.uint8)
threshold = cv2.inRange(hsv, lower_range, upper_range ) #Capture threshold of brick within set color range
clean_mask = denoise_img(threshold)
area = np.count_nonzero(clean_mask)
print('Current area is: ' + str(area))
if area > area_limit:
    print('We found a brick!')
    print('Area is: ' + str(area))
"""

#NOTE if we want to merge mask with original image we can use result = bitwise_and(cur_img, clean_mask, mask = NONE)
while True:
    cv2.imshow('Color_Frame', cur_img)
    for i in range(4):
        if i == 3:
            #Special Case we have a red brick
            color_state = 'r'
            new_range = color_range / 2
            lower_range_1 = np.array([(HUE_RED_1 - new_range), 100, 100], dtype=np.uint8)
            lower_range_2 = np.array([(HUE_RED_2 - new_range), 100, 100], dtype=np.uint8)
            upper_range_1 = np.array([(HUE_RED_1 + new_range), 255, 255], dtype=np.uint8)
            upper_range_2 = np.array([(HUE_RED_2 + new_range), 255, 255], dtype=np.uint8)
            threshold_1 = cv2.inRange(hsv, lower_range_1, upper_range_1 )
            threshold_2 = cv2.inRange(hsv, lower_range_2, upper_range_2 )
            threshold = cv2.bitwise_or(threshold_1, threshold_2, mask=None) # combine both threshold images
            clean_mask = denoise_img(threshold, 5)
        else:
            lower_range, upper_range, color_state = brick_range(i)
            threshold = cv2.inRange(hsv, lower_range, upper_range ) #Capture threshold of brick within set color range
            clean_mask = denoise_img(threshold, 5)

        area = np.count_nonzero(clean_mask)
        print('Current area is: ' + str(area))
        if area > area_limit:
            #Count total number of white pixels in mask. If its above our area limit we found brick
            print('We found a brick!')
            print('Area is: ' + str(area))
            # cv2.imshow('Hue_Frame', clean_mask)
            # cv2.imshow('Before_denoise', threshold)
            #This is where we would now update status and move to grabbing mode
            
            img, contours, hierarchy = cv2.findContours(clean_mask, 1, 2)
            
            # cnt = contours[0]
            M = cv2.moments(img)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print(cx, cy)
            
            # rect = cv2.minAreaRect(hierarchy)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(cur_img,[box], 0, (0,0,255), 2)

            # x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(cur_img,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.circle(cur_img, (cx, cy), 10, (255, 0, 0), -1)
            cv2.imshow("Center_Detection", cur_img)

        else:
            print('NO Brick of color: {} found'.format(color_state))
        




    
    key = cv2.waitKey(1)
    if key == 27 or key == ord("q"): 
        break

cv2.destroyAllWindows()