# This Python Script will take given brick color ranges and return isolated image of just that brick
# as well as the brick center location and orientation
import rospy
import cv2
import os
from cv2 import bitwise_and
import numpy as np
from geometry_msgs.msg import Pose

# Sofware flags
DEBUG = True
USE_CAMERA_FEED = False
DISPLAY_IMAGES = True

# Color isolation parameters
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

class BrickDetector:
    def __init__(self):
        self.publisher = rospy.Publisher("/brick_detector", Pose, queue_size=10)

    # This function reads every image from the folder
    def load_img_from_folder(self, folder_path):
        pictures = []
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                pictures.append(img)
        return pictures

    # This function removes isolated pixels
    def denoise_img(self, image):
        kernel = np.ones((3,3), np.uint8)
        # Erode image to remove noise, then dilate image to return it to original size  
        eroded_image_1 = cv2.erode(image, kernel, iterations=3)
        dilated_image_1 = cv2.dilate(eroded_image_1, kernel, iterations=3)
        
        # Dilate image to get rid of empty spots in brick blob, then erode image to return it to original size
        dilated_image_2 = cv2.dilate(dilated_image_1, kernel, iterations=2)
        eroded_image_2 = cv2.erode(dilated_image_2, kernel, iterations=2)
        return eroded_image_2

    # This function returns the hue color range for specific brick
    def brick_hue_range(self, state):
        colors = {
            0: [HUE_YELLOW, 'yellow'],
            1: [HUE_BLUE , 'blue'],
            2: [HUE_GREEN, 'green'],
        }
        cur_h, cur_color = colors.get(state)
        lower = np.array([(cur_h - color_range), 100, 100], dtype=np.uint8)
        upper = np.array([(cur_h + color_range), 255, 255], dtype=np.uint8)
        return lower, upper, cur_color
        
    def init():
        if not USE_CAMERA_FEED:
            # TODO: Call method to use live camera feed?
            pass
        else:
            file_path = './brick_pictures'
            images = load_img_from_folder(file_path)
            cur_img = images[9]
        hsv = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV) # converts photo from RGB to HSV

        # NOTE: if we want to merge mask with original image we can use result = bitwise_and(cur_img, clean_mask, mask = NONE)
        while True:
            for i in range(4):
                if i == 3:
                    # Special Case we have a red brick
                    color_state = 'red'
                    new_range = color_range / 2
                    lower_range_1 = np.array([(HUE_RED_1 - new_range), 100, 100], dtype=np.uint8)
                    lower_range_2 = np.array([(HUE_RED_2 - new_range), 100, 100], dtype=np.uint8)
                    upper_range_1 = np.array([(HUE_RED_1 + new_range), 255, 255], dtype=np.uint8)
                    upper_range_2 = np.array([(HUE_RED_2 + new_range), 255, 255], dtype=np.uint8)
                    threshold_1 = cv2.inRange(hsv, lower_range_1, upper_range_1 )
                    threshold_2 = cv2.inRange(hsv, lower_range_2, upper_range_2 )
                    threshold = cv2.bitwise_or(threshold_1, threshold_2, mask=None) # combine both threshold images
                else:
                    lower_range, upper_range, color_state = brick_hue_range(i)
                    threshold = cv2.inRange(hsv, lower_range, upper_range ) #Capture threshold of brick within set color range

                # Count total number of white pixels in mask. If it's above our area limit we found brick
                clean_mask = denoise_img(threshold)
                area = np.count_nonzero(clean_mask)
                if area > area_limit:
                    if DISPLAY_IMAGES:
                        three_chan_clean_mask = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
                        displayed_img = np.concatenate((cur_img, three_chan_clean_mask), axis=1)
                        cv2.imshow('Detected Brick and Denoised Brick Mask', displayed_img)
                    
                    # Calculate brick center
                    M = cv2.moments(clean_mask)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.circle(cur_img, (cx, cy), 10, (255, 0, 0), -1)
                    
                    # Find contours of brick to get bounding box
                    contour_results = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contour_results[-2]
                    cv2.drawContours(cur_img, contours, -1, (0, 255, 0), 3)

                    contours = np.array(contours)
                    contours = np.concatenate(contours).ravel()
                    contours = np.reshape(contours, (len(contours)//2, 2))
                    
                    rect = cv2.minAreaRect(np.array(contours))
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(cur_img, [box], 0, (0,0,255), 2)
                    
                    highest_point_index = np.argmin(box, axis=0)[1]
                    highest_point_neighbor_1_index = highest_point_index + 1 if highest_point_index + 1 < len(box) else 0
                    highest_point_neighbor_2_index = highest_point_index - 1 if highest_point_index - 1 > -1 else len(box) - 1
                    highest_point = box[highest_point_index]
                    highest_point_neighbor_1 = box[highest_point_neighbor_1_index]
                    highest_point_neighbor_2 = box[highest_point_neighbor_2_index]
                    first_side = np.linalg.norm(highest_point - highest_point_neighbor_1)
                    second_side = np.linalg.norm(highest_point - highest_point_neighbor_2)
                    cv2.circle(cur_img, (highest_point[0], highest_point[1]), 5, (0, 255, 255), -1)
                    if first_side > second_side:
                        cv2.circle(cur_img, (highest_point_neighbor_1[0], highest_point_neighbor_1[1]), 5, (0, 0, 255), -1)
                        # print("Long side is side 1, between", highest_point, highest_point_neighbor_1, first_side)
                        brick_angle = np.arctan2(highest_point_neighbor_1[1] - highest_point[1], highest_point_neighbor_1[0] - highest_point[0]) * (180.0/np.pi)
                    else:
                        cv2.circle(cur_img, (highest_point_neighbor_2[0], highest_point_neighbor_2[1]), 5, (0, 0, 255), -1)
                        # print("Long side is side 2, between", highest_point, highest_point_neighbor_2, second_side)
                        brick_angle = np.arctan2(highest_point_neighbor_2[1] - highest_point[1], highest_point_neighbor_2[0] - highest_point[0]) * (180.0/np.pi)
                    
                    min_brick_angle = min(brick_angle, abs(180 - brick_angle))
                    if DEBUG:
                        print('{} brick detected with area of {}, center at ({}, {}), rotated by {:.3f}, bounding box of {}.'
                        .format(color_state.capitalize(), area, cx, cy, min_brick_angle, [[box[i][0], box[i][1]] for i in range(len(box))]))
                # else:
                #     if DEBUG:
                #         print('No Brick of color: {} found'.format(color_state))
                
            
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"): 
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node("brick_detector")
    init()