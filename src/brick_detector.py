# This Python Script will take given brick color ranges and return isolated image of just that brick
# as well as the brick center location and orientation
import rospy
from geometry_msgs.msg import Pose
#!/usr/bin/env python2
import cv2
import os
from cv2 import bitwise_and
from cv2 import HoughCircles
from cv2 import HOUGH_GRADIENT
from cv2 import COLOR_BGR2GRAY
from cv2 import boundingRect
import numpy as np

# Sofware flags
DEBUG = False
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
color_state = "" # Start off with empty color state
"""
NOTE Red is a special case since color goes past total range we have to split into 2 cases
Threshold one is from 169 - 179, Threshold 2 is from 0 - 10. In this case our range from middle is only 5 not 10
Then we have to combine results of both thresholds using bitwise_or() operator
"""

class BrickDetector:
    def __init__(self):
        self.brick_detector_pub = rospy.Publisher("brick_detector", Pose, queue_size=10)   

        if USE_CAMERA_FEED:
            #Live Camera Information
            self.camera = cv2.VideoCapture(4) #4 is the port for my external camera
            # Show error if camera doesnt show up
            if not self.camera.isOpened():
                raise Exception("Could not open video device")
            # Set picture Frame. High quality is 1280 by 720
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            file_path = "./brick_pictures"
            images = self.load_img_from_folder(file_path)
            self.raw_img = images["Brick_photo_6.jpg"]
        
        rospy.Timer(rospy.Duration(1.0/20.0), self.detect_brick)
        # self.detect_brick(None)

    #This function reads every image from the folder
    #TODO: Delete when switching to live camera
    def load_img_from_folder(self, folder_path):
        pictures = {}
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                pictures[filename] = img
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
            0: [HUE_YELLOW, "yellow"],
            1: [HUE_BLUE , "blue"],
            2: [HUE_GREEN, "green"],
        }
        cur_h, cur_color = colors.get(state)
        lower = np.array([(cur_h - color_range), 100, 100], dtype=np.uint8)
        upper = np.array([(cur_h + color_range), 255, 255], dtype=np.uint8)
        return lower, upper, cur_color
            
    def detect_brick(self, event):
        # NOTE: if we want to merge mask with original image we can use result = bitwise_and(raw_img, clean_mask, mask = NONE)
        # while True:
        if USE_CAMERA_FEED:
            ret, self.raw_img = self.camera.read() #get current image feed from camera
            if not ret:
                print("Error. Unable to capture Frame")
                return
        self.hsv = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2HSV) # converts photo from RGB to HSV
        for i in range(4):
            if i == 3:
                # Special Case we have a red brick
                color_state = "red"
                new_range = color_range / 2
                lower_range_1 = np.array([(HUE_RED_1 - new_range), 100, 100], dtype=np.uint8)
                lower_range_2 = np.array([(HUE_RED_2 - new_range), 100, 100], dtype=np.uint8)
                upper_range_1 = np.array([(HUE_RED_1 + new_range), 255, 255], dtype=np.uint8)
                upper_range_2 = np.array([(HUE_RED_2 + new_range), 255, 255], dtype=np.uint8)
                threshold_1 = cv2.inRange(self.hsv, lower_range_1, upper_range_1 )
                threshold_2 = cv2.inRange(self.hsv, lower_range_2, upper_range_2 )
                threshold = cv2.bitwise_or(threshold_1, threshold_2, mask=None) # combine both threshold images
            else:
                lower_range, upper_range, color_state = self.brick_hue_range(i)
                threshold = cv2.inRange(self.hsv, lower_range, upper_range ) #Capture threshold of brick within set color range

            # Count total number of white pixels in mask. If it"s above our area limit we found brick
            clean_mask = self.denoise_img(threshold)
            area = np.count_nonzero(clean_mask)
            if area > area_limit:
                
                #lets create an output image merging 2 photos
                output = cv2.bitwise_and(self.raw_img, self.raw_img, mask=clean_mask)
            
                
                # Find contours of brick to get bounding box
                if (int(cv2.__version__[0]) > 3):
                    contours, higherarch = cv2.findContours(clean_mask, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
                else:
                    _ , contours, higherarch = cv2.findContours(clean_mask, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
                
                

                #cv2.drawContours(self.raw_img, contours, -1, (0, 255, 0), 3)

                #contours = np.array(contours)
                #print(contours.shape)

                #contours = np.concatenate(contours).ravel()
                #contours = np.reshape(contours, (len(contours)//2, 2))
                #print(contours.shape)
                # gray = cv2.GaussianBlur( clean_mask, (5, 5), 0 )
                # circles = HoughCircles(gray, HOUGH_GRADIENT, 1.2, 10, param1=100,param2=60,minRadius=0, maxRadius=300)

                # circles = np.uint8(np.around(circles))
                # for j in circles[0,:]:
                #     # draw the outer circle
                #     cv2.circle(self.raw_img,(j[0],j[1]),j[2],(0,0,0),2)
                #     # draw the center of the circle


                #Apptempt to find largest contour
                if len(contours) != 0:
                    c = max(contours, key=cv2.contourArea)
                    x,y,w,h = cv2.boundingRect(c)
                    #cv2.rectangle(self.raw_img,(x,y), (x+w, y+h), (0,255,0),2)
                    # epsilon = 0.02 * cv2.arcLength(c, True)
                    # approximations = cv2.approxPolyDP(c, epsilon, True)
                    # cv2.drawContours(self.raw_img, [approximations], 0, (0,255,0), 3)
                    rect = cv2.minAreaRect(np.array(c))
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    print(box)
                    cv2.drawContours(self.raw_img, [box], 0, (0,0,255), 2)

                    #Logic for creating mask 
                    #corner diag
                    corner_pointx1, corner_pointy1 = box[0]
                    corner_pointx2, corner_pointy2 = box[2]
                    y1 = max(0, corner_pointy1)
                    y2 = max(0, corner_pointy2)
                    x1 = max(0, min(corner_pointx1, corner_pointx2))
                    x2 = max(0, max(corner_pointx1, corner_pointx2))

                    dim = box[2] - box[0]
                    mask = np.zeros_like(self.raw_img)
                    cv2.rectangle(mask, (x1,y1), (x2,y2), (255,255,255), -1)
                    masked_img = cv2.bitwise_and(self.raw_img, mask)
                    
                    grey = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(grey, (5,5),0)
                    #Abstract edges
                    canny = cv2.Canny(blur, 10, 80, apertureSize = 3)
                    cv2.imshow("Canny_frame", canny)
                    #Hough Circles logic
                    circles = HoughCircles(canny, HOUGH_GRADIENT, 1.2, 10, param1=90,param2=50,minRadius=20, maxRadius=100)
                    if circles is not None:
                        circles = np.uint8(np.around(circles))
                        for j in circles[0,:]:
                            # draw the outer circle
                            cv2.circle(self.raw_img,(j[0],j[1]),j[2],(0,0,0),2)
                            # draw the center of the circle
                    
                    #Calculate brick center
                    M = cv2.moments(c)
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    cv2.circle(self.raw_img, (cx, cy), 10, (255, 0, 0), -1)
                    
                    #Logic to get rectangle angle
                    highest_point_index = np.argmin(box, axis=0)[1]
                    highest_point_neighbor_1_index = highest_point_index + 1 if highest_point_index + 1 < len(box) else 0
                    highest_point_neighbor_2_index = highest_point_index - 1 if highest_point_index - 1 > -1 else len(box) - 1
                    highest_point = box[highest_point_index]
                    highest_point_neighbor_1 = box[highest_point_neighbor_1_index]
                    highest_point_neighbor_2 = box[highest_point_neighbor_2_index]
                    first_side = np.linalg.norm(highest_point - highest_point_neighbor_1)
                    second_side = np.linalg.norm(highest_point - highest_point_neighbor_2)
                    
                    if first_side > second_side:
                        brick_angle = np.arctan2(highest_point_neighbor_1[1] - highest_point[1], highest_point_neighbor_1[0] - highest_point[0]) * (180.0/np.pi)
                    else:
                        brick_angle = np.arctan2(highest_point_neighbor_2[1] - highest_point[1], highest_point_neighbor_2[0] - highest_point[0]) * (180.0/np.pi)
                    
                    min_brick_angle = min(brick_angle, abs(180 - brick_angle))

                # for j,cnt in enumerate(contours):
                #     print(j, len(cnt))
                #     #color = ((j * 35) % 255, (j * 50) % 255, 255)
                #     epsilon = 0.01 * cv2.arcLength(cnt, True)
                #     approximations = cv2.approxPolyDP(cnt, epsilon, True)
                #     cv2.drawContours(self.raw_img, [approximations], 0, (0,255,0), 3)
                #     cv2.waitKey(0)
                #     cv2.imshow("Color_frame", self.raw_img)

                
                if DISPLAY_IMAGES:
                    cv2.imshow("Detected Brick", clean_mask)
                    cv2.imshow("Color_frame", self.raw_img)
                    
                if DEBUG:
                    print("{} brick detected with area of {}, center at ({}, {}), rotated by {:.3f}, bounding box of {}."
                    .format(color_state.capitalize(), area, cx, cy, min_brick_angle, [[box[i][0], box[i][1]] for i in range(len(box))]))
            # else:
            #     if DEBUG:
            #         print("No Brick of color: {} found".format(color_state))
        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"): 
            return


if __name__ == "__main__":
    rospy.init_node("brick_detector")
    brick_detector = BrickDetector()
    rospy.spin()
    # cv2.destroyAllWindows()