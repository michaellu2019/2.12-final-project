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
HUE_YELLOW = 30
area_limit = 50000 # if sum is less than 50,000 pixels we can assume we do not see a full brick. TODO find correct area estimate
color_state = '' # Start off with empty color state

#This function reads every image from the folder
#TODO: Delete when switching to live camera
def load_img_from_folder(folder_path):
    pictures = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            pictures.append(img)
    return pictures

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
cur_img = images[1]

#NOTE if we want to merge mask with original image we can use result = bitwise_and(cur_img, clean_mask, mask = NONE)
detected_lines = False
while True:
    if not detected_lines:
        cv2.imshow('Color_Frame', cur_img)
        gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray', gray)
        lower_threshold = 10
        upper_threshold = 100
        edges = cv2.Canny(gray,lower_threshold,upper_threshold,apertureSize = 3)
        cv2.imshow("Canny_Image", edges)
        cv2.waitKey(3)
        min_intersections = 200
        lines = cv2.HoughLines(edges,1,np.pi/180,min_intersections) 
        num_lines = 0
        shape = lines.shape
        print(shape)
        if shape is not None:
            for i in range(shape[0]):                         # Plot lines over original feed
                for rho,theta in lines[i]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(cur_img,(x1,y1),(x2,y2),(0,0,255),2)
                    num_lines += 1
        cv2.imshow("Line_Detected_Image", cur_img)
        cv2.waitKey(5)
        print("Detecting Lines...")
        detected_lines = True
        
    
    key = cv2.waitKey(1)
    if key == 27 or key == ord("q"): 
        break

cv2.destroyAllWindows()