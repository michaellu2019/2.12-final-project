import cv2
import os
import numpy as np
#This File will caputre Hue Saturation and Value from picture

#This function gets mouse position
def show_distance(event, x, y, args, params):
    global point
    point = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN: #if you click with left mouse show the HSV
        print("original BGR: ", cur_img[point[1], point[0]])
        print("HSV val: ", hsv[point[1], point[0]])


#This function reads every image from the folder
def load_img_from_folder(folder_path):
    pictures = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            pictures.append(img)
    return pictures

# Set up window
point = (0,0) # start point with arbitrary value
file_path = '/home/cvdarbeloff/Documents/2120/2.12-final-project/src/brick_pictures' ; # folder to read photos from
images = load_img_from_folder(file_path)
cur_img = images[8]
hsv = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV) # converts photo from RGB to HSV
cv2.namedWindow('Color_Frame')
cv2.namedWindow('Hue_Frame')
cv2.setMouseCallback("Color_Frame", show_distance)
#cv2.setMouseCallback("Hue_Frame", show_distance)
## ok cool
while True:
    
    
    cur_hsv = hsv[point[1], point[0]]
    #cv2.putText(cur_img, "{} HSV val".format(cur_hsv), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1,  (0, 0, 0), 1)
    #print('HSV is: ', cur_hsv)
    cv2.imshow('Color_Frame', cur_img)
    cv2.imshow('Hue_Frame', hsv)
    key = cv2.waitKey(1)
    if key == 27 or key == ord("q"):
        cv2.destroyAllWindows()
        break
    



cv2.waitKey(0)
