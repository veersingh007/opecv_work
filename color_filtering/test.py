import cv2 
import numpy as np 

img_path="/home/veersingh/color_filtering/balls-red-blue-yellow-green.jpg"
cap = cv2.VideoCapture(img_path) 

_, frame = cap.read() 
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        

#lower_red = np.array([0,50,50])
lower_red = np.array([-10,100,100])
#upper_red = np.array([10,255,255])
upper_red = np.array([10,255,255])


lower_green = np.array([50,100,100])
upper_green = np.array([70,255,255])

lower_blue = np.array([100,100,100])
upper_blue = np.array([130,255,255])
#print("lower_red=",lower_red)


maskr = cv2.inRange(hsv, lower_red, upper_red) 
red = cv2.bitwise_and(frame,frame, mask= maskr) 

maskg = cv2.inRange(hsv, lower_green, upper_green)
green = cv2.bitwise_and(frame,frame, mask= maskg) 


maskb = cv2.inRange(hsv, lower_blue, upper_blue)

blue = cv2.bitwise_and(frame,frame, mask= maskb) 
cv2.imshow('frame',frame) 

#cv2.imshow('mask',mask) 
cv2.imshow('red',red) 
cv2.imshow('green',green) 
cv2.imshow('blue',blue) 
k = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows() 
cap.release() 
