#https://gist.github.com/TheSalarKhan/7c3d01ad13b0e7e5985a

import numpy as np
import cv2

class BGSMog:
	def __init__(self):
		self.fgbg = cv2.createBackgroundSubtractorMOG2(history=12011, varThreshold=10, detectShadows=False)
	def get_fg_image(self,frame):
		frmbg = self.fgbg.apply(frame)
		return frmbg

class BackGroundSubtractor:
	# When constructing background subtractor, we
	# take in two arguments:
	# 1) alpha: The background learning factor, its value should
	# be between 0 and 1. The higher the value, the more quickly
	# your program learns the changes in the background. Therefore, 
	# for a static background use a lower value, like 0.001. But if 
	# your background has moving trees and stuff, use a higher value,
	# maybe start with 0.01.
	# 2) firstFrame: This is the first frame from the video/webcam.
	def __init__(self,alpha,firstFrame):
		self.alpha  = alpha
		self.backGroundModel = firstFrame

	def getForeground(self,frame):
		# apply the background averaging formula:
		# NEW_BACKGROUND = CURRENT_FRAME * ALPHA + OLD_BACKGROUND * (1 - APLHA)

#		self.backGroundModel =  frame * self.alpha + self.backGroundModel * (1 - self.alpha)

		# after the previous operation, the dtype of
		# self.backGroundModel will be changed to a float type
		# therefore we do not pass it to cv2.absdiff directly,
		# instead we acquire a copy of it in the uint8 dtype
		# and pass that to absdiff.

		return cv2.absdiff(self.backGroundModel.astype(np.uint8),frame)


def find_contours_img(frame,orgframe):
        area_threshold = 1150
        _frame, contours, hierarchy = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        final_contours=[]
        for c in contours:
            if cv2.contourArea(c) < area_threshold:
                continue
            #print("area=",cv2.contourArea(c))

            final_contours.append(c)
        #    (x, y, w, h) = cv2.boundingRect(c)
            #print("inside c=",c)
        #    box_colour = (0,255,0)
        #    cv2.rectangle(orgframe, (x, y), (int(x+w),int(y+h)), box_colour, 2)
        #cv2.imshow("framecont",orgframe)
        return final_contours
def draw_contours(orgframe,f_contours):
    for c in f_contours:
        box_colour=(0,255,0)
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(orgframe, (x, y), (int(x+w),int(y+h)), box_colour, 2)
    cv2.imshow("draw_contours", orgframe)

video_file="/home/veersingh/sprint_8/people-counting-opencv/pc_video.mp4"
#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(video_file)

# Just a simple function to perform
# some filtering before any further processing.
def denoise(frame):
    frame = cv2.medianBlur(frame,5)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    
    return frame

ret,frame = cam.read()
if ret is True:
	backSubtractor = BackGroundSubtractor(0.001,denoise(frame))
	#bgsobj=BGSMog()
	run = True
else:
	run = False

while(run):
	# Read a frame from the camera
	ret,frame = cam.read()

	# If the frame was properly read.
	if ret is True:
		# Show the filtered image


		h,w,c=frame.shape
		#print("shape=",w,h,c)
		fram=cv2.resize(frame,(int(w/2),int(h/2)))

		denoise_img=denoise(fram)
		#cv2.imshow('input',denoise(fram))
		cv2.imshow('input',denoise_img)

		#bgsimg=bgsobj.get_fg_image(denoise_img)
		#cv2.imshow("bgsimg",bgsimg)

		# get the foreground
		foreGround = backSubtractor.getForeground(denoise(frame))

		#ret, foreGroundb = cv2.threshold(foreGround, 35, 255, 0)
		foreGroundb = cv2.cvtColor(foreGround, cv2.COLOR_BGR2GRAY)
		# Apply thresholding on the background and display the resulting mask
		ret, mask = cv2.threshold(foreGroundb, 55, 255, cv2.THRESH_BINARY)

		# Note: The mask is displayed as a RGB image, you can
		# display a grayscale image by converting 'foreGround' to
		# a grayscale before applying the threshold.


		h,w=mask.shape
		#print("shape=",w,h,c)
		mask=cv2.resize(mask,(int(w/2),int(h/2)))

		f_contours=find_contours_img(mask,fram)
		draw_contours(fram,f_contours)
		cv2.imshow('mask',mask)

		key = cv2.waitKey(10) & 0xFF
	else:
		break

	if key == 27:
		break

cam.release()
cv2.destroyAllWindows()
