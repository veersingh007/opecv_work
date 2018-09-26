import numpy as np
import cv2
import dlib
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from time import time
from time import clock
from datetime import datetime
import requests
import json


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


def find_contours_img(frame):
        area_threshold = 1400#1150
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
#video_file="http://ca006.cameramanager.com/stream/jpeg/recording?user_id=48382&session_id=7d376f0c37093839cbef2cbe74ba0ea0&camera_id=559475"
#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(video_file)

# Just a simple function to perform
# some filtering before any further processing.

def line_erode(bw):
    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    cols = horizontal.shape[1]
    horizontal_size = int(cols / 25)

    print("horizontal_size=",horizontal_size)
    print("cols=",cols)

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 4))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)


    horizontalStructure2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    horizontal = cv2.dilate(horizontal, horizontalStructure2)
    horizontal = cv2.erode(horizontal, horizontalStructure2)
    return horizontal



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


def json_prepare(cam_id, count_IN, count_OUT, time):
	dict_data = {}
	dict_data["people_in"] = count_IN
	dict_data["people_out"] = count_OUT
	dict_data["created_date"] = time
	dict_data["camera_id"] = cam_id
	return dict_data


# ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
ct = CentroidTracker(maxDisappeared=1, maxDistance=30)

trackableObjects = {}
count=0
trackers = []
totalDown = 0
totalUp = 0
W = None
H = None
first_frame=False
pause = False
wt=100
prev_time=None
count_in=0
count_out=0
start=time()
while(run):
	# Read a frame from the camera
	#cv2.waitKey()
	key = cv2.waitKey(wt) & 0xFF
	if(key==ord('p')):
		pause =not pause
	if(key==ord('s')):
		wt = int(wt*2)
	if(key==ord('f')):
		wt = int(wt/2)

	if(pause):
		continue
	ret,frame = cam.read()

	cur_time = time()
	if not prev_time:
		prev_time = cur_time
	# If the frame was properly read.
	if ret is True:
		# Show the filtered image

		h,w,c=frame.shape
		#print("shape=",w,h,c)
		fram=cv2.resize(frame,(int(w/2),int(h/2)))
		cv2.imshow("fram",fram)


		# cy = 80
		# cx=160
		# ch=220
		# cw=320

		cy = 100
		cx=260
		ch=180
		cw=200

		# cy = 150
		# cx=200
		# ch=250
		# cw=220
		crop_img = fram[cy:cy + ch, cx:cx + cw]
		if W is None or H is None:
			(H, W) = crop_img.shape[:2]
		if not first_frame:
			backSubtractor = BackGroundSubtractor(0.001, denoise(crop_img))
			run = True
			first_frame=True
			continue


		#cv2.imshow("crop_img",crop_img)

		crop_img_dn=denoise(crop_img)
		#cv2.imshow('input',denoise(fram))
		cv2.imshow('input',crop_img_dn)

		#bgsimg=bgsobj.get_fg_image(denoise_img)
		#cv2.imshow("bgsimg",bgsimg)

		# get the foreground
		foreGround = backSubtractor.getForeground(crop_img_dn)

		#ret, foreGroundb = cv2.threshold(foreGround, 35, 255, 0)
		foreGroundb = cv2.cvtColor(foreGround, cv2.COLOR_BGR2GRAY)
		# Apply thresholding on the background and display the resulting mask
		ret, mask = cv2.threshold(foreGroundb, 40, 255, cv2.THRESH_BINARY)

		# Note: The mask is displayed as a RGB image, you can
		# display a grayscale image by converting 'foreGround' to
		# a grayscale before applying the threshold.


		h,w=mask.shape
		#print("shape=",w,h,c)
		#mask=cv2.resize(mask,(int(w/2),int(h/2)))

		mask = line_erode(mask)
		f_contours=find_contours_img(mask)
		draw_contours(crop_img.copy(),f_contours)

		cv2.imshow('mask',mask)
		cv2.line(crop_img, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

		rects = []
		count += 1
		status = "Waiting"
		# if(count%5 == 0):
		# 	print("if count=",count)
		# 	status = "Detecting"
		# 	trackers = []
		#
		# 	for c in f_contours:
		# 		box_colour=(0,255,0)
		# 		(x, y, w, h) = cv2.boundingRect(c)
		# 		print(" x=", x, " y=", y, " w=", w, " h=", h)
		# 		(startX, startY, endX, endY) = (x,y,x+w,y+h)
        #
		# 		tracker = dlib.correlation_tracker()
		# 		rect = dlib.rectangle(startX, startY, endX, endY)
		# 		tracker.start_track(crop_img, rect)
		# 		trackers.append(tracker)
        #
		# 		cv2.rectangle(crop_img, (x, y), (int(x+w),int(y+h)), box_colour, 2)
		# 		#cv2.imshow("draw_contours", orgframe)
		# else:
		# 	print("else count=",count)
		# 	for tracker in trackers:
		# 		print("in else tracker")
		# 		status = "Tracking"
		# 		tracker.update(crop_img)
		# 		pos = tracker.get_position()
		# 		startX = int(pos.left())
		# 		startY = int(pos.top())
		# 		endX = int(pos.right())
		# 		endY = int(pos.bottom())
		# 		print("startx=",startX," startY=",startY, " endX=", endX, " endY=",endY)
		# 		rects.append((startX, startY, endX, endY))
		# 		cv2.rectangle(crop_img, (startX, startY), (int(endX),int(endY)), box_colour, 2)

		for c in f_contours:
			box_colour = (0, 255, 0)
			(x, y, w, h) = cv2.boundingRect(c)
			(startX, startY, endX, endY) = (x, y, x + w, y + h)
			rects.append((startX, startY, endX, endY))
			cv2.rectangle(crop_img, (startX, startY), (int(endX), int(endY)), box_colour, 2)


		objects = ct.update(rects)
		for (objectID, centroid) in objects.items():
			to = trackableObjects.get(objectID, None)
			if to is None:
				to = TrackableObject(objectID, centroid)
			
			else:
				#for c in to.centroids:
				#	print("c====",c)
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				print("to.counted=",to.counted)
				print("direction",direction)
				# print("centroid[1]=",centroid)
				# print("centroids=",to.centroids)
				#if not to.counted:
				#for cc in to.centroids:
				#	print("cc=",cc)
				#	cx=cc[0]
				#	cy=cc[1]
				#	cv2.circle(denoise_img, (cx,cy), 1, (0,0,255), thickness=2, lineType=8, shift=1)

				dr_last_point = None
				centroid_len=len(to.centroids)
				if(centroid_len>40):
					del to.centroids[:(centroid_len-40)]
				for cc in to.centroids:
					# print("cc=",cc)
					cx = cc[0]
					cy = cc[1]
					if(dr_last_point == None):
						dr_last_point = (cx,cy)
						continue
					# cx=cc[0]
					# cy=cc[1]
					# cv2.circle(crop_img, (cx,cy), 1, (0,0,255), thickness=2, lineType=8, shift=1)
					cv2.line(crop_img, (dr_last_point), (cx,cy), (0, 255, 0), thickness=1, lineType=8)
					dr_last_point = (cx, cy)

				if not to.counted:
					print("centroid[1]=",centroid[1])
					print("H/2=",H/2)
					if direction < 0 and centroid[1] < H // 2:
						up_dir_dist = H/2-centroid[1]
						print("up_dir_dist=",up_dir_dist)
						if(up_dir_dist>0 and up_dir_dist <H/4):
							totalUp += 1
							# count_in += 1
							to.counted = True
					elif direction > 0 and centroid[1] > H // 2:
						down_dir_dist = centroid[1] -H/2
						print("down_dir_dist=",down_dir_dist)
						if(down_dir_dist>0 and down_dir_dist<H/4):
							totalDown += 1
							# count_out += 1
							to.counted = True

			trackableObjects[objectID] = to
			text = "ID {} {}".format(objectID, to.counted)

			cv2.putText(crop_img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
			cv2.circle(crop_img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)




			print("to=",to)

		info = [
			("Up", totalUp),
			("Down", totalDown),
			("Status", status),
		]

		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			# cv2.putText(crop_img, text, (10, H - ((i * 20) + 20)),
			# 			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			cv2.putText(crop_img, text, (10,i*20+20),
						cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


		# if (True):
		# 	in_out = "{}: {}".format("count in", str(count_in))
		# 	cv2.putText(crop_img, in_out, (120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        #
		# 	in_out = "{}: {}".format("count out", str(count_out))
		# 	cv2.putText(crop_img, in_out, (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        #
		# 	p_time = int(time()-start)
		# 	time_str = "{}: {}".format("time", str(p_time))
		# 	cv2.putText(crop_img, time_str, (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        #
		# if(cur_time - prev_time >= 10):
        #
		# 	prev_time = cur_time
		# 	start = time()
        #
		# 	cur_machine_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
		# 	camera_id="559475"
		# 	arr = json_prepare(camera_id, str(count_in), str(count_out), cur_machine_time)  # +" "+str("10:00:00"))
        #
		# 	url = "http://sales.gizmohelp.com/mcrm/2.1-test/people_counting"
		# 	headers = {'content-type': "application/json", 'cache-control': "no-cache"}
        #
		# 	# response = requests.request("POST", url, data=json.dumps(arr), headers=headers)
		# 	# print("response=",response.text)
		# 	count_in=0
		# 	count_out=0

		cv2.imshow("new denoise_img",crop_img)
		key = cv2.waitKey(1) & 0xFF


	else:
		break

	if key == 27:
		break

cam.release()
cv2.destroyAllWindows()
