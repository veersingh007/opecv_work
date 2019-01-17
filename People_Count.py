import cv2
import numpy as np
import math as m
from time import time
import time as tm
from datetime import datetime
import sys
from datetime import datetime
import json
import traceback
import requests
import re
#import gen_session

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject


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

		self.backGroundModel =  frame * self.alpha + self.backGroundModel * (1 - self.alpha)

		# after the previous operation, the dtype of
		# self.backGroundModel will be changed to a float type
		# therefore we do not pass it to cv2.absdiff directly,
		# instead we acquire a copy of it in the uint8 dtype
		# and pass that to absdiff.

		return cv2.absdiff(self.backGroundModel.astype(np.uint8),frame)


class People_Counter:

    def __init__(self, vid_name, starting_time,end_time, date, camera_id ,dist_threshold = 100, area_threshold = 10000, max_area_threshold = 15000,
        reference_line = 100, ref_line_min_x = 0, ref_line_max_x = 0, ref_line_min_y = 0, ref_line_max_y = 0, width_threshold = 600,
                crop_left = 200, wait_flag = 0, box_colour = (0, 0, 0), centroid_colour = (0,0,255),
                id_colour = (0,255,0), ref_line_colour = (255,255,255), entry_text_colour = (250, 0, 0), 
                exit_text_colour = (0, 0, 255),reference_line_offset=0, title = "2", show_fgbg = 0, draw = True, cam_rotat =0 ):
        self.vid_name = vid_name
        self.dist_threshold = dist_threshold
        self.area_threshold = area_threshold
        self.max_area_threshold = max_area_threshold
        self.reference_line = reference_line
        self.ref_line_min_x = ref_line_min_x
        self.ref_line_max_x = ref_line_max_x
        self.ref_line_min_y = ref_line_min_y
        self.ref_line_max_y = ref_line_max_y
        self.width_threshold = width_threshold
        self.crop_left = crop_left
        # self.vid_start = 500
        self.wait_flag = wait_flag
        self.box_colour = box_colour
        self.centroid_colour = centroid_colour
        self.id_colour = id_colour
        self.ref_line_colour = ref_line_colour
        self.entry_text_colour = entry_text_colour
        self.exit_text_colour = exit_text_colour
        self.EntranceCounter = 0
        self.ExitCounter = 0
        self.prevCentroids = None
        self.frames_processed = 0
        self.title = camera_id#title
        self.show_fgbg = show_fgbg
        self.output_list = []
        self.draw = draw
        self.reference_line_offset = reference_line_offset
        self.starting_time=starting_time
        self.end_time=end_time
        #self.date=datetime.today().strftime('%Y-%m-%d')
        self.date=datetime.today().strftime('%Y-%m-%d %H:%M:%S')

        self.camera_id=camera_id
        self.cam_rotat = cam_rotat
        self.post_data_file= open("post_data_file.txt","a")


    def denoise(self, frame):
        frame = cv2.medianBlur(frame, 25)
        frame = cv2.GaussianBlur(frame, (25, 25), 0)

        return frame

    def line_erode(self,bw):
        horizontal = np.copy(bw)
        vertical = np.copy(bw)

        cols = horizontal.shape[1]
        horizontal_size = int(cols / 25)

        # print("horizontal_size=",horizontal_size)
        # print("cols=",cols)

        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 4))

        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)


        horizontalStructure2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        horizontal = cv2.dilate(horizontal, horizontalStructure2)
        horizontal = cv2.erode(horizontal, horizontalStructure2)
        return horizontal



    def find_contours_img(self, frame):
        area_threshold = 2700#1400#1150
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

    def draw_contours(self,orgframe,f_contours):
        for c in f_contours:
            box_colour=(0,255,0)
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(orgframe, (x, y), (int(x+w),int(y+h)), box_colour, 2)
        # cv2.imshow("draw_contours", orgframe)



    def rotateImage(self,image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result


    def distance(self, a, b):
        return m.sqrt( (a[0] - b[0])**2 + ((a[1] - b[1])/4)**2 )

    '''Find if a person is above or below reference line'''
    def find_region(self, x, y):
        #y2 = self.ref_line_max_y
        #y1 = self.ref_line_min_y
        x2 = self.ref_line_max_x
        x1 = self.ref_line_min_x

        yh = self.ref_line_max_y - self.ref_line_min_y
        y2 = y1 = self.ref_line_min_y + yh/2
        if(y2 == y1):
            return y1-y
        if(x2 == x1):
            return x1-x
        #return (-((y-y1)/(y2-y1)) + ((x-x1)/(x2-x1)))
        if(x<x1 or x>x2 or y<y1 or y>y2):
            return 99 
        return (-((y-y1)/(y2-y1)) + ((x-x1)/(x2-x1)))


    '''CAll this function'''
    def read_video(self):
        fgbg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=10, detectShadows=True)
        cap = cv2.VideoCapture(self.vid_name)

        c=0
        while c<5:
            cap.read()
            c+=1
        dis = 0

        prev_time = time()
        last_time = 0
        count_no_frame=0
        count = 0
        reinitialize = 0
        pflag = False
        i=0

        wt = 1#100
        pause = False
        first_frame = False
        W = None
        H = None
        ct = CentroidTracker(maxDisappeared=1, maxDistance=1200)
        trackableObjects = {}
        totalDown = 0
        totalUp = 0

        fp_count=0
        skip_frames=115
        #skip_frames=75


        #out = cv2.VideoWriter('./output.avi', -1, 20.0, (640,480))

        #size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
        #out = cv2.VideoWriter('./001_output.mp4',fourcc, 29.0, size, False)


        file_count=0
        print("Starting infinite loop and continue to read frames")
        while True:
            # print("hello")
            #print("fil=",file_count%1000)
            if(0 == file_count%1000):
                #print("check file")
                #lines
                print("Check if we have data to post")
                file_fd=open('post_data_file.txt','r')
                lines=file_fd.readlines()
                #print("lines=",lines)
                for line in lines:
                    #print("line=",line)
                    headers = {'content-type': "application/json",'cache-control': "no-cache"}
                    url = "http://app.gizmosmart.io/iot/1.3/public/peopleCounting/token/51987536541"
                    try:
                        arr = line
                        response = requests.request("POST", url, data=arr, headers=headers)
                        #print("in file post=",response.text)
                    except Exception as e:
                        #print ("Post Error: ",str(e)) 
                        print("Failed to post data: %s",str(e))
                        #self.post_data_file.write(arr+"\n")
                        #continue
                #print("truncate the file")
                #file_fd.truncate(0)
                with open('post_data_file.txt','a+') as fil:
                    fil.truncate(0)
                file_count=0

                        
            file_count+=1
            key = cv2.waitKey(wt) & 0xFF
            if (key == ord('p')):
                pause = not pause
            if (key == ord('s')):
                wt = int(wt * 2)
            if (key == ord('f')):
                wt = int(wt / 2)
            if(key == ord('q')):
                break

            if (pause):
                continue
            ret, frame = cap.read()

            #out.write(frame)

            if not ret:
                print("ret is null, could not open camera")
                tm.sleep(1)
                #continue

            fp_count +=1

            #if fp_count<skip_frames:
            #    continue
            #if(fp_count%3):
            #    continue

            if(self.cam_rotat):
                frame=self.rotateImage(frame, 90)

            #cv2.imshow("framee",frame)
            #continue


            if ret is True:
                # Show the filtered image

                h, w, c = frame.shape
                #print("shape=",w,h,c)
                fram = cv2.resize(frame, (int(w / 2), int(h / 2)))
                if self.draw:
                    cv2.imshow("original fram", fram)

            if not ret:
                count += 1
                print("ret is zero,count=",count)
                        #iscamopened=cap.isOpened()
                        #print("cap opened=",iscamopened)
                        #if(not iscamopened):
                        #    break
                if count >= 5:
                    print("frame reached max unread frame 5")
                            #reinitialize camera
                    print("release the camera",self.camera_id)
                    cap.release()
                    reinitialize += 1
                    if reinitialize <= 4000:
                        #cam_url=gen_session.get_cam_url(self.camera_id)
                        cam_url=self.vid_name
                        print(reinitialize," attempt to reopen camera",cam_url)
                        cap = cv2.VideoCapture(cam_url)
                        if(cap.isOpened()):
                            count=0
                            reinitialize = 0
                            continue
                        else:
                            print("camera could not be opned")
                                    #count=0
                            continue
                            #break
                    else:
                        print("reached to max 4000 reinitialized camera attempt")
                        break
                else:
                    continue

            cy = 350
            cx = 50
            ch = 380
            cw = 200


            cx = self.ref_line_min_x
            cw = self.ref_line_max_x - cx
            cy=self.ref_line_min_y
            ch=self.ref_line_max_y-cy



            #cy = 100
            #cx = 260
            #ch = 180
            #cw = 200

            # cy = 150
            # cx=200
            # ch=250
            # cw=220
            crop_img = fram[cy:cy + ch, cx:cx + cw]
            if W is None or H is None:
                (H, W) = crop_img.shape[:2]
            if not first_frame:
                #backSubtractor = BackGroundSubtractor(0.007, self.denoise(crop_img))
                backSubtractor = BackGroundSubtractor(0.007, self.denoise(crop_img))
                run = True
                first_frame = True
                continue

            # cv2.imshow("crop_img",crop_img)

            crop_img_dn = self.denoise(crop_img)
            # # cv2.imshow('input',denoise(fram))
            # cv2.imshow('input', crop_img_dn)
            #
            # # bgsimg=bgsobj.get_fg_image(denoise_img)
            # # cv2.imshow("bgsimg",bgsimg)
            foreGround = backSubtractor.getForeground(crop_img_dn)

            # ret, foreGroundb = cv2.threshold(foreGround, 35, 255, 0)
            foreGroundb = cv2.cvtColor(foreGround, cv2.COLOR_BGR2GRAY)
            # Apply thresholding on the background and display the resulting mask
            #ret, mask = cv2.threshold(foreGroundb, 38, 255, cv2.THRESH_BINARY)
            ret, mask = cv2.threshold(foreGroundb, 25, 255, cv2.THRESH_BINARY)
            cv2.imshow("mask",mask)
            h, w = mask.shape

            mask = self.line_erode(mask)
            f_contours = self.find_contours_img(mask)
            self.draw_contours(crop_img.copy(), f_contours)

            #if self.show_fgbg == 1:
            #    cv2.imshow('mask', mask)

            cv2.line(crop_img, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

            rects = []
            count += 1
            status = "Waiting"

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
                    # for c in to.centroids:
                    #	print("c====",c)
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    dr_last_point = None
                    centroid_len = len(to.centroids)
                    if (centroid_len > 80):
                        del to.centroids[:(centroid_len - 80)]
                    for cc in to.centroids:
                        # print("cc=",cc)
                        cx = cc[0]
                        cy = cc[1]
                        if (dr_last_point == None):
                            dr_last_point = (cx, cy)
                            continue
                        # cx=cc[0]
                        # cy=cc[1]
                        # cv2.circle(crop_img, (cx,cy), 1, (0,0,255), thickness=2, lineType=8, shift=1)
                        cv2.line(crop_img, (dr_last_point), (cx, cy), (0, 255, 0), thickness=1, lineType=8)
                        dr_last_point = (cx, cy)

                    if not to.counted:
                        # print("centroid[1]=", centroid[1])
                        # print("H/2=", H / 2)
                        if direction < 0 and centroid[1] < H // 2:
                            up_dir_dist = H / 2 - centroid[1]
                            # print(H/4, " up_dir_dist=", up_dir_dist)
                            if (up_dir_dist > 0 and up_dir_dist < H / 4):
                            #if (up_dir_dist > 0 and up_dir_dist < H / 6):
                                # print("caputred up")
                                totalUp += 1
                                # count_in += 1
                                to.counted = True
                        elif direction > 0 and centroid[1] > H // 2:
                            down_dir_dist = centroid[1] - H / 2
                            # print(H/4, " down_dir_dist=", down_dir_dist)
                            if (down_dir_dist > 0 and down_dir_dist < H / 4):
                            #if (down_dir_dist > 0 and down_dir_dist < H / 6):
                                # print("caputured down")
                                totalDown += 1
                                # count_out += 1
                                to.counted = True

                trackableObjects[objectID] = to
                text = "ID {} {}".format(objectID, to.counted)

                cv2.putText(crop_img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
                cv2.circle(crop_img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                # print("to=", to)

            info = [
                #("Up", totalUp),
                #("Down", totalDown),
                ("IN", totalUp),
                ("OUT", totalDown),
                #("Status", status),
            ]

            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                # cv2.putText(crop_img, text, (10, H - ((i * 20) + 20)),
                # 			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(crop_img, text, (10, i * 20 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            if self.draw:
                cv2.imshow("pc roi", crop_img)
            key = cv2.waitKey(1) & 0xFF

            #post the in/out status to server
            cur_time = time() - prev_time

            if (cur_time > 10 and cur_time - last_time > 10):
            #if (cur_time > 100000 and cur_time - last_time > 100000):
                last_time = cur_time
                cur_machine_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
                count_IN = totalUp
                count_OUT = totalDown

                arr = self.json_prepare(self.camera_id, str(count_IN), str(count_OUT),
                                    cur_machine_time)  # +" "+str("10:00:00"))
                print("arr=",arr)
                #print("arr=",json.dumps(arr))

                #url = "http://sales.gizmohelp.com/mcrm/2.1-test/people_counting"
                #arr = {'camera_id': '559475', 'created_date': '2018-10-03 10:00:00', 'people_out': '0', 'people_in': '0'}
            # arr = "{\"camera_id\":\"528365\",\"created_date\":\"2018-10-03 10:00:00\",\"people_in\":\"1\",\"people_out\":\"1\"}"
                headers = {'content-type': "application/json",'cache-control': "no-cache"}
                #print("post data=",json.dumps(arr))

                # response = requests.request("POST", url, data=json.dumps(arr), headers=headers)
                #
                # print("response.txt=",response.text)



                url = "http://app.gizmosmart.ios/iot/1.3/public/peopleCounting/token/51987536541"
                #arr = {"cameraId": 550113 , "data": [{"countIn": "1", "countOut": "2","date": "2018-11-13", "time": "03:00:01" }]}
                headers = {'content-type': "application/json",'cache-control': "no-cache"}
                #print("post data=",json.dumps(arr))
                try:
                    print("post count status:")
                    #response = requests.request("POST", url, data=json.dumps(arr), headers=headers)
                    response = requests.request("POST", url, data=arr, headers=headers)
                    print("response=",response.text)
                except Exception as e:
                    print ("Post Error: ",str(e)) 
                    self.post_data_file.write(arr+"\n")
                    continue
                    #continue


             #   response = requests.request("POST", url, data=arr, headers=headers)
                
             #   print("response.txt=",response.text)
            #if (cur_time > 10 and cur_time - last_time > 10):


                totalUp=0
                totalDown=0



        print("Processing Complete")
        print("Entrances: ", self.EntranceCounter)
        print(self.camera_id," Exits:", self.ExitCounter) 
        cap.release()
        #out.release()
        

    '''Process the fetched frame to count people'''
    def process_frame(self, img, fgbg):

        id = 0
        ''' BG Subtract '''
        frame = fgbg.apply(img)
        ret, frame = cv2.threshold(frame, 135, 255, 0)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #frame = cv2.erode(frame, ker, iterations = 4)

        ''' Erosion Dilation '''
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        frame = cv2.dilate(frame, ker, iterations = 4)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        frame = cv2.erode(frame, ker, iterations = 2)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        frame = cv2.dilate(frame, ker, iterations = 2)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        frame = cv2.erode(frame, ker, iterations = 4)

        ''' Contour Detection '''
        _frame, contours, hierarchy = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        centroids = []

        ''' Draw Contours '''
        if self.draw:
            for c in range(len(contours)):
                if cv2.contourArea(contours[c]) > self.area_threshold:
                    cv2.drawContours(img, contours, c, (0,255,0), 1)

        ''' Process Contours '''
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)

            if cv2.contourArea(c) < self.area_threshold:
                continue
            
            if  self.wait_flag == 1:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break

            z = self.find_region((x+w/2), (y+h/2))
            #print("zzzzz=",z)
            if(z==99):
                print("z==99 so continueing")
                continue
            print("z!=99 so not continueing")            

            if z <=0:
                region = 1
            else:
                region = 2
            if self.draw:
                cv2.rectangle(img, (x, y), (int(x+w),int(y+h)), self.box_colour, 2)
                cv2.circle(img, (int(x+w/2),int(y+h/2)), 5, self.centroid_colour, thickness=-1, lineType=8, shift=0)
                cv2.circle(img, (int(x),int(y)), 5, (0,0,255), thickness=-1, lineType=8, shift=0)
            centroids.append([int(x+w/2),int(y+h/2), region, False])        
            id = id + 1

        currCentroids = centroids

        if self.prevCentroids is None:
            self.prevCentroids = currCentroids
        else:
            for j in currCentroids:
                # print("jjjjj=",j)
                for k in self.prevCentroids:
                    # print("kkkkk=",k)
                    # print("distance=",self.distance(j,k))
                    if self.distance(j, k) < self.dist_threshold:
                        j[3] = k[3]
                        if not j[3]:
                            if j[2] == 1 and k[2] == 2:
                                print("j2=",j[2])
                                print("k2=",k[2])
                                print("entrance counter=",self.EntranceCounter)
                                j[3] = True
                                #self.EntranceCounter += 1
                                self.ExitCounter += 1
                            elif j[2] == 2 and k[2] == 1:
                                print("j2=",j[2])
                                print("k2=",k[2])
                                print("exit counter=",self.ExitCounter)
                                j[3] = True
                                self.EntranceCounter += 1
                                #self.ExitCounter += 1
                        cv2.putText(img, "ID: {}".format(str(j[3])), (j[0],j[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.id_colour, 2)
            self.prevCentroids = currCentroids
        if self.draw:
            cv2.putText(img, "Entrances: {}".format(str(self.EntranceCounter)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.entry_text_colour, 2)
            cv2.putText(img, "Exits: {}".format(str(self.ExitCounter)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.exit_text_colour, 2)
            cv2.line(img, (self.ref_line_min_x, self.ref_line_min_y), (self.ref_line_max_x,self.ref_line_max_y), self.ref_line_colour, 3)


        return img, frame

    '''Prepare JSON to be sent to server'''
    def json_prepare(self,cam_id, count_IN, count_OUT, time):#, date):
        dict = {}
        data_list = []
        dict_data = {}
        dict_data["countIn"] = count_IN
        dict_data["countOut"] = count_OUT
        date,time = re.split(' ',time)        
        dict_data["date"] = date
        dict_data["time"] = time
        data_list.append(dict_data)
        dict["cameraId"] = int(cam_id)
        dict["data"] = data_list
        json_data = json.dumps(dict)
        return json_data


#    def json_prepare(self,cam_id, count_IN, count_OUT, time):
#        dict_data = {}
#        dict_data["people_in"] = count_IN
#        dict_data["people_out"] = count_OUT
#        dict_data["created_date"] = time
#        dict_data["camera_id"] = cam_id
#        return dict_data
