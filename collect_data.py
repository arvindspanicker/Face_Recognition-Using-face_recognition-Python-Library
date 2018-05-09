import face_recognition
import cv2
import os
import string
import threading
import Queue
import time

#Variables
name = raw_input('Enter Name: ').lower()
camera = cv2.VideoCapture('rtmp://192.168.20.144:1935/flash/12:admin:EYE3inapp')
count = 0

while count <= 1000:
	rval, frame = camera.read()
	cv2.imshow("Video", frame)
	key = cv2.waitKey(20)
	if key == 1048603: # exit on ESC
		break
	if key == 1048586: # Enter button to take picture 
		folder = 'train/images/'+name
		if not os.path.exists(folder):
			original_umask = os.umask(0)
			os.makedirs(folder)
		cv2.imwrite(folder + '/' + name + str(count) + '.jpg',frame)
		count +=1


cv2.destroyWindow('Video')


