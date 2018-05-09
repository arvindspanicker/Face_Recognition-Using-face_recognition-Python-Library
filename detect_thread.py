import math
from sklearn import neighbors
import os
import os.path
import cv2
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import Queue
import threading



''' Not Tested Yet '''

camera = cv2.VideoCapture('rtmp://192.168.20.144:1935/flash/12:admin:EYE3inapp')

def predict(knn_clf=None, model_path="trained_knn_model.clf", distance_threshold=0.6):	
	if knn_clf is None and model_path is None:
		raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
	if knn_clf is None:
		with open(model_path, 'rb') as f:
			knn_clf = pickle.load(f)
	frame_process = True
	while True:
		if not inputQueue1.empty():
			if frame_process == True:
				image_np = inputQueue1.get()
				image_np = image_np[:, :, ::-1]
				face_locations = face_recognition.face_locations(image_np)
				if len(face_locations) == 0:
					predictionQueue.put([])
					continue
				faces_encodings = face_recognition.face_encodings(image_np, known_face_locations=face_locations)
				closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
				are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]
				predictionQueue.put([(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), face_locations, are_matches)])
			frame_process = not frame_process


def show_prediction_labels_on_image():
	if not predictionQueue.empty():
		predictions = predictionQueue.get()
		image_np = inputQueue2.get()
		for name, (top, right, bottom, left) in predictions:
			cv2.rectangle(image_np,(left, top), (right, bottom),(0, 0, 255),3)
			name = name.encode("UTF-8")
			cv2.putText(image_np, name ,(left +6 ,bottom - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),1,cv2.CV_AA)
		for name, (top, right, bottom, left) in predictions:
		    print("- Found {} at ({}, {})".format(name, left, top))
		cv2.imshow('Window',image_np)
		key = cv2.waitKey(20)
		if key == 1048603: #esc key to quit
			exit(0)
		

def get_image():
	print 'Video Stream Started'
	get_frame = True
	while True:
		if get_frame:
			ret, frame = camera.read()
			inputQueue1.put(frame)
			inputQueue2.put(frame)
		get_frame = not get_frame
		


if __name__ == "__main__":
	global frame_process 
	frame_process = True
	global inputQueue1 
	inputQueue1 = Queue.Queue()
	global inputQueue2 
	inputQueue2 = Queue.Queue()
	global predictionQueue 
	predictionQueue = Queue.Queue()
	global ALLOWED_EXTENSIONS 
	ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

	predict_thread = []
	for i in range (0,1):
		t = threading.Thread(target = predict, args = ())
		predict_thread.append(t)

	get_feed_thread = threading.Thread(target = get_image, args = ())

	for i in range (0,1):
		predict_thread[i].daemon = True
 
	get_feed_thread.daemon = True

	for i in range (0,1):
		predict_thread[i].start()

	get_feed_thread.start()

	while True:
		show_prediction_labels_on_image() 
