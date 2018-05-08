import math
from sklearn import neighbors
import os
import os.path
import cv2
import pickle
import face_recognition



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

camera = cv2.VideoCapture('rtmp://192.168.20.144:1935/flash/12:admin:EYE3inapp')

def predict(image_np, knn_clf=None, model_path=None, distance_threshold=0.6):
	if knn_clf is None and model_path is None:
		raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
	if knn_clf is None:
		with open(model_path, 'rb') as f:
			knn_clf = pickle.load(f)
	face_locations = face_recognition.face_locations(image_np)
	if len(face_locations) == 0:
		return []
	faces_encodings = face_recognition.face_encodings(image_np, known_face_locations=face_locations)
	closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
	are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]
	return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), face_locations, are_matches)]


def show_prediction_labels_on_image(image_np, predictions):
	for name, (top, right, bottom, left) in predictions:
		cv2.rectangle(image_np,(left, top), (right, bottom),(0, 0, 255),3)
		name = name.encode("UTF-8")
		cv2.putText(image_np, name ,(left +6 ,bottom - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),1,cv2.CV_AA)
	cv2.imshow('Window',image_np)
	key = cv2.waitKey(20)
	if key == 1048603:
		exit(0)


if __name__ == "__main__":
	while True:
		ret, frame = camera.read()
		predictions = predict(frame, model_path="trained_knn_model.clf")
	    	for name, (top, right, bottom, left) in predictions:
		    print("- Found {} at ({}, {})".format(name, left, top))
		show_prediction_labels_on_image(frame, predictions) 
