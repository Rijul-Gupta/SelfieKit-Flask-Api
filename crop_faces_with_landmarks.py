from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
from pathlib import Path
import json
import random
import dlib
import time


def crop_face_from_landmarks(image, landmarks):

	padding = 0.05;
	bbox_min = np.min(landmarks, axis=0)
	bbox_max = np.max(landmarks, axis=0)
	crop_width = bbox_max[0] - bbox_min[0]
	crop_height = bbox_max[1] - bbox_min[1]
	
	bbox_min[0] = bbox_min[0] - int(crop_width*padding)
	bbox_min[1] = bbox_min[1] - int(crop_height*padding)
	
	bbox_max[0] = bbox_max[0] + int(crop_width*padding)
	bbox_max[1] = bbox_max[1] + int(crop_height*padding)
	
	bbox_has_oob = False;
	for n in bbox_min:
		if( n < 0):
			bbox_has_oob = True;
	for n in bbox_max:
		if( n < 0):
			bbox_has_oob = True;
	if(bbox_has_oob == True):
		print('bbox is out of bounds, skipping')
		continue
	
	if(min(crop_width, crop_height) < crop_size_thresh):
		print("crop is too small, skipping")
		continue
		
	crop = image[bbox_min[1]:bbox_max[1],bbox_min[0]:bbox_max[0]]'
	return crop
	

def reshape_for_polyline(array):
    """Reshape image so that it works with polyline."""
    return np.array(array, np.int32).reshape((-1, 1, 2))

json_save_name = 'all_landmarks.json'
landmarks_dict = {}
if(os.path.exists(json_save_name)):
	with open(json_save_name, 'r') as fp:
		landmarks_dict = json.load(fp)
		

dec_time = time.time()

padding = 0.05;
crop_size_thresh = 128;
for key, value in landmarks_dict.items():
	image = cv2.imread(key)
	for i, landmarks in enumerate(value):
		print(landmarks)
		
		file_base, file_ext = os.path.splitext(key)
		
		crop_save_name = "crops/" + file_base.replace("faces/", "").replace("/", "_") + '_' + str(i) + file_ext
		mask_save_name = "masks/" + file_base.replace("faces/", "").replace("/", "_") + '_' + str(i) + file_ext
		
		if os.path.isfile(crop_save_name):
			print("cropped image exists, skipping")
			continue
		
		bbox_min = np.min(landmarks, axis=0)
		bbox_max = np.max(landmarks, axis=0)
		crop_width = bbox_max[0] - bbox_min[0]
		crop_height = bbox_max[1] - bbox_min[1]
		
		bbox_min[0] = bbox_min[0] - int(crop_width*padding)
		bbox_min[1] = bbox_min[1] - int(crop_height*padding)
		
		bbox_max[0] = bbox_max[0] + int(crop_width*padding)
		bbox_max[1] = bbox_max[1] + int(crop_height*padding)
		
		bbox_has_oob = False;
		for n in bbox_min:
			if( n < 0):
				bbox_has_oob = True;
		for n in bbox_max:
			if( n < 0):
				bbox_has_oob = True;
		if(bbox_has_oob == True):
			print('bbox is out of bounds, skipping')
			continue
		
		if(min(crop_width, crop_height) < crop_size_thresh):
			print("crop is too small, skipping")
			continue
			
		crop = image[bbox_min[1]:bbox_max[1],bbox_min[0]:bbox_max[0]]
		
		eyebrow_points = []
		min_x = bbox_max[0]
		max_x = bbox_min[0]
		min_y = bbox_max[1]
		for l in landmarks[22:27]:
			lx = l[0]
			ly = l[1]
			if(ly < min_y):
				min_y = ly 
			if(lx > max_x):
				max_x = lx
			if(lx < min_x):
				min_x = lx
		eyebrow_points.append([max_x, min_y])
# 		eyebrow_points.append([min_x, min_y])
		
			
		min_x = bbox_max[0]
		max_x = bbox_min[0]
		min_y = bbox_max[1]
		for l in landmarks[17:22]:
			lx = l[0]
			ly = l[1]
			if(ly < min_y):
				min_y = ly 
			if(lx > max_x):
				max_x = lx
			if(lx < min_x):
				min_x = lx
# 		eyebrow_points.append([max_x, min_y])
		eyebrow_points.append([min_x, min_y])
	
		
		print(landmarks[22:27])
		print(landmarks[17:22])
		print(eyebrow_points)
			
		
# 		poly = reshape_for_polyline((landmarks[0:17] + list(reversed(landmarks[22:27])) + list(reversed(landmarks[17:22]))))
		
		poly = reshape_for_polyline((landmarks[0:17] + eyebrow_points))
		
		black_image = np.zeros(image.shape, np.uint8)
		cv2.fillConvexPoly(black_image, poly, (255, 255, 255))
		
		black_crop = black_image[bbox_min[1]:bbox_max[1],bbox_min[0]:bbox_max[0]]
		
		
		
		cv2.imwrite(crop_save_name, crop)
		cv2.imwrite(mask_save_name, black_crop)
# 		print(file_base, file_ext)
# 		print(crop_save_name)
# 		
# 		print(bbox_min, bbox_max)
# 		print(crop_width, crop_height)
# 		print("\n\n")
# 		cv2.imshow('im', image)
# 		cv2.imshow('crop', crop)
# 		cv2.imshow('black_crop', black_crop)
# 		cv2.waitKey(0)
		
		


	
