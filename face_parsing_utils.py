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
import math

import dlib.cuda as cuda
print(cuda.get_num_devices())

dlib.DLIB_USE_CUDA = True
print(dlib.DLIB_USE_CUDA)

from cropper import align_crop_5pts_opencv as align_crop
import cropper

from db_utils import *

face_factor = 0.7
landmark_factor = 0.35
align_type = 'similarity'
order = 3
mode = "constant"
# padding_width = 0.1;
# padding_bottom = 0.1
# padding_top = 0.5;
db_size = 128;
mean_lm = cropper._DEFAULT_MEAN_LANDMARKS


def draw_point(img, p, color ) :
#     cv2.circle( img, tuple(p), 2, color, cv2.cv.CV_FILLED, cv2.LINE_AA, 0 )
    cv2.circle(img, tuple(p), 2, color, thickness=2, lineType=8, shift=0)
    
def reshape_for_polyline(array):
    """Reshape image so that it works with polyline."""
    return np.array(array, np.int32).reshape((-1, 1, 2))

def mask_image(img, mask):
	gray_1024 = np.zeros(img.shape)
	mask_val = mask.astype(np.float)/255.0	
	foreground2 = cv2.multiply(mask_val, img.astype(np.float))
	background2 = cv2.multiply(1 - mask_val, gray_1024.astype(np.float))
	masked_image2 = cv2.add(foreground2, background2).astype(np.uint8)
	return masked_image2


def dilate_mask(img, isBig=True):
	kernel_size_small = int(math.ceil(max(img.shape)/128.0) * 2.0) + 1
	kernel_size_big = int(math.ceil(max(img.shape)/128.0) * 6.0) + 1
	kernel_small = np.ones((kernel_size_small,kernel_size_small),np.uint8)
	kernel_big = np.ones((kernel_size_big,kernel_size_big),np.uint8)
	
	num_iterations = 2
	if(isBig == False):
		num_iterations = 1
		kernel_size_small = int(math.ceil(max(img.shape)/128.0) * 1.0) + 1
		kernel_small = np.ones((kernel_size_small,kernel_size_small),np.uint8)

	bordersize = 100
	original_height = img.shape[0]
	original_width = img.shape[1]
	img = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	img = cv2.dilate(img,kernel_small,iterations = num_iterations)
# 	img = cv2.dilate(img,kernel_big,iterations = 1)
	img = img[bordersize:(original_height + bordersize), bordersize:(original_width + bordersize)]
	return img
	
def erode_mask(img):
	kernel_size_small = int(math.ceil(max(img.shape)/128.0) * 2.0) + 1
	kernel_size_big = int(math.ceil(max(img.shape)/128.0) * 6.0) + 1
	kernel_small = np.ones((kernel_size_small,kernel_size_small),np.uint8)
	kernel_big = np.ones((kernel_size_big,kernel_size_big),np.uint8)

	
	bordersize = 100
	original_height = img.shape[0]
	original_width = img.shape[1]
	img = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	img = cv2.erode(img,kernel_small,iterations = 1)
	img = img[bordersize:(original_height + bordersize), bordersize:(original_width + bordersize)]
	return img
	
def blur_mask(img, isBig=True):
	kernel_size_small = int(math.ceil(max(img.shape)/256.0) * 1.0) + 1
	kernel_size_big = int(math.ceil(max(img.shape)/128.0) * 6.0) + 1
	kernel_small = np.ones((kernel_size_small,kernel_size_small),np.uint8)
	kernel_big = np.ones((kernel_size_big,kernel_size_big),np.uint8)
	bordersize = 100
	original_height = img.shape[0]
	original_width = img.shape[1]
	img = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )


	if(isBig == True):
		img = cv2.blur(img,(kernel_size_big,kernel_size_big))
		img = cv2.blur(img,(kernel_size_big,kernel_size_big))
	else:
		img = cv2.blur(img,(kernel_size_small,kernel_size_small))
	

	img = img[bordersize:(original_height + bordersize), bordersize:(original_width + bordersize)]
	return img
	
def smooth_mask(img, isSmall=False):

	kernel_size_small = int(math.ceil(max(img.shape)/128.0) * 2.0) + 1
	kernel_size_big = int(math.ceil(max(img.shape)/128.0) * 6.0) + 1
	kernel_small = np.ones((kernel_size_small,kernel_size_small),np.uint8)
	kernel_big = np.ones((kernel_size_big,kernel_size_big),np.uint8)
	bordersize = 100
	original_height = img.shape[0]
	original_width = img.shape[1]
	img = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	
	if(isSmall == False):
		img = cv2.blur(img,(kernel_size_small,kernel_size_small))
		img = cv2.blur(img,(kernel_size_small,kernel_size_small))
		img = cv2.blur(img,(kernel_size_small,kernel_size_small))
		img = cv2.blur(img,(kernel_size_big,kernel_size_big))
		img = cv2.blur(img,(kernel_size_big,kernel_size_big))
		img = cv2.blur(img,(kernel_size_big,kernel_size_big))
		lower_red = np.array([64,64,64])
		upper_red = np.array([255,255,255])
	else:
		lower_red = np.array([128,128,128])
		upper_red = np.array([255,255,255])
		img = cv2.blur(img,(kernel_size_small,kernel_size_small))
		img = cv2.blur(img,(kernel_size_small,kernel_size_small))
		img = cv2.blur(img,(kernel_size_small,kernel_size_small))
	
	
		
	img_mask = cv2.inRange(img, lower_red, upper_red)
	img[np.where(img_mask==0)] = [0,0,0]
	img[np.where(img_mask==255)] = [255, 255, 255]

	img = img[bordersize:(original_height + bordersize), bordersize:(original_width + bordersize)]
	return img
	
# Draw voronoi diagram
def draw_voronoi(img, subdiv) :
 
    ( facets, centers) = subdiv.getVoronoiFacetList([])

    mouth_numbers = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
    
   #  jaw = (new_landmarks[0:17])
# 		left_eyebrow = (new_landmarks[22:27])
# 		right_eyebrow = (new_landmarks[17:22])
# 		nose_bridge = (new_landmarks[27:31])
# 		lower_nose = (new_landmarks[30:36])
# 		left_eye = (new_landmarks[42:48])
# 		right_eye = (new_landmarks[36:42])
# 		outer_lip = (new_landmarks[48:60])
# 		inner_lip = (new_landmarks[60:68])
# 		
    left_eye_numbers = [22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47]
    right_eye_numbers = [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41]
    face_numbers = [i for i in range(0, 17)]
    face_numbers += [27, 28, 29, 30, 31, 32, 33, 34, 35, 68, 69, 70, 71, 72, 78, 79, 80, 81, 82,  73, 74, 75, 76, 77, 83, 84, 85, 86, 87]
    
    
    
    mouth_numbers = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
    left_eye_numbers = [36, 37, 38, 39, 40, 41]
    left_eyebrow_numbers = [17, 18, 19, 20, 21]
    right_eye_numbers = [42, 43, 44, 45, 46, 47]
    right_eyebrow_numbers = [22, 23, 24, 25, 26]
    nose_numbers = [27, 28, 29, 30, 31, 32, 33, 34, 35, 78, 79, 80, 70, 71, 72]
    face_numbers = [81, 82, 68, 69]
    
    left_eye_img = img.copy()
    left_eyebrow_img = img.copy()
    right_eye_img = img.copy()
    right_eyebrow_img = img.copy()
    nose_img = img.copy()
    mouth_img = img.copy()
    face_cut_img = img.copy()
    other_img = img.copy()
    
    
    
    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
         
        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = (int( i/len(facets) * 255.0), 255 - int( i/len(facets) * 255.0), int( i/len(facets) * 255.0))
        
        this_img = other_img

        color = (0, 0, 0)
        if(i in mouth_numbers):
        	this_img = mouth_img
        	color = (255, 0, 0)
        if(i in left_eye_numbers):
        	this_img = left_eye_img
        	color = (0, 255, 0)
        if(i in right_eye_numbers):
        	this_img = right_eye_img
        	color = (0, 255, 0)
        if(i in left_eyebrow_numbers):
        	this_img = left_eyebrow_img
        	color = (0, 128, 0)
        if(i in right_eyebrow_numbers):
        	this_img = right_eyebrow_img
        	color = (0, 128, 0)
        if(i in nose_numbers):
        	this_img = nose_img
        	color = (0, 0, 255)
        if(i in face_numbers):
        	this_img = face_cut_img
        	color = (128, 128, 128)
        	
        ifacets = np.array([ifacet])
        
        cv2.fillConvexPoly(this_img, ifacet, (255,255,255), cv2.LINE_AA, 0);
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
#         cv2.circle(img, (centers[i][0], centers[i][1]), 1,  (0, 0, 0), thickness=1, lineType=8, shift=0)
        font                   = cv2.FONT_HERSHEY_COMPLEX_SMALL
        bottomLeftCornerOfText = (int(centers[i][0] - 5), centers[i][1])
        fontScale              = 0.5
        fontColor              = (255,255,255)
        lineType               = 1
        cv2.putText(img,str(i), 
        	bottomLeftCornerOfText, 
        	font, 
        	fontScale,
        	fontColor,
        	lineType)
        
    return [left_eye_img, left_eyebrow_img, right_eye_img, right_eyebrow_img, nose_img, mouth_img, face_cut_img]

def get_landmarks_from_image(image):
	faces = detector(image, 1)
	returned_landmarks = []
	for face in faces:
		predicted_info = predictor(image, face)
		detected_landmarks = predicted_info.parts()
		landmarks = [[p.x, p.y] for p in detected_landmarks]
		returned_landmarks.append(landmarks)
	return returned_landmarks

def landmarks_are_big_enough(landmarks, min_face_size):
	bbox_min = np.min(landmarks, axis=0)
	bbox_max = np.max(landmarks, axis=0)
	crop_width = bbox_max[0] - bbox_min[0]
	crop_height = bbox_max[1] - bbox_min[1]
	
	if(crop_width < min_face_size):
		return False;
	if(crop_height < min_face_size):
		return False;
	return True;
	
def filter_tuples_by_min_face_size(images_left_to_crop, min_face_size = 224):
	images_left_to_crop_return = []
	for (raw_image_path, crop_path, landmark_string) in images_left_to_crop:
		
		if(landmark_string == "no_landmarks_found"):
			print("image has no landmarks, continuing")
			continue
		if(landmark_string == "image_is_none"):
			print("image is none, continuing")
			continue
			
		landmark_list = convert_landmark_string_to_list(landmark_string)
		big_enough = landmarks_are_big_enough(landmark_list, min_face_size)
		if(big_enough is False):
			continue;
		images_left_to_crop_return.append((raw_image_path, crop_path, landmark_list))
		
	return images_left_to_crop_return
		

def get_bounding_box_with_landmarks(landmarks, image, padding_left = 0.1, padding_right = 0.1, padding_top = 1.0, padding_bottom = 0.33):
	bbox_min = np.min(landmarks, axis=0)
	bbox_max = np.max(landmarks, axis=0)
	crop_width = bbox_max[0] - bbox_min[0]
	crop_height = bbox_max[1] - bbox_min[1]
	crop_height_with_padding = crop_height + int(crop_height*padding_top) + int(crop_height*padding_bottom)
	crop_width_with_padding = crop_width + int(crop_width*padding_left) + int(crop_width*padding_right)
	extra_width_padding_perc = (((crop_height_with_padding - crop_width_with_padding)/crop_width) * 0.5)
	padding_left += extra_width_padding_perc
	padding_right += extra_width_padding_perc
	
	bbox_min[0] = bbox_min[0] - int(crop_width*padding_left)
	bbox_min[1] = bbox_min[1] - int(crop_height*padding_top)
	
	bbox_max[0] = bbox_max[0] + int(crop_width*padding_right)
	bbox_max[1] = bbox_max[1] + int(crop_height*padding_bottom)
	
# 	top, bottom, left, right - 
	border_top = int(crop_height*padding_top) * 2 
	border_bottom = int(crop_height*padding_bottom) * 2
	border_left = int(crop_width*padding_left) * 2
	border_right = int(crop_width*padding_right) * 2
	replicate = cv2.copyMakeBorder(image,border_top,border_bottom,border_left,border_right,cv2.BORDER_REPLICATE)
	bbox_min[0] = bbox_min[0] + border_left
	bbox_max[0] = bbox_max[0] + border_left
	
	bbox_min[1] = bbox_min[1] + border_top
	bbox_max[1] = bbox_max[1] + border_top
	
	
# 	if(bbox_min[0] < 0):
# 		bbox_min[0] = 0
# 	if(bbox_min[1] < 0):
# 		bbox_min[1] = 0
# 	if(bbox_max[0] < 0):
# 		bbox_max[0] = 0
# 	if(bbox_max[1] < 0):
# 		bbox_max[1] = 0
# 		
# 		
# 	if(bbox_max[0] > image.shape[1]):
# 		bbox_max[0] = image.shape[1]
# 	if(bbox_max[1] > image.shape[0]):
# 		bbox_max[1] = image.shape[0]
	
	return bbox_min, bbox_max, replicate, [border_top, border_bottom, border_left, border_right]
		
def crop_image_with_bounding_box(image, bbox_min, bbox_max):
	crop = image[bbox_min[1]:bbox_max[1],bbox_min[0]:bbox_max[0]]
	return crop

def unreplicate_image_with_border_array(image, border_array):
# 	[border_top, border_bottom, border_left, border_right] = border_array
	border_top = border_array[0]
	border_bottom = border_array[1]
	border_left = border_array[2]
	border_right = border_array[3]
	unreplicated_image = image[border_top:-border_bottom, border_left:-border_right, :]
	return unreplicated_image

def get_new_landmarks_with_bounding_box(landmarks, padding_left = 0.1, padding_right = 0.1, padding_top = 1.0, padding_bottom = 0.33):
	bbox_min = np.min(landmarks, axis=0)
	bbox_max = np.max(landmarks, axis=0)
	crop_width = bbox_max[0] - bbox_min[0]
	crop_height = bbox_max[1] - bbox_min[1]
	crop_height_with_padding = crop_height + int(crop_height*padding_top) + int(crop_height*padding_bottom)
	crop_width_with_padding = crop_width + int(crop_width*padding_left) + int(crop_width*padding_right)
	extra_width_padding_perc = (((crop_height_with_padding - crop_width_with_padding)/crop_width) * 0.5)
	padding_left += extra_width_padding_perc
	padding_right += extra_width_padding_perc
	
	bbox_min[0] = bbox_min[0] - int(crop_width*padding_left)
	bbox_min[1] = bbox_min[1] - int(crop_height*padding_top)
	
	bbox_max[0] = bbox_max[0] + int(crop_width*padding_right)
	bbox_max[1] = bbox_max[1] + int(crop_height*padding_bottom)
	
# 	top, bottom, left, right - 
	border_top = int(crop_height*padding_top) * 2 
	border_bottom = int(crop_height*padding_bottom) * 2
	border_left = int(crop_width*padding_left) * 2
	border_right = int(crop_width*padding_right) * 2
# 	replicate = cv2.copyMakeBorder(image,border_top,border_bottom,border_left,border_right,cv2.BORDER_REPLICATE)
	bbox_min[0] = bbox_min[0] + border_left
	bbox_max[0] = bbox_max[0] + border_left
	
	bbox_min[1] = bbox_min[1] + border_top
	bbox_max[1] = bbox_max[1] + border_top
	
	
	new_landmarks = []
	for l in landmarks:
		new_lx = l[0] + border_left
		new_ly = l[1] + border_top
		new_lx = int(new_lx - bbox_min[0])
		new_ly = int(new_ly - bbox_min[1])
		new_landmarks.append([new_lx, new_ly])
	return new_landmarks
	
def get_face_mask_with_landmarks_and_crop(new_landmarks, crop, bbox_min, bbox_max):
	
	
	if(bbox_max[0] > (bbox_min[0] + crop.shape[1])):
		bbox_max[0] = (bbox_min[0] + crop.shape[1])
	if(bbox_max[1] > (bbox_min[1] + crop.shape[0])):
		bbox_max[1] = (bbox_min[1] + crop.shape[0])
		
	
		

	crop_width_padded = bbox_max[0] - bbox_min[0] - 1
	crop_height_padded = bbox_max[1] - bbox_min[1] - 1
		
	jaw = (new_landmarks[0:17])
	left_eyebrow = (new_landmarks[22:27])
	right_eyebrow = (new_landmarks[17:22])
	nose_bridge = (new_landmarks[27:31])
	lower_nose = (new_landmarks[30:35])
	left_eye = (new_landmarks[42:48])
	right_eye = (new_landmarks[36:42])
	outer_lip = (new_landmarks[48:60])
	inner_lip = (new_landmarks[60:68])

	x_left_eye = [p[0] for p in left_eye]
	y_left_eye = [p[1] for p in left_eye]
	centroid_left_eye = (sum(x_left_eye) / len(left_eye), sum(y_left_eye) / len(left_eye))
	
	left_eyebutt = []
	left_eyetop = []
	left_eyetopmid = []
	left_forhead = []
	for b in left_eyebrow:
		bx = b[0]
		by = b[1]
		centroidx = centroid_left_eye[0]
		centroidy = centroid_left_eye[1]
	
	
		distx = (bx - centroidx)
		disty = (by - centroidy)
		newbx = int(bx - distx*2.0)
		newby = int(by - disty*2.0)
		newbxtop = int(bx + distx*0.2)
		newbytop = int(by + disty*0.2)
		newbxtopmid = int(bx + distx*0.3)
		newbytopmid = int(by + disty*0.3)
	
		newfx = int(bx + distx*0.5)
		newfy = int(by + disty*1.5)
	
	
	
		if(newbx < 0):
			newbx = 0
		if(newbx > crop_width_padded):
			newbx = crop_width_padded
		if(newby < 0):
			newby = 0
		if(newby > crop_height_padded):
			newby = crop_height_padded
		
		if(newbxtop < 0):
			newbxtop = 0
		if(newbxtop > crop_width_padded):
			newbxtop = crop_width_padded
		if(newbytop < 0):
			newbytop = 0
		if(newbytop > crop_height_padded):
			newbytop = crop_height_padded
		
		if(newbxtopmid < 0):
			newbxtopmid = 0
		if(newbxtopmid > crop_width_padded):
			newbxtopmid = crop_width_padded
		if(newbytopmid < 0):
			newbytopmid = 0
		if(newbytopmid > crop_height_padded):
			newbytopmid = crop_height_padded
	
		if(newfx < 0):
			newfx = 0
		if(newfx > crop_width_padded):
			newfx = crop_width_padded
		if(newfy < 0):
			newfy = 0
		if(newfy > crop_height_padded):
			newfy = crop_height_padded
		
		left_eyebutt.append([newbx, newby])
		left_eyetop.append([newbxtop, newbytop])
		left_eyetopmid.append([newbxtopmid, newbytopmid])
		left_forhead.append([newfx, newfy])
	


	x_right_eye = [p[0] for p in right_eye]
	y_right_eye = [p[1] for p in right_eye]
	centroid_right_eye = (sum(x_right_eye) / len(left_eye), sum(y_right_eye) / len(left_eye))
	right_eyebutt = []
	right_eyetop = []
	right_eyetopmid = []
	right_forhead = []
	for b in right_eyebrow:
		bx = b[0]
		by = b[1]
		centroidx = centroid_right_eye[0]
		centroidy = centroid_right_eye[1]
		distx = (bx - centroidx)
		disty = (by - centroidy)
		newbx = int(bx - distx*2.0)
		newby = int(by - disty*2.0)
		newbxtop = int(bx + distx*0.2)
		newbytop = int(by + disty*0.2)
		newbxtopmid = int(bx + distx*0.3)
		newbytopmid = int(by + disty*0.3)
	
		newfx = int(bx + distx*0.5)
		newfy = int(by + disty*1.5)
	
		if(newbx < 0):
			newbx = 0
		if(newbx > crop_width_padded):
			newbx = crop_width_padded
		if(newby < 0):
			newby = 0
		if(newby > crop_height_padded):
			newby = crop_height_padded
		
		if(newbxtop < 0):
			newbxtop = 0
		if(newbxtop > crop_width_padded):
			newbxtop = crop_width_padded
		if(newbytop < 0):
			newbytop = 0
		if(newbytop > crop_height_padded):
			newbytop = crop_height_padded
		
		
		if(newbxtopmid < 0):
			newbxtopmid = 0
		if(newbxtopmid > crop_width_padded):
			newbxtopmid = crop_width_padded
		if(newbytopmid < 0):
			newbytopmid = 0
		if(newbytopmid > crop_height_padded):
			newbytopmid = crop_height_padded
		
		
		if(newfx < 0):
			newfx = 0
		if(newfx > crop_width_padded):
			newfx = crop_width_padded
		if(newfy < 0):
			newfy = 0
		if(newfy > crop_height_padded):
			newfy = crop_height_padded
		
		
		right_eyebutt.append([newbx, newby])
		right_eyetop.append([newbxtop, newbytop])
		right_eyetopmid.append([newbxtopmid, newbytopmid])
		right_forhead.append([newfx, newfy])



	bottom_left_eye = list((new_landmarks[45:48] + [new_landmarks[42]]))
	left_eyebrows_and_bottom_eye = (left_eyetopmid + bottom_left_eye)

	bottom_right_eye = list((new_landmarks[39:42] + [new_landmarks[36]]))
	right_eyebrows_and_bottom_eye = (right_eyetopmid + bottom_right_eye)
	right_eyetop_and_bottom_eye = ([new_landmarks[0]] + list(reversed(right_eyetop)))
	left_eyetop_and_bottom_eye = ([new_landmarks[16]] + list((left_eyetop)))
	left_forhead_points = ([new_landmarks[16]] + list((left_forhead)) + [new_landmarks[0]])
	right_forhead_points = ([new_landmarks[16]] + list(reversed(right_forhead)) + [new_landmarks[0]])

	top_forheard_points = (new_landmarks[0] + right_forhead[2] + left_forhead[2] + new_landmarks[16])

	inner_left_eye_points = (new_landmarks[36:42])
	inner_right_eye_points = (new_landmarks[42:48])
	jaw_to_mouth_points = list(( new_landmarks[2:15] + list(reversed(new_landmarks[31:36]))  ))


	half_face = (new_landmarks[0:17] + list(reversed(left_eyetop)) + list(reversed(right_eyetop)))
	full_face = (new_landmarks[0:17] + list(reversed(left_forhead)) + list(reversed(right_forhead)))

# 	black_image_full_face = np.zeros(((bbox_max[1] - bbox_min[1]), (bbox_max[0] - bbox_min[0]), 3), np.uint8)
	black_image_full_face = np.zeros(crop.shape, np.uint8)
	black_image_half_face = np.zeros(crop.shape, np.uint8)
# 	black_image_half_face_not_conv = np.zeros(crop.shape, np.uint8)
# 	black_image_right_eye = black_image_left_eye.copy()
# 	black_image_full_face = black_image_left_eye.copy()
# 	black_image_right_eyetop = black_image_left_eye.copy()
# 	black_image_outer_mouth = black_image_left_eye.copy()





	poly_left_eye = reshape_for_polyline(left_eyebrows_and_bottom_eye)
	poly_right_eye = reshape_for_polyline(right_eyebrows_and_bottom_eye)
	poly_half_face = reshape_for_polyline(half_face)
	poly_full_face = reshape_for_polyline(full_face)
	poly_right_eyetop = reshape_for_polyline(right_eyetop_and_bottom_eye)
	poly_left_eyetop = reshape_for_polyline(left_eyetop_and_bottom_eye)

	poly_left_forhead = reshape_for_polyline(left_forhead_points)
	poly_right_forhead = reshape_for_polyline(right_forhead_points)
	poly_top_forhead = reshape_for_polyline(top_forheard_points)

	poly_inner_left_eye = reshape_for_polyline(inner_left_eye_points)
	poly_inner_right_eye = reshape_for_polyline(inner_right_eye_points)
	poly_inner_mouth = reshape_for_polyline(inner_lip)
	poly_outer_mouth = reshape_for_polyline(jaw_to_mouth_points)


	# cv2.fillConvexPoly(black_image_full_face, poly_full_face, (255, 255, 255))
# 	cv2.fillConvexPoly(black_image_full_face, poly_right_forhead, (255, 255, 255))
# 	cv2.fillConvexPoly(black_image_full_face, poly_left_forhead, (255, 255, 255))
# 	cv2.fillConvexPoly(black_image_full_face, poly_top_forhead, (255, 255, 255))


	cv2.fillPoly(black_image_full_face, [poly_full_face], (255, 255, 255))
# 	cv2.fillConvexPoly(black_image_full_face, poly_right_forhead, (255, 255, 255))
# 	cv2.fillConvexPoly(black_image_full_face, poly_left_forhead, (255, 255, 255))
	cv2.fillPoly(black_image_full_face, [poly_top_forhead], (255, 255, 255))
	
	cv2.fillPoly(black_image_half_face, [poly_half_face], (255, 255, 255))
# 	cv2.fillConvexPoly(black_image_half_face, poly_half_face, (255, 255, 255))
# 	cv2.fillConvexPoly(black_image_half_face, poly_left_forhead, (255, 255, 255))


# 	cv2.fillConvexPoly(black_image_outer_mouth, poly_outer_mouth, (255, 255, 255))

	# 		gray_1024 = np.zeros(crop.shape)
	# 		mask_val = black_image.astype(np.float)/255.0	
	# 		foreground2 = cv2.multiply(mask_val, crop.astype(np.float))
	# 		background2 = cv2.multiply(1 - mask_val, gray_1024.astype(np.float))
	# 		masked_image2 = cv2.add(foreground2, background2).astype(np.uint8)
	# 		

	new_landmarks_added = new_landmarks + left_eyebutt + left_eyetop + right_eyebutt + right_eyetop

	# Read in the image.
	img = crop.copy()

	size = img.shape
	rect = (0, 0, size[1], size[0])
	subdiv  = cv2.Subdiv2D(rect); 


	# Rectangle to be used with Subdiv2D
	size = img.shape
	rect = (0, 0, size[1], size[0])

	# Create an instance of Subdiv2D
	subdiv = cv2.Subdiv2D(rect);

	# Create an array of points.
	points = new_landmarks_added.copy()
	points = new_landmarks.copy()
	
	
# 	This makes sure one of our points isn't outside our crop
	points_holder = [];
	for l in points:
		new_lx = l[0]
		new_ly = l[1]
		if(new_lx < 0):
			new_lx = 0
		if(new_ly < 0):
			new_ly = 0
		if(new_lx > bbox_max[0]):
			new_lx = bbox_max[0]
		if(new_ly > bbox_max[1]):
			new_ly = bbox_max[1]
		points_holder.append([new_lx, new_ly])
	points = points_holder.copy()


	# Insert points into subdiv
	for p in points :
		if(p[0] >= rect[2]):
			p[0] = rect[2] - 1
		if(p[1] >= rect[3]):
			p[1] = rect[3] - 1
		subdiv.insert((p[0], p[1]))
 
# 	for p in new_landmarks_added :
# 		draw_point(img, p, (255,255,255))
		

	# Allocate space for Voronoi Diagram
	img_voronoi = np.zeros(img.shape, dtype = img.dtype)

	# Draw Voronoi diagram
	[left_eye_img_v, left_eyebrow_img_v, right_eye_img_v, right_eyebrow_img_v, nose_img_v, mouth_img_v, face_cut_img_v] = draw_voronoi(img_voronoi,subdiv)

	lower_red = np.array([128,128,128])
	upper_red = np.array([255,255,255])

	left_eye_img_v = cv2.inRange(left_eye_img_v, lower_red, upper_red)
	left_eyebrow_img_v = cv2.inRange(left_eyebrow_img_v, lower_red, upper_red)
	right_eye_img_v = cv2.inRange(right_eye_img_v, lower_red, upper_red)
	right_eyebrow_img_v = cv2.inRange(right_eyebrow_img_v, lower_red, upper_red)
	nose_img_v = cv2.inRange(nose_img_v, lower_red, upper_red)
	mouth_img_v = cv2.inRange(mouth_img_v, lower_red, upper_red)
	face_cut_img_v = cv2.inRange(face_cut_img_v, lower_red, upper_red)
	
	
	
	
	left_eye_img_v[np.where(black_image_full_face[:,:,0]==[0])] = [0]
	left_eyebrow_img_v[np.where(black_image_half_face[:,:,0]==[0])] = [0]
	right_eye_img_v[np.where(black_image_full_face[:,:,0]==[0])] = [0]
	right_eyebrow_img_v[np.where(black_image_half_face[:,:,0]==[0])] = [0]
	nose_img_v[np.where(black_image_full_face[:,:,0]==[0])] = [0]
	mouth_img_v[np.where(black_image_full_face[:,:,0]==[0])] = [0]
	face_cut_img_v[np.where(black_image_full_face[:,:,0]==[0])] = [0]

	
	

# 
# 
# 	black_image_outer_mouth_dilated = black_image_outer_mouth.copy()
# 	black_image_outer_mouth_dilated = erode_mask(black_image_outer_mouth_dilated.copy())
# 	# 		black_image_outer_mouth_dilated = dilate_mask(black_image_outer_mouth_dilated.copy())
# 	# 		black_image_outer_mouth_dilated = dilate_mask(black_image_outer_mouth_dilated.copy())
# 	# 		black_image_outer_mouth_dilated = dilate_mask(black_image_outer_mouth_dilated.copy())
# 	black_image_outer_mouth_dilated = smooth_mask(black_image_outer_mouth_dilated.copy())
# 
# 	# 		cv2.imshow('black_image_outer_mouth_dilated', black_image_outer_mouth_dilated)
# 	mouth_img_v[np.where(black_image_outer_mouth_dilated[:,:,0]==[0])] = [0]

	mask_holder_image = black_image_full_face.copy()


	mask_holder_image[np.where(left_eye_img_v==[255])] = [255,0,0]
	mask_holder_image[np.where(left_eyebrow_img_v==[255])] = [0, 128, 128]
	mask_holder_image[np.where(right_eye_img_v==[255])] = [0, 0, 255]
	mask_holder_image[np.where(right_eyebrow_img_v==[255])] = [128, 128, 0]
	mask_holder_image[np.where(nose_img_v==[255])] = [0, 255, 0]
	mask_holder_image[np.where(mouth_img_v==[255])] = [255, 0, 255]

	cv2.fillConvexPoly(mask_holder_image, poly_inner_left_eye, (0, 255, 255))
	cv2.fillConvexPoly(mask_holder_image, poly_inner_right_eye, (255, 255, 0))
	cv2.fillConvexPoly(mask_holder_image, poly_inner_mouth, (0, 128, 0))
	
	return mask_holder_image
	
def final_full_face_big_before_seamless_clone(mask, eyemask):
	new_mask = mask.copy()
	new_mask = dilate_mask(new_mask)
	new_mask = dilate_mask(new_mask)
	new_mask[np.where(eyemask==[255])] = 0
	new_mask = blur_mask(new_mask)
	return new_mask
	
def align_crop_and_mask_with_landmarks(new_landmarks, crop, mask_face):
	
	
	oijoilandmarks = cropper.convert_68pt_to_5pt(new_landmarks)
	crop_size = [int(crop.shape[0]/(face_factor)), int(crop.shape[1]/(face_factor))]
	img_crop, img_align, img_mask = align_crop(crop.copy(),
							  oijoilandmarks.copy(),
							  mean_lm,
							  crop_size=crop_size.copy(),
							  face_factor=face_factor,
							  landmark_factor=landmark_factor,
							  align_type=align_type,
							  order=order,
							  mode=mode,
							  background_color=128)

	img_crop_mask_face, img_align_mask_face, img_mask_mask_face = align_crop(mask_face.copy(),
							  oijoilandmarks.copy(),
							  mean_lm,
							  crop_size=crop_size.copy(),
							  face_factor=face_factor,
							  landmark_factor=landmark_factor,
							  align_type=align_type,
							  order=order,
							  mode=mode,
							  background_color=0)    
	return img_crop, img_crop_mask_face

def unalign_crop(aligned, original, new_landmarks):
	oijoilandmarks = cropper.convert_68pt_to_5pt(new_landmarks)
	crop_size = [int(original.shape[0]/(face_factor)), int(original.shape[1]/(face_factor))]
	img_crop_ret2, img_return2, img_mask2 = cropper.align_crop_5pts_opencv(aligned,
                                  unalign_crop.copy(),
                                  croppermean_lm,
                                  crop_size=crop_size,
                                  face_factor=face_factor,
                                  landmark_factor=landmark_factor,
                                  align_type=align_type,
                                  order=order,
                                  mode=mode,
                                  original_crop=original)
	return img_return2
                                  
def extract_segment_from_face(mask, value):

	delta_val = 64;
	
	lower_val_1 = max(0, value[0] - delta_val)
	lower_val_2 = max(0, value[1] - delta_val)
	lower_val_3 = max(0, value[2] - delta_val)
	
	upper_val_1 = min(255, value[0] + delta_val)
	upper_val_2 = min(255, value[1] + delta_val)
	upper_val_3 = min(255, value[2] + delta_val)
	
	lower_val_mask = np.array([lower_val_1, lower_val_2, lower_val_3])
	upper_val_mask = np.array([upper_val_1, upper_val_2, upper_val_3])
	
	img_mask = cv2.inRange(mask, lower_val_mask, upper_val_mask)
	masked_image = mask.copy()
	masked_image[np.where(img_mask==0)] = [0,0,0]
	masked_image[np.where(img_mask==255)] = [255, 255, 255]
	return masked_image
	
def add_padding_to_crop_array(crop_array, padding, image_size):
	width = crop_array[1] - crop_array[0]
	height = crop_array[3] - crop_array[2]
	delta_width = width*padding
	delta_height = height*padding
	
	new_x0 = max(0, (crop_array[0] - delta_width))
	new_x1 = min(image_size[0], (crop_array[1] + delta_width))
	
	new_y0 = max(0, (crop_array[2] - delta_height))
	new_y1 = min(image_size[1], (crop_array[3] + delta_height))
	
	return [int(new_x0), int(new_x1), int(new_y0), int(new_y1)]
	
def crop_image(image, array):
	return image[array[0]:array[1], array[2]:array[3]]

def uncrop_image(image, original, array):
	original[array[0]:array[1], array[2]:array[3]] = image
	return original

def get_crop_array(image,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = image > 0

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    
    try:
    	x0, y0 = coords.min(axis=0)
    	x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    	return [x0, x1, y0, y1]
    except ValueError as e:
    	print(e)
#     	cv2.imshow('image', image)
#     	cv2.waitKey(0)
    	return 
    	
def generate_individual_masks_from_face_mask(crop, mask_face):

	
	
	left_eye_mask = extract_segment_from_face(mask_face, [255,0,0])
	left_eyebrow_mask = extract_segment_from_face(mask_face, [0, 128, 128])
	right_eye_mask = extract_segment_from_face(mask_face, [0, 0, 255])
	right_eyebrow_mask = extract_segment_from_face(mask_face, [128, 128, 0])	
	mouth_mask = extract_segment_from_face(mask_face, [255, 0, 255])
	nose_mask = extract_segment_from_face(mask_face, [0, 255, 0])
	face_cut_mask = extract_segment_from_face(mask_face, [255,255,255])
	left_inner_eye_mask = extract_segment_from_face(mask_face, [0, 255, 255])
	right_inner_eye_mask = extract_segment_from_face(mask_face, [255, 255, 0])
	mouth_inner_mask = extract_segment_from_face(mask_face, [0, 128, 0])

	

	kernel_size_small = int(math.ceil(max(crop.shape)/256.0) * 1.0) + 1
	
	kernel = np.ones((kernel_size_small,kernel_size_small),np.uint8)

	left_eye_mask_open = cv2.morphologyEx(left_eye_mask, cv2.MORPH_OPEN, kernel)
	left_eyebrow_mask_open = cv2.morphologyEx(left_eyebrow_mask, cv2.MORPH_OPEN, kernel)
	right_eye_mask_open = cv2.morphologyEx(right_eye_mask, cv2.MORPH_OPEN, kernel)
	right_eyebrow_mask_open = cv2.morphologyEx(right_eyebrow_mask, cv2.MORPH_OPEN, kernel)
	mouth_mask_open = cv2.morphologyEx(mouth_mask, cv2.MORPH_OPEN, kernel)
	nose_mask_open = cv2.morphologyEx(nose_mask, cv2.MORPH_OPEN, kernel)
	face_cut_mask_open = cv2.morphologyEx(face_cut_mask, cv2.MORPH_OPEN, kernel)
	left_inner_eye_mask_open = cv2.morphologyEx(left_inner_eye_mask, cv2.MORPH_OPEN, kernel)
	right_inner_eye_mask_open = cv2.morphologyEx(right_inner_eye_mask, cv2.MORPH_OPEN, kernel)
	mouth_inner_mask_open = cv2.morphologyEx(mouth_inner_mask, cv2.MORPH_OPEN, kernel)
	
	return_masks = []
	return_masks.append(left_eye_mask_open)
	return_masks.append(left_eyebrow_mask_open)
	return_masks.append(right_eye_mask_open)
	return_masks.append(right_eyebrow_mask_open)
	return_masks.append(mouth_mask_open)
	return_masks.append(nose_mask_open)
	return_masks.append(face_cut_mask_open)
	return_masks.append(left_inner_eye_mask_open)
	return_masks.append(right_inner_eye_mask_open)
	return_masks.append(mouth_inner_mask_open)
	
	return return_masks
	

	left_eye_crop_array = get_crop_array(left_eye_mask_open[:,:,0])
	left_eyebrow_crop_array = get_crop_array(left_eyebrow_mask_open[:,:,0])
	right_eye_crop_array = get_crop_array(right_eye_mask_open[:,:,0])
	right_eyebrow_crop_array = get_crop_array(right_eyebrow_mask_open[:,:,0])
	mouth_crop_array = get_crop_array(mouth_mask_open[:,:,0])
	nose_crop_array = get_crop_array(nose_mask_open[:,:,0])
	face_cut_crop_array = get_crop_array(face_cut_mask_open[:,:,0])
	left_inner_eye_crop_array = get_crop_array(left_inner_eye_mask_open[:,:,0])
	right_inner_eye_crop_array = get_crop_array(right_inner_eye_mask_open[:,:,0])
	mouth_inner_crop_array = get_crop_array(mouth_inner_mask_open[:,:,0])
	
	


	if(left_eye_crop_array is None):
		return None, None, None
	if(left_eyebrow_crop_array is None):
		return None, None, None
	if(right_eye_crop_array is None):
		return None, None, None
	if(right_eyebrow_crop_array is None):
		return None, None, None
	if(mouth_crop_array is None):
		return None, None, None
	if(nose_crop_array is None):
		return None, None, None
	if(face_cut_crop_array is None):
		return None, None, None
	if(left_inner_eye_crop_array is None):
		return None, None, None
	if(right_inner_eye_crop_array is None):
		return None, None, None
	if(mouth_inner_crop_array is None):
		return None, None, None

	left_eye_crop_array = add_padding_to_crop_array(left_eye_crop_array.copy(), 0.2, mask_face.shape)
	left_eyebrow_crop_array = add_padding_to_crop_array(left_eyebrow_crop_array.copy(), 0.2, mask_face.shape)
	right_eye_crop_array = add_padding_to_crop_array(right_eye_crop_array.copy(), 0.2, mask_face.shape)
	right_eyebrow_crop_array = add_padding_to_crop_array(right_eyebrow_crop_array.copy(), 0.2, mask_face.shape)
	mouth_crop_array = add_padding_to_crop_array(mouth_crop_array.copy(), 0.2, mask_face.shape)
	nose_crop_array = add_padding_to_crop_array(nose_crop_array.copy(), 0.2, mask_face.shape)
	face_cut_crop_array = add_padding_to_crop_array(face_cut_crop_array.copy(), 0.2, mask_face.shape)
	left_inner_eye_crop_array = add_padding_to_crop_array(left_inner_eye_crop_array.copy(), 0.2, mask_face.shape)
	right_inner_eye_crop_array = add_padding_to_crop_array(right_inner_eye_crop_array.copy(), 0.2, mask_face.shape)
	mouth_inner_crop_array = add_padding_to_crop_array(mouth_inner_crop_array.copy(), 0.2, mask_face.shape)

	left_inner_eye_crop_array = left_eye_crop_array.copy()
	right_inner_eye_crop_array = right_eye_crop_array.copy()
	mouth_inner_crop_array = mouth_crop_array.copy()
	

	left_eye_mask_open_crop = crop_image(left_eye_mask_open.copy(), left_eye_crop_array.copy())
	left_eyebrow_mask_open_crop = crop_image(left_eyebrow_mask_open.copy(), left_eyebrow_crop_array.copy())
	right_eye_mask_open_crop = crop_image(right_eye_mask_open.copy(), right_eye_crop_array.copy())
	right_eyebrow_mask_open_crop = crop_image(right_eyebrow_mask_open.copy(), right_eyebrow_crop_array.copy())
	mouth_mask_open_crop = crop_image(mouth_mask_open.copy(), mouth_crop_array.copy())
	nose_mask_open_crop = crop_image(nose_mask_open.copy(), nose_crop_array.copy())
	face_cut_mask_open_crop = crop_image(face_cut_mask_open.copy(), face_cut_crop_array.copy())
	left_inner_eye_mask_open_crop = crop_image(left_inner_eye_mask_open.copy(), left_inner_eye_crop_array.copy())
	right_inner_eye_mask_open_crop = crop_image(right_inner_eye_mask_open.copy(), right_inner_eye_crop_array.copy())
	mouth_inner_mask_open_crop = crop_image(mouth_inner_mask_open.copy(), mouth_inner_crop_array.copy())

	left_eye_original_crop = crop_image(crop.copy(), left_eye_crop_array.copy())
	left_eyebrow_original_crop = crop_image(crop.copy(), left_eyebrow_crop_array.copy())
	right_eye_original_crop = crop_image(crop.copy(), right_eye_crop_array.copy())
	right_eyebrow_original_crop = crop_image(crop.copy(), right_eyebrow_crop_array.copy())
	mouth_original_crop = crop_image(crop.copy(), mouth_crop_array.copy())
	nose_original_crop = crop_image(crop.copy(), nose_crop_array.copy())
	face_cut_original_crop = crop_image(crop.copy(), face_cut_crop_array.copy())
	left_inner_eye_original_crop = crop_image(crop.copy(), left_inner_eye_crop_array.copy())
	right_inner_eye_original_crop = crop_image(crop.copy(), right_inner_eye_crop_array.copy())
	mouth_inner_original_crop = crop_image(crop.copy(), mouth_inner_crop_array.copy())
	
	return_masks = []
	return_crop_arrays = []
	return_crops = []
	
	return_masks.append(left_eye_mask_open_crop)
	return_masks.append(left_eyebrow_mask_open_crop)
	return_masks.append(right_eye_mask_open_crop)
	return_masks.append(right_eyebrow_mask_open_crop)
	return_masks.append(mouth_mask_open_crop)
	return_masks.append(nose_mask_open_crop)
	return_masks.append(face_cut_mask_open_crop)
	return_masks.append(left_inner_eye_mask_open_crop)
	return_masks.append(right_inner_eye_mask_open_crop)
	return_masks.append(mouth_inner_mask_open_crop)
	
	return_crop_arrays.append(left_eye_crop_array)
	return_crop_arrays.append(left_eyebrow_crop_array)
	return_crop_arrays.append(right_eye_crop_array)
	return_crop_arrays.append(right_eyebrow_crop_array)
	return_crop_arrays.append(mouth_crop_array)
	return_crop_arrays.append(nose_crop_array)
	return_crop_arrays.append(face_cut_crop_array)
	return_crop_arrays.append(left_inner_eye_crop_array)
	return_crop_arrays.append(right_inner_eye_crop_array)
	return_crop_arrays.append(mouth_inner_crop_array)
	
	return_crops.append(left_eye_original_crop)
	return_crops.append(left_eyebrow_original_crop)
	return_crops.append(right_eye_original_crop)
	return_crops.append(right_eyebrow_original_crop)
	return_crops.append(mouth_original_crop)
	return_crops.append(nose_original_crop)
	return_crops.append(face_cut_original_crop)
	return_crops.append(left_inner_eye_original_crop)
	return_crops.append(right_inner_eye_original_crop)
	return_crops.append(mouth_inner_original_crop)
	
	
	
	
	return return_masks, return_crop_arrays, return_crops
	
def get_masks_from_new_landmarks(new_landmarks, crop_width_padded, crop_height_padded):

	
	jaw = (new_landmarks[0:17])
	left_eyebrow = (new_landmarks[22:27])
	right_eyebrow = (new_landmarks[17:22])
	nose_bridge = (new_landmarks[27:31])
	lower_nose = (new_landmarks[30:35])
	left_eye = (new_landmarks[42:48])
	right_eye = (new_landmarks[36:42])
	outer_lip = (new_landmarks[48:60])
	inner_lip = (new_landmarks[60:68])
	
	x_left_eye = [p[0] for p in left_eye]
	y_left_eye = [p[1] for p in left_eye]
	centroid_left_eye = (sum(x_left_eye) / len(left_eye), sum(y_left_eye) / len(left_eye))
	left_eyebutt = []
	left_eyetop = []
	left_eyetopmid = []
	for b in left_eyebrow:
		bx = b[0]
		by = b[1]
		centroidx = centroid_left_eye[0]
		centroidy = centroid_left_eye[1]
		distx = (bx - centroidx)
		disty = (by - centroidy)
		newbx = int(bx - distx*2.0)
		newby = int(by - disty*2.0)
		newbxtop = int(bx + distx*0.6)
		newbytop = int(by + disty*0.6)
		newbxtopmid = int(bx + distx*0.3)
		newbytopmid = int(by + disty*0.3)
		
		
		
		if(newbx < 0):
			newbx = 0
		if(newbx > crop_width_padded):
			newbx = crop_width_padded
		if(newby < 0):
			newby = 0
		if(newby > crop_height_padded):
			newby = crop_height_padded
			
		if(newbxtop < 0):
			newbxtop = 0
		if(newbxtop > crop_width_padded):
			newbxtop = crop_width_padded
		if(newbytop < 0):
			newbytop = 0
		if(newbytop > crop_height_padded):
			newbytop = crop_height_padded
			
		if(newbxtopmid < 0):
			newbxtopmid = 0
		if(newbxtopmid > crop_width_padded):
			newbxtopmid = crop_width_padded
		if(newbytopmid < 0):
			newbytopmid = 0
		if(newbytopmid > crop_height_padded):
			newbytopmid = crop_height_padded
		
		left_eyebutt.append([newbx, newby])
		left_eyetop.append([newbxtop, newbytop])
		left_eyetopmid.append([newbxtopmid, newbytopmid])
		

	x_right_eye = [p[0] for p in right_eye]
	y_right_eye = [p[1] for p in right_eye]
	centroid_right_eye = (sum(x_right_eye) / len(left_eye), sum(y_right_eye) / len(left_eye))
	right_eyebutt = []
	right_eyetop = []
	right_eyetopmid = []
	for b in right_eyebrow:
		bx = b[0]
		by = b[1]
		centroidx = centroid_right_eye[0]
		centroidy = centroid_right_eye[1]
		distx = (bx - centroidx)
		disty = (by - centroidy)
		newbx = int(bx - distx*2.0)
		newby = int(by - disty*2.0)
		newbxtop = int(bx + distx*0.6)
		newbytop = int(by + disty*0.6)
		newbxtopmid = int(bx + distx*0.3)
		newbytopmid = int(by + disty*0.3)
		
		if(newbx < 0):
			newbx = 0
		if(newbx > crop_width_padded):
			newbx = crop_width_padded
		if(newby < 0):
			newby = 0
		if(newby > crop_height_padded):
			newby = crop_height_padded
			
		if(newbxtop < 0):
			newbxtop = 0
		if(newbxtop > crop_width_padded):
			newbxtop = crop_width_padded
		if(newbytop < 0):
			newbytop = 0
		if(newbytop > crop_height_padded):
			newbytop = crop_height_padded
			
			
		if(newbxtopmid < 0):
			newbxtopmid = 0
		if(newbxtopmid > crop_width_padded):
			newbxtopmid = crop_width_padded
		if(newbytopmid < 0):
			newbytopmid = 0
		if(newbytopmid > crop_height_padded):
			newbytopmid = crop_height_padded
			
		right_eyebutt.append([newbx, newby])
		right_eyetop.append([newbxtop, newbytop])
		right_eyetopmid.append([newbxtopmid, newbytopmid])


	jaw_lowered_to_include_neck = []
	x_nose = [p[0] for p in lower_nose]
	y_nose = [p[1] for p in lower_nose]
	centroid_nose = (sum(x_nose) / len(lower_nose), sum(y_nose) / len(lower_nose))
	for b in jaw:
		bx = b[0]
		by = b[1]
		centroidx = centroid_nose[0]
		centroidy = centroid_nose[1]
		distx = (bx - centroidx)
		disty = (by - centroidy)
		newbx = int(bx - distx*0.0)
		newby = int(by + disty*0.5)

		
		if(newbx < 0):
			newbx = 0
		if(newbx > crop_width_padded):
			newbx = crop_width_padded
		if(newby < 0):
			newby = 0
		if(newby > crop_height_padded):
			newby = crop_height_padded
			
		
			
		jaw_lowered_to_include_neck.append([newbx, newby])

		
	
	bottom_left_eye = list((new_landmarks[45:48] + [new_landmarks[42]]))
	left_eyebrows_and_bottom_eye = (left_eyetopmid + bottom_left_eye)
	
	bottom_right_eye = list((new_landmarks[39:42] + [new_landmarks[36]]))
	right_eyebrows_and_bottom_eye = (right_eyetopmid + bottom_right_eye)
	right_eyetop_and_bottom_eye = ([new_landmarks[0]] + list(reversed(right_eyetop)))
	left_eyetop_and_bottom_eye = ([new_landmarks[16]] + list((left_eyetop)))

	pts = np.array(new_landmarks[:17]).astype(np.int32)
	baseline_y = (pts[0,1] + pts[-1,1]) / 2
	
	upper_pts = pts[1:-1,:].copy()
	upper_pts[:,1] = baseline_y + (baseline_y-upper_pts[:,1])
	
# 	lower_pts = pts[1:-1,:].copy()
# 	lower_pts[:,1] = baseline_y - (baseline_y-upper_pts[:,1]) * 2 // 3
	
	full_face_big = (jaw_lowered_to_include_neck + list(upper_pts[::-1,:])) 
# 	full_face_big = (list(lower_pts) + list(upper_pts[::-1,:])) 
	
	full_face = (new_landmarks[0:17] + list(reversed(left_eyetop)) + list(reversed(right_eyetop)))
# 	full_face_big = 

	black_image_left_eye = np.zeros((crop_width_padded, crop_height_padded, 3), np.uint8)
	black_image_right_eye = black_image_left_eye.copy()
	black_image_full_face = black_image_left_eye.copy()
	black_image_full_face_big = black_image_left_eye.copy()
	
	black_image_right_eyetop = black_image_left_eye.copy()
	black_image_lips_only = black_image_left_eye.copy()
	
	black_image_inner_mouth = black_image_left_eye.copy()
	
	black_image_lips_outer = black_image_left_eye.copy()
	black_image_lips_inner = black_image_left_eye.copy()
	black_image_both_eyes = black_image_left_eye.copy()
	black_image_both_eyeshadow = black_image_left_eye.copy()
	black_image_both_eyeliner = black_image_left_eye.copy()
	black_image_both_eyebrows = black_image_left_eye.copy()
	black_image_both_cheeks = black_image_left_eye.copy()
	
	
# 	black_image_eyebrows = black_image_left_eye.copy()
	
	poly_left_eye = reshape_for_polyline(left_eyebrows_and_bottom_eye)
	
	poly_right_eye = reshape_for_polyline(right_eyebrows_and_bottom_eye)
	poly_full_face = reshape_for_polyline(full_face)
	poly_full_face_big = reshape_for_polyline(full_face_big)
	poly_right_eyetop = reshape_for_polyline(right_eyetop_and_bottom_eye)
	poly_left_eyetop = reshape_for_polyline(left_eyetop_and_bottom_eye)
	
	
	poly_outer_lips = reshape_for_polyline(outer_lip)
	poly_inner_lips = reshape_for_polyline(inner_lip)
	poly_left_eye_only = reshape_for_polyline(left_eye)
	poly_right_eye_only = reshape_for_polyline(right_eye)
	
	
	poly_left_eyebrow = reshape_for_polyline(left_eyebrow)
	poly_right_eyebrow = reshape_for_polyline(right_eyebrow)

	
	cv2.fillConvexPoly(black_image_both_eyes, poly_left_eye_only, (255, 255, 255))
	cv2.fillConvexPoly(black_image_both_eyes, poly_right_eye_only, (255, 255, 255))
	
	cv2.fillPoly(black_image_lips_outer, [poly_outer_lips], (255, 255, 255))
	cv2.fillPoly(black_image_lips_outer, [poly_inner_lips], (0, 0, 0))
	
	
	cv2.fillConvexPoly(black_image_right_eye, poly_right_eye, (255, 255, 255))
	cv2.fillPoly(black_image_full_face, [poly_full_face], (255, 255, 255))
	cv2.fillPoly(black_image_full_face_big, [poly_full_face_big], (255, 255, 255))
# 	cv2.fillConvexPoly(black_image_full_face, poly_right_eyetop, (255, 255, 255))
# 	cv2.fillConvexPoly(black_image_full_face, poly_left_eyetop, (255, 255, 255))
	cv2.fillPoly(black_image_lips_only, [poly_outer_lips], (255, 255, 255))
	cv2.fillPoly(black_image_lips_only, [poly_inner_lips], (0, 0, 0))
	cv2.fillPoly(black_image_inner_mouth, [poly_inner_lips], (255, 255, 255))
	

	black_image_skin = black_image_full_face.copy()
	cv2.fillConvexPoly(black_image_skin, poly_right_eye, (0, 0, 0))
	cv2.fillConvexPoly(black_image_skin, poly_left_eye, (0, 0, 0))
	cv2.fillConvexPoly(black_image_skin, poly_outer_lips, (0, 0, 0))
	
# 	cv2.polylines(black_image_256, [inner_lip], True, (0,0,0), thickness)
# 	
# 	cv2.polylines(black_image_both_eyebrows,[poly_left_eyebrow],False,(255,255,255),thickness = int((black_image_left_eye.shape[0]/256.0 + 1) * 2.0))
# 	cv2.polylines(black_image_both_eyebrows,[poly_right_eyebrow],False,(255,255,255),thickness = int((black_image_left_eye.shape[0]/256.0 + 1) * 2.0))

	
	
	#voronoi
	
	new_landmarks_added = new_landmarks + left_eyebutt + left_eyetop + right_eyebutt + right_eyetop
	
	# Read in the image.
	
	img = np.zeros((crop_width_padded, crop_height_padded, 3), np.uint8)
	
	size = (crop_width_padded, crop_height_padded, 3)
	rect = (0, 0, size[1], size[0])
	subdiv  = cv2.Subdiv2D(rect); 

	
	# Define window names
	win_delaunay = "Delaunay Triangulation"
	win_voronoi = "Voronoi Diagram"

	# Turn on animation while drawing triangles
	animate = True
 
	# Define colors for drawing.
	delaunay_color = (255,255,255)
	points_color = (0, 0, 255)

	
 
	# Keep a copy around
	img_orig = img.copy();
 
	# Rectangle to be used with Subdiv2D
	size = img.shape
	rect = (0, 0, size[1], size[0])
 
	# Create an instance of Subdiv2D
	subdiv = cv2.Subdiv2D(rect);

	# Create an array of points.
	points = new_landmarks_added.copy()


	# Insert points into subdiv
	for p in points :
		if(p[0] >= rect[2]):
			p[0] = rect[2] - 1
		if(p[1] >= rect[3]):
			p[1] = rect[3] - 1
		subdiv.insert((p[0], p[1]))
	 
# 			Show animation
# 			if animate :
# 				img_copy = img_orig.copy()
# 				Draw delaunay triangles
# 				draw_delaunay( img_copy, subdiv, (255, 255, 255) );
# 				cv2.imshow(win_delaunay, img_copy)
# 				cv2.waitKey(10)

	# Draw delaunay triangles
# 		draw_delaunay( img, subdiv, (255, 255, 255) );
#  
# 		# Draw points
# 		for p in points :
# 			draw_point(img, p, (0,0,255))

	# Allocate space for Voronoi Diagram
	img_voronoi = np.zeros(img.shape, dtype = img.dtype)

	# Draw Voronoi diagram
# 	[left_eye_img_v, right_eye_img_v, mouth_img_v, other_img_v, eyes_only_img, eyebrow_img, nose_img] = draw_voronoi(img_voronoi,subdiv)
	[left_eye_img, left_eyebrow_img, right_eye_img, right_eyebrow_img, nose_img, mouth_img, face_cut_img] = draw_voronoi(img_voronoi,subdiv)


	
	black_image_both_eyebrows[np.where(left_eyebrow_img==255)] = 255
	black_image_both_eyebrows[np.where(right_eyebrow_img==255)] = 255
	
	kernel_size = int(math.ceil(max(img.shape)/128.0) * 4.0) + 1
	kernel = np.ones((kernel_size,kernel_size),np.uint8)

	
	black_image_both_eyes_dilated = dilate_mask(black_image_both_eyes.copy(), isBig=False)
	black_image_both_eyes_dilated_2 = dilate_mask(black_image_both_eyes_dilated.copy())
	black_image_both_eyeshadow = smooth_mask(black_image_both_eyes_dilated_2.copy())
	black_image_both_eyeshadow[np.where(black_image_both_eyes_dilated==255)] = 0
	
	
	
	black_image_both_eyeliner = dilate_mask(black_image_both_eyes.copy(), isBig=False)
	black_image_both_eyeliner[np.where(black_image_both_eyes==255)] = 0
	
	black_image_both_eyebrows = erode_mask(black_image_both_eyebrows.copy())
	black_image_both_eyebrows = erode_mask(black_image_both_eyebrows.copy())
	black_image_both_eyebrows = erode_mask(black_image_both_eyebrows.copy())
	cv2.polylines(black_image_both_eyebrows,[poly_left_eyebrow],False,(255,255,255),thickness = int((black_image_left_eye.shape[0]/256.0 + 1) * 2.0))
	cv2.polylines(black_image_both_eyebrows,[poly_right_eyebrow],False,(255,255,255),thickness = int((black_image_left_eye.shape[0]/256.0 + 1) * 2.0))
	
	black_image_both_eyebrows_smoothed = smooth_mask(black_image_both_eyebrows.copy(), isSmall=True)


	nose_img_smoothed = smooth_mask(nose_img.copy())
	nose_img_eroded = erode_mask(nose_img_smoothed.copy())
	
# 	black_image_both_eyeshadow[np.where(nose_img_eroded==255)] = 0
	black_image_both_eyeshadow[np.where(black_image_both_eyebrows==255)] = 0
	black_image_both_eyebrows[np.where(nose_img_eroded==255)] = 0
	
	black_image_full_face = smooth_mask(black_image_full_face.copy())
	black_image_full_face[np.where(black_image_both_eyes==255)] = 0
	black_image_full_face[np.where(black_image_lips_outer==255)] = 0
	black_image_full_face[np.where(black_image_inner_mouth==255)] = 0
# 	black_image_full_face[np.where(black_image_both_eyebrows_smoothed==255)] = 0
	
	
	mask_crop_arr = []
	mask_crop_arr.append(black_image_both_eyeshadow)
	mask_crop_arr.append(black_image_both_eyeliner)
	mask_crop_arr.append(black_image_lips_outer)
	mask_crop_arr.append(black_image_both_eyebrows_smoothed)
	mask_crop_arr.append(black_image_inner_mouth)
	mask_crop_arr.append(black_image_full_face)
	mask_crop_arr.append(black_image_both_eyes)
	mask_crop_arr.append(black_image_full_face_big)
	

	return mask_crop_arr
	
def segment_crop_with_mask(crop, mask, average, cutoutMasks=None):
	mask_smoothed = smooth_mask(mask.copy())
	
	if(cutoutMasks is not None):
		for mask in cutoutMasks:
			mask_smoothed[np.where(mask==255)] = 0
			
	mask_smoothed_dilated = dilate_mask(mask_smoothed.copy())
	mask_smoothed_dilated_small = dilate_mask(mask_smoothed.copy(), isBig = False)
	
	
	
		
	mask_smoothed_dilated_blurred = blur_mask(mask_smoothed_dilated.copy())
	mask_smoothed_dilated_blurred_small = blur_mask(mask_smoothed_dilated_small.copy())
	crop_array = get_crop_array(mask_smoothed_dilated_blurred[:,:,0])
	
	if(crop_array is None):
		return None, None, None, None, None, None, None, None, None
		
	cropped_crop = crop_image(crop.copy(), crop_array)
	cropped_mask = crop_image(mask_smoothed_dilated_blurred.copy(), crop_array)
	cropped_mask_small = crop_image(mask_smoothed_dilated_blurred_small.copy(), crop_array)
	
	
	cropped_mask_float = cropped_mask.copy().astype(np.float)/255.0
# 	print("FUCK")
# 	print(cropped_mask_float.shape, cropped_crop.shape)
# 	print(mask.shape, crop.shape)
# 	
# 	if(mask.shape != crop.shape):
# 		cv2.imshow('mask', mask)
# 		cv2.imshow('crop', crop)
# 		cv2.waitKey(0)
		
	foreground = cv2.multiply(cropped_mask_float, cropped_crop.astype(np.float))
	background = cv2.multiply(1 - cropped_mask_float, cv2.resize(average, (cropped_crop.shape[1], cropped_crop.shape[0])).astype(np.float))
	masked_image_crop = cv2.add(foreground, background).astype(np.uint8)
	
	cropped_mask_float_small = cropped_mask_small.copy().astype(np.float)/255.0
	foreground_small = cv2.multiply(cropped_mask_float_small, cropped_crop.astype(np.float))
	background_small = cv2.multiply(1 - cropped_mask_float_small, cv2.resize(average, (cropped_crop.shape[1], cropped_crop.shape[0])).astype(np.float))
	masked_image_crop_small = cv2.add(foreground_small, background_small).astype(np.uint8)
	
		
	return cropped_crop, cropped_mask, cropped_mask_small, masked_image_crop, masked_image_crop_small, mask_smoothed_dilated_blurred, mask_smoothed_dilated_blurred_small, mask_smoothed, crop_array
	
def preprocess_image_for_db(image):
	image_1 = image[:,:,0]
	image_1_resize = cv2.resize(image_1, (db_size, db_size))
	
	return image_1_resize
	
def get_search_db_from_dir(data_dir):
	image_names = sorted(Path(data_dir).glob('*.png'))
	image_names += sorted(Path(data_dir).glob('**/*.png'))
	image_names += sorted(Path(data_dir).glob('*.jpg'))
	image_names += sorted(Path(data_dir).glob('**/*.jpg'))
	image_names += sorted(Path(data_dir).glob('*.jpeg'))
	image_names += sorted(Path(data_dir).glob('**/*.jpeg'))
	image_names = list(set(image_names))
	
	image_names = [str(i) for i in image_names]
	image_names = image_names[0:100]
	image_db = []
	for image_str in image_names:
		image = cv2.imread(image_str)
		image_1_resize = preprocess_image_for_db(image)
		image_db.append([image_1_resize, image_str])
	return image_db
	

def search_db_with_image(db, image):
	image_1_resize = preprocess_image_for_db(image)
	min_val = None
	min_dist = 1000000
	for image, str in db:
		dist_euclidean = np.linalg.norm(image-image_1_resize)
		if(dist_euclidean <  min_dist):
			min_dist = dist_euclidean
			min_val = [image, str]
	return min_val
		
		
def combine_images_with_mask(image, background_image, mask, un_mask):
	
	smoothed_mask = smooth_mask(mask.copy())
	smoothed_mask_dilated = dilate_mask(smoothed_mask.copy())
	
	if(un_mask is not None):
		un_mask_dilated = dilate_mask(un_mask.copy())
		smoothed_mask_dilated[np.where(un_mask_dilated==[255])] = [0]
	
	smoothed_mask_blurred = blur_mask(smoothed_mask_dilated.copy())
	
	smoothed_mask_blurred_float = smoothed_mask_blurred.copy().astype(np.float)/255.0
	
	print(image.shape, background_image.shape, mask.shape)
	
	
	
	foreground = cv2.multiply(1 - smoothed_mask_blurred_float,  background_image.astype(np.float))
	background = cv2.multiply(smoothed_mask_blurred_float, cv2.resize(image, (background_image.shape[1], background_image.shape[0])).astype(np.float) )
	masked_image = cv2.add(foreground, background).astype(np.uint8)
	
	return masked_image
	
def combine_images_with_mask_no_preprocess(image, background_image, mask):

	smoothed_mask_blurred_float = mask.copy().astype(np.float)/255.0
	foreground = cv2.multiply(1 - smoothed_mask_blurred_float,  background_image.astype(np.float))
	background = cv2.multiply(smoothed_mask_blurred_float, cv2.resize(image, (background_image.shape[1], background_image.shape[0])).astype(np.float) )
	masked_image = cv2.add(foreground, background).astype(np.uint8)
	return masked_image
	
def seamless_clone_image_with_mask(image, insert, mask, unmask, crop_array):

	c = crop_array
# 	mask_crop_array = get_crop_array(mask)
	[x0, x1, y0, y1] = get_crop_array(mask[:,:,0])
	
	diff_y = int((x1 + x0)/2.0 - mask.shape[0]/2.0)
	diff_x = int((y1 + y0)/2.0 - mask.shape[1]/2.0)

	center_tuple = (int(    ((c[3] - c[2])/2.0) + c[2]), int( ((c[1] - c[0])/2.0) + c[0]))
	center_tuple = (center_tuple[0] + diff_x, center_tuple[1] + diff_y)

	outimage = cv2.seamlessClone(insert.copy(), image.copy(), mask.copy() , center_tuple , cv2.NORMAL_CLONE )
	return outimage
	
def combine_image_and_mask_without_seamelss_clone(image, insert, mask, unmask, crop_array):

	c = crop_array
	image_crop = crop_image(image, crop_array)
	if(insert.shape != image_crop.shape):
		insert = cv2.resize(insert, (image_crop.shape[1], image_crop.shape[0]))
	if(mask.shape != image_crop.shape):
		mask = cv2.resize(mask, (image_crop.shape[1], image_crop.shape[0]))
	print(image_crop.shape, insert.shape, mask.shape)
	combined = combine_images_with_mask_no_preprocess(insert, image_crop, mask)
	output_combined = uncrop_image(combined, image, crop_array)
	return output_combined




def convert_68pt_to_5pt(landmarks):
	left_eye_x = round((landmarks[43][0] + landmarks[44][0] + landmarks[46][0] + landmarks[47][0])/4.0)
	left_eye_y = round((landmarks[43][1] + landmarks[44][1] + landmarks[46][1] + landmarks[47][1])/4.0)
	
	right_eye_x = round((landmarks[37][0] + landmarks[38][0] + landmarks[40][0] + landmarks[41][0])/4.0)
	right_eye_y = round((landmarks[37][1] + landmarks[38][1] + landmarks[40][1] + landmarks[41][1])/4.0)
	
	nose_x = round((landmarks[30][0] + landmarks[33][0])/2.0)
	nose_y = round((landmarks[30][1] + landmarks[33][1])/2.0)
	
	nose_x = round((landmarks[30][0]))
	nose_y = round((landmarks[30][1]))            
	
# 	mapped_landmarks = [[left_eye_x, left_eye_y], [right_eye_x, right_eye_y], [nose_x, nose_y], [landmarks[54][0], landmarks[54][1]], [landmarks[48][0], landmarks[48][1]]]
	mapped_landmarks = [[right_eye_x, right_eye_y], [left_eye_x, left_eye_y], [nose_x, nose_y], [landmarks[48][0], landmarks[48][1]], [landmarks[54][0], landmarks[54][1]]]
# 	mapped_landmarks.shape = -1, 5, 2
	return np.array(mapped_landmarks)
	
def align_crop_5pts_opencv(img,
                           crop,
                           mask,
                           src_landmarks,
                           mean_landmarks,
                           align_type='affine',
                           order=3,
                           mode='edge'):
    """Align and crop a face image by 5 landmarks.

    Arguments:
        img             : Face image to be aligned and cropped.
        src_landmarks   : 5 landmarks:
                              [[left_eye_x, left_eye_y],
                               [right_eye_x, right_eye_y],
                               [nose_x, nose_y],
                               [left_mouth_x, left_mouth_y],
                               [right_mouth_x, right_mouth_y]].
        mean_landmarks  : Mean shape, should be normalized in [-0.5, 0.5].
        crop_size       : Output image size.
        face_factor     : The factor of face area relative to the output image.
        landmark_factor : The factor of landmarks' area relative to the face.
        align_type      : 'similarity' or 'affine'.
        order           : The order of interpolation. The order has to be in the range 0-5:
                              - 0: INTER_NEAREST
                              - 1: INTER_LINEAR
                              - 2: INTER_AREA
                              - 3: INTER_CUBIC
                              - 4: INTER_LANCZOS4
                              - 5: INTER_LANCZOS4
        mode            : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                          Points outside the boundaries of the input are filled according
                          to the given mode.
    """
    # set OpenCV
    inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
             3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
    border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
              'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
              'wrap': cv2.BORDER_WRAP}

    # check
    assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant', 'edge', 'symmetric', 'reflect', 'wrap']

    # move
    img_shape_original = img.shape;
    crop_size = img_shape_original[0]
#     move = np.array([img.shape[1] // 2, img.shape[0] // 2])
# 
#     # pad border
#     v_border = img.shape[0] - crop_size
#     w_border = img.shape[1] - crop_size
#     original_src_landmarks = src_landmarks.copy()
# 
#     if v_border < 0:
#         v_half = (-v_border + 1) // 2
#         img = np.pad(img, ((v_half, v_half), (0, 0), (0, 0)), mode=mode)
#         src_landmarks += np.array([0, v_half])
#         move += np.array([0, v_half])
#     if w_border < 0:
#         w_half = (-w_border + 1) // 2
#         img = np.pad(img, ((0, 0), (w_half, w_half), (0, 0)), mode=mode)
#         src_landmarks += np.array([w_half, 0])
#         move += np.array([w_half, 0])

    # estimate transform matrix
    mean_landmarks = np.array(mean_landmarks).astype('float64')
    src_landmarks = np.array(src_landmarks).astype('float64')
#     mean_landmarks -= np.array([mean_landmarks[0, :] + mean_landmarks[1, :]]) / 2.0  # middle point of eyes as center
#     trg_landmarks = mean_landmarks * (crop_size * face_factor * landmark_factor) + move
    trg_landmarks = mean_landmarks
    
    
    print(trg_landmarks, src_landmarks)
    if align_type == 'affine':
        tform = cv2.estimateAffine2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]
        tform_reverse = cv2.estimateAffine2D(src_landmarks, trg_landmarks, ransacReprojThreshold=np.Inf)[0]
    else:
        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]
        tform_reverse = cv2.estimateAffinePartial2D(src_landmarks, trg_landmarks, ransacReprojThreshold=np.Inf)[0]

    # fix the translation to match the middle point of eyes
    trg_mid = (trg_landmarks[0, :] + trg_landmarks[1, :]) / 2.0
    src_mid = (src_landmarks[0, :] + src_landmarks[1, :]) / 2.0
    new_trg_mid = cv2.transform(np.array([[trg_mid]]), tform)[0, 0]
    
    tform[:, 2] += src_mid - new_trg_mid
    tform_reverse[:, 2] -= src_mid - new_trg_mid

    # warp image by given transform
#     output_shape = (crop_size // 2 + move[1] + 1, crop_size // 2 + move[0] + 1)
    output_shape = (img.shape[0], img.shape[1]);

    img_align = cv2.warpAffine(img, tform, output_shape[::-1], flags=cv2.WARP_INVERSE_MAP + inter[order], borderMode=border[mode])
    mask_align = cv2.warpAffine(mask, tform, output_shape[::-1], flags=cv2.WARP_INVERSE_MAP + inter[order], borderMode=border[mode])
    outimage_2 = get_seamless_clone_for_aligned_image_and_mask(crop, img_align, mask_align)
    
    return outimage_2, mask_align
    
    arr1 = np.array([src_landmarks[0], src_landmarks[1], src_landmarks[3], src_landmarks[4]], dtype = "float32")
    arr2 = np.array([trg_landmarks[0], trg_landmarks[1], trg_landmarks[3], trg_landmarks[4]], dtype = "float32")
#     print(arr1, arr2)
#     M = cv2.getPerspectiveTransform(arr1, arr2)
    M = get_avg_trans(src_landmarks, trg_landmarks)
    print(tform)
    img_align_pers = cv2.warpPerspective(img, M, output_shape[::-1], flags=inter[order], borderMode=border[mode])
    mask_align_pers = cv2.warpPerspective(mask, M, output_shape[::-1], flags=inter[order], borderMode=border[mode])
    
    outimage_2_pers = get_seamless_clone_for_aligned_image_and_mask(crop, img_align_pers, mask_align_pers)
    
    outimage_2_comb = get_seamless_clone_for_aligned_image_and_mask(outimage_2_pers, img_align, mask_align)
    
    
    return outimage_2, outimage_2_pers, outimage_2_comb
    
def get_seamless_clone_for_aligned_image_and_mask(crop, img_align, mask_align):
	
	
	img_crop = img_align[0:crop.shape[0], 0:crop.shape[1]]
	mask_crop = mask_align[0:crop.shape[0], 0:crop.shape[1]]
	
    
    
	print(mask_crop.shape)
	print(crop.shape)
	if(mask_crop.shape != crop.shape):
		mask_crop = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]))
		img_crop = cv2.resize(img_crop, (crop.shape[1], crop.shape[0]))
		print("FUCK")
		print(mask_crop.shape)
	
	mask_crop_fore = blur_mask(erode_mask(mask_crop.copy()))
	mask_crop_float = mask_crop_fore.astype(np.float)/255.0
	mask_crop = dilate_mask(mask_crop)
	mask_crop = blur_mask(mask_crop)
	
	foreground = cv2.multiply(mask_crop_float, img_crop.astype(np.float))
	background = cv2.multiply(1 - mask_crop_float, crop.astype(np.float))
	img_return = cv2.add(foreground.astype(np.uint8), background.astype(np.uint8))


	mask_crop_array = get_crop_array(dilate_mask(dilate_mask(dilate_mask(mask_crop)))[:,:,0])
	
	
	mask_y_top_diff = int(mask_crop_array[0])
	mask_y_bot_diff = int((mask_crop.shape[0] - mask_crop_array[1]))
	mask_x_left = int(mask_crop_array[2])
	mask_x_right = int(mask_crop.shape[1] - mask_crop_array[3])
	crop_array = (0, int(crop.shape[0]), 0, int(crop.shape[1]))
	crop_array = list(crop_array)
	crop_array[0] = crop_array[0] + mask_y_top_diff
	crop_array[1] = crop_array[1] - mask_y_bot_diff
	crop_array[2] = crop_array[2] + mask_x_left
	crop_array[3] = crop_array[3] - mask_x_right
	crop_array = tuple(crop_array)
	
	c = crop_array

	
	mask_crop = crop_image(mask_crop, mask_crop_array)
	img_return = crop_image(img_return, mask_crop_array)
	img_crop = crop_image(img_crop, mask_crop_array)

	
    
	center_tuple = (int(    ((c[3] - c[2])/2.0) + c[2] ), int( ((c[1] - c[0])/2.0) + c[0]  ))
#     center_tuple = (int(crop.shape[1]/2.0), int(crop.shape[0]/2.0))
#     outimage = cv2.seamlessClone(img_crop.copy().astype(np.uint8), crop.copy().astype(np.uint8), mask_crop.copy().astype(np.uint8) , center_tuple , cv2.NORMAL_CLONE )
	outimage_2 = cv2.seamlessClone(img_return.copy().astype(np.uint8), crop.copy().astype(np.uint8), mask_crop.copy().astype(np.uint8) , center_tuple , cv2.NORMAL_CLONE )
	return outimage_2
# eyes_search_dir = 'kylie-scrapes-google_crops_aligned_segmented/masks/eyes'
# eyebrows_search_dir = 'kylie-scrapes-google_crops_aligned_segmented/masks/eyebrows'
# nose_search_dir = 'kylie-scrapes-google_crops_aligned_segmented/masks/nose'
# mouth_search_dir = 'kylie-scrapes-google_crops_aligned_segmented/masks/mouth'
# face_search_dir = 'kylie-scrapes-google_crops_aligned_segmented/masks/face'
# 
# 
# eyes_search_db = get_search_db_from_dir(eyes_search_dir)
# print(eyes_search_dir)
# 
# eyebrows_search_db = get_search_db_from_dir(eyebrows_search_dir)
# print(eyebrows_search_dir)
# 
# nose_search_db = get_search_db_from_dir(nose_search_dir)
# print(nose_search_dir)
# 
# mouth_search_db = get_search_db_from_dir(mouth_search_dir)
# print(mouth_search_dir)
# 
# face_search_db = get_search_db_from_dir(face_search_dir)
# print(face_search_dir)
# 
# 
# data_dir = 'touchup-mobile-uploads-camera'
# face_landmark_shape_file = 'shape_predictor_68_face_landmarks.dat'
# 
# num_to_save = 10
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(face_landmark_shape_file)
#     
# 
# img_names = sorted(Path(data_dir).glob('*.png'))
# img_names += sorted(Path(data_dir).glob('**/*.png'))
# img_names += sorted(Path(data_dir).glob('*.jpg'))
# img_names += sorted(Path(data_dir).glob('**/*.jpg'))
# img_names += sorted(Path(data_dir).glob('*.jpeg'))
# img_names += sorted(Path(data_dir).glob('**/*.jpeg'))
# 
# img_names = list(set(img_names))
# img_names = [str(i) for i in img_names]
# 
# 
# 
# 
# random.shuffle(img_names)
# total_images = len(img_names)
# 
# 
# 
# 
# for i, img_name in enumerate(img_names):
# 	img_name = str(img_name)
# 	image = cv2.imread(img_name)
# 	
# 	values = get_landmarks_from_image(image)
# 	if(len(values) == 0):
# 		continue
# 		
# 	landmarks = values[0]
# 	if(landmarks is None):
# 		continue
# 		
# 	bbox_min, bbox_max = get_bounding_box_with_landmarks(landmarks)
# 	if(bbox_min is None):
# 		continue
# 		
# 	crop = crop_image_with_bounding_box(image, bbox_min, bbox_max)
# 	
# 	new_landmarks = get_new_landmarks_with_bounding_box(landmarks, bbox_min, bbox_max)
# 	face_mask = get_face_mask_with_landmarks_and_crop(new_landmarks, crop, bbox_min, bbox_max)
# 	aligned_crop, aligned_mask = align_crop_and_mask_with_landmarks(new_landmarks, crop, face_mask)
# 	return_masks, return_crop_arrays, return_crops = segment_crop_with_face_mask(aligned_crop, aligned_mask)
# 	
# 	if(return_masks is None):
# 		continue
# 		
# 	left_inner_eye_mask = return_masks[7]
# 	right_inner_eye_mask = return_masks[8]
# 	mouth_inner_mask = return_masks[9]
# 	
# 	running_image = aligned_crop.copy()
# 	
# 	
# 	# return_masks.append(left_eye_mask_open_crop)
# # 	return_masks.append(left_eyebrow_mask_open_crop)
# # 	return_masks.append(right_eye_mask_open_crop)
# # 	return_masks.append(right_eyebrow_mask_open_crop)
# # 	return_masks.append(mouth_mask_open_crop)
# # 	return_masks.append(nose_mask_open_crop)
# # 	return_masks.append(face_cut_mask_open_crop)
# # 	return_masks.append(left_inner_eye_mask_open_crop)
# # 	return_masks.append(right_inner_eye_mask_open_crop)
# # 	return_masks.append(mouth_inner_mask_open_crop)
# # 	
# 	for i, (mask, crop_array, crop) in enumerate(zip(return_masks, return_crop_arrays, return_crops)):
# 		search_db = eyes_search_db
# 		inner_mask = left_inner_eye_mask.copy()
# 		if(i == 2):
# 			search_db = eyes_search_db
# 			inner_mask = right_inner_eye_mask.copy()
# 		if(i == 1 or i == 3):
# 			search_db = eyebrows_search_db
# 			inner_mask = None
# 		if(i == 4):
# 			search_db = mouth_search_db
# 			inner_mask = mouth_inner_mask.copy()
# 		if(i == 5):
# 			search_db = nose_search_db
# 			inner_mask = None
# 		if(i == 6):
# 			search_db = face_search_db
# 			inner_mask = None
# 			continue
# 			
# 		[closest_image_mask, closest_string] = search_db_with_image(search_db, mask)
# 		closest_string_original = closest_string.replace('/masks/', '/crops/')
# 		closest_image_crop = cv2.imread(closest_string_original)
# 		k = combine_images_with_mask(closest_image_crop.copy(), crop.copy(), mask.copy(), inner_mask)
# 		running_image = seamless_clone_image_with_mask(running_image, k, mask, None, crop_array)
# 	
# 
# # 	cv2.imshow('closest_image_mask', closest_image_mask)
# # 	cv2.imshow('closest_image_crop', closest_image_crop)
# 	cv2.imshow('running_image', cv2.resize(running_image, (512, 512)))
# # 	cv2.imshow('face_mask', face_mask)
# 	cv2.imshow('aligned_crop', cv2.resize(aligned_crop, (512, 512)))
# # 	cv2.imshow('aligned_mask', aligned_mask)
# 	
# 	cv2.waitKey(0)
# 	
# 	
# 	
# 	
# 		

# with open(json_save_name, 'w') as fp:
#     json.dump(landmarks_dict, fp)

	
