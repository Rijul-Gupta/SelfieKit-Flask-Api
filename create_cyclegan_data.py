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


import cv2
import numpy as np
import random


from test_multi_stgan import generate_session_and_graph_model_2, generate_image_from_crop_stgan_from_preload_2
from test_multi_stgan import  generate_image_from_crop_with_fullstring, generate_image_from_crop_with_fullstring_baseimage

import tflib as tl
import tensorflow as tf

 
# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
 
# Draw a point
def draw_point(img, p, color ) :
#     cv2.circle( img, tuple(p), 2, color, cv2.cv.CV_FILLED, cv2.LINE_AA, 0 )
    cv2.circle(img, tuple(p), 2, color, thickness=1, lineType=8, shift=0)

 
 
# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
         
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
 

		
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
    left_eye_img = img.copy()
    right_eye_img = img.copy()
    mouth_img = img.copy()
    other_img = img.copy()
    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
         
        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = (int( i/len(facets) * 255.0), 255 - int( i/len(facets) * 255.0), int( i/len(facets) * 255.0))
        
        this_img = other_img

        
        if(i in mouth_numbers):
        	this_img = mouth_img
        if(i in left_eye_numbers):
        	this_img = left_eye_img
        if(i in right_eye_numbers):
        	this_img = right_eye_img
        if(i in face_numbers):
        	this_img = other_img
        	
        ifacets = np.array([ifacet])
        
        cv2.fillConvexPoly(this_img, ifacet, (255,255,255), cv2.LINE_AA, 0);
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 1,  (0, 0, 0), thickness=1, lineType=8, shift=0)
        
        
    return [left_eye_img, right_eye_img, mouth_img, other_img]

def erode_mask(img, type="big"):
	kernel_size_extrasmall = int(math.ceil(max(img.shape)/128.0) * 1.0) + 1
	kernel_size_small = int(math.ceil(max(img.shape)/128.0) * 2.0) + 1
	kernel_size_big = int(math.ceil(max(img.shape)/128.0) * 6.0) + 1
	kernel_extrasmall = np.ones((kernel_size_extrasmall,kernel_size_extrasmall),np.uint8)
	kernel_small = np.ones((kernel_size_small,kernel_size_small),np.uint8)
	kernel_big = np.ones((kernel_size_big,kernel_size_big),np.uint8)
	
	
	if(type == "small"):
		kernel_big = kernel_small
	if(type == "extrasmall"):
		kernel_big = kernel_extrasmall

	
	bordersize = 100
	original_height = img.shape[0]
	original_width = img.shape[1]
	img = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	img = cv2.erode(img,kernel_big,iterations = 1)
	img = img[bordersize:(original_height + bordersize), bordersize:(original_width + bordersize)]
	return img
	

def dilate_mask(img):
	kernel_size_small = int(math.ceil(max(img.shape)/128.0) * 2.0) + 1
	kernel_size_big = int(math.ceil(max(img.shape)/128.0) * 6.0) + 1
	kernel_small = np.ones((kernel_size_small,kernel_size_small),np.uint8)
	kernel_big = np.ones((kernel_size_big,kernel_size_big),np.uint8)

	bordersize = 100
	original_height = img.shape[0]
	original_width = img.shape[1]
	img = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	img = cv2.dilate(img,kernel_small,iterations = 1)
	img = img[bordersize:(original_height + bordersize), bordersize:(original_width + bordersize)]
	return img
	
def blur_mask(img):
	kernel_size_small = int(math.ceil(max(img.shape)/128.0) * 2.0) + 1
	kernel_size_big = int(math.ceil(max(img.shape)/128.0) * 6.0) + 1
	kernel_small = np.ones((kernel_size_small,kernel_size_small),np.uint8)
	kernel_big = np.ones((kernel_size_big,kernel_size_big),np.uint8)
	bordersize = 100
	original_height = img.shape[0]
	original_width = img.shape[1]
	img = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	
	img = cv2.blur(img,(kernel_size_small,kernel_size_small))
	img = cv2.blur(img,(kernel_size_small,kernel_size_small))
	img = cv2.blur(img,(kernel_size_small,kernel_size_small))
# 	img = cv2.blur(img,(kernel_size_big,kernel_size_big))
# 	img = cv2.blur(img,(kernel_size_big,kernel_size_big))
# 	img = cv2.blur(img,(kernel_size_big,kernel_size_big))

	img = img[bordersize:(original_height + bordersize), bordersize:(original_width + bordersize)]
	return img
	
def smooth_mask(img):

	kernel_size_small = int(math.ceil(max(img.shape)/128.0) * 2.0) + 1
	kernel_size_big = int(math.ceil(max(img.shape)/128.0) * 6.0) + 1
	kernel_small = np.ones((kernel_size_small,kernel_size_small),np.uint8)
	kernel_big = np.ones((kernel_size_big,kernel_size_big),np.uint8)
	bordersize = 100
	original_height = img.shape[0]
	original_width = img.shape[1]
	img = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	
	img = cv2.blur(img,(kernel_size_small,kernel_size_small))
	img = cv2.blur(img,(kernel_size_small,kernel_size_small))
	img = cv2.blur(img,(kernel_size_small,kernel_size_small))
	img = cv2.blur(img,(kernel_size_big,kernel_size_big))
	img = cv2.blur(img,(kernel_size_big,kernel_size_big))
	img = cv2.blur(img,(kernel_size_big,kernel_size_big))

	
	lower_red = np.array([64,64,64])
	upper_red = np.array([255,255,255])
		
	img_mask = cv2.inRange(img, lower_red, upper_red)
	img[np.where(img_mask==0)] = [0,0,0]
	img[np.where(img_mask==255)] = [255, 255, 255]
		
	
# 	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_small, iterations = 1)
# 	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_small, iterations = 1)
# 	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_small, iterations = 1)
# 	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_big, iterations = 1)
# 	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_big, iterations = 1)
# 	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_big, iterations = 1)
	img = img[bordersize:(original_height + bordersize), bordersize:(original_width + bordersize)]
	return img
	
def mask_image(img, mask):
	gray_1024 = np.zeros(img.shape)
	mask_val = mask.astype(np.float)/255.0	
	foreground2 = cv2.multiply(mask_val, img.astype(np.float))
	background2 = cv2.multiply(1 - mask_val, gray_1024.astype(np.float))
	masked_image2 = cv2.add(foreground2, background2).astype(np.uint8)
	return masked_image2


def reshape_for_polyline(array):
    """Reshape image so that it works with polyline."""
    return np.array(array, np.int32).reshape((-1, 1, 2))


def get_crop_array(image,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = image > 0

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    return [x0, x1, y0, y1]
    
def crop_image(image, array):
	return image[array[0]:array[1], array[2]:array[3]]
	
def uncrop_image(image, original, array):
	original[array[0]:array[1], array[2]:array[3]] = image
	return original
	
def get_stgan_inputs(crop, landmarks):
	bbox_min = np.min(landmarks, axis=0)
	bbox_max = np.max(landmarks, axis=0)
	crop_width = bbox_max[0] - bbox_min[0]
	crop_height = bbox_max[1] - bbox_min[1]
	
	bbox_min[0] = bbox_min[0] - int(crop_width*padding)
	bbox_min[1] = bbox_min[1] - int(crop_height*padding)
	
	bbox_max[0] = bbox_max[0] + int(crop_width*padding)
	bbox_max[1] = bbox_max[1] + int(crop_height*padding)
	

	if((bbox_max[1] - bbox_min[1]) > crop.shape[0]):
		bbox_max[1] = bbox_min[1] + crop.shape[0]
	if((bbox_max[0] - bbox_min[0]) > crop.shape[1]):
		bbox_max[0] = bbox_min[0] + crop.shape[1]

	
	crop_width_padded = bbox_max[0] - bbox_min[0] - 1
	crop_height_padded = bbox_max[1] - bbox_min[1] - 1
	
	
	
	new_landmarks = []
	for l in landmarks:
		new_lx = int(l[0] - bbox_min[0])
		new_ly = int(l[1] - bbox_min[1])
		new_landmarks.append([new_lx, new_ly])
	
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

	
	bottom_left_eye = list((new_landmarks[45:48] + [new_landmarks[42]]))
	left_eyebrows_and_bottom_eye = (left_eyetopmid + bottom_left_eye)
	
	bottom_right_eye = list((new_landmarks[39:42] + [new_landmarks[36]]))
	right_eyebrows_and_bottom_eye = (right_eyetopmid + bottom_right_eye)
	right_eyetop_and_bottom_eye = ([new_landmarks[0]] + list(reversed(right_eyetop)))
	left_eyetop_and_bottom_eye = ([new_landmarks[16]] + list((left_eyetop)))

	full_face = (new_landmarks[0:17] + list(reversed(left_eyetop)) + list(reversed(right_eyetop)))

	black_image_left_eye = np.zeros(((bbox_max[1] - bbox_min[1]), (bbox_max[0] - bbox_min[0]), 3), np.uint8)
	black_image_right_eye = black_image_left_eye.copy()
	black_image_full_face = black_image_left_eye.copy()
	black_image_right_eyetop = black_image_left_eye.copy()
	
	poly_left_eye = reshape_for_polyline(left_eyebrows_and_bottom_eye)
	poly_right_eye = reshape_for_polyline(right_eyebrows_and_bottom_eye)
	poly_full_face = reshape_for_polyline(full_face)
	poly_right_eyetop = reshape_for_polyline(right_eyetop_and_bottom_eye)
	poly_left_eyetop = reshape_for_polyline(left_eyetop_and_bottom_eye)

	
	cv2.fillConvexPoly(black_image_left_eye, poly_left_eye, (255, 255, 255))
	cv2.fillConvexPoly(black_image_right_eye, poly_right_eye, (255, 255, 255))
	cv2.fillConvexPoly(black_image_full_face, poly_full_face, (255, 255, 255))
	cv2.fillConvexPoly(black_image_full_face, poly_right_eyetop, (255, 255, 255))
	cv2.fillConvexPoly(black_image_full_face, poly_left_eyetop, (255, 255, 255))

# 		gray_1024 = np.zeros(crop.shape)
# 		mask_val = black_image.astype(np.float)/255.0	
# 		foreground2 = cv2.multiply(mask_val, crop.astype(np.float))
# 		background2 = cv2.multiply(1 - mask_val, gray_1024.astype(np.float))
# 		masked_image2 = cv2.add(foreground2, background2).astype(np.uint8)
# 		
	
	
	
	#voronoi
	
	new_landmarks_added = new_landmarks + left_eyebutt + left_eyetop + right_eyebutt + right_eyetop
	
	# Read in the image.
	img = crop.copy()
	
	size = img.shape
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
	[left_eye_img_v, right_eye_img_v, mouth_img_v, other_img_v] = draw_voronoi(img_voronoi,subdiv)
	
	
	lower_red = np.array([128,128,128])
	upper_red = np.array([255,255,255])
	
	left_v_mask = cv2.inRange(left_eye_img_v, lower_red, upper_red)
	left_eye_img_v[np.where(left_v_mask==0)] = [0,0,0]
	left_eye_img_v[np.where(left_v_mask==255)] = [255, 255, 255]
	cv2.fillConvexPoly(left_eye_img_v, poly_left_eye, (255, 255, 255))
	left_eye_img_v = mask_image(left_eye_img_v.copy(), black_image_full_face.copy())
	
	
	
	right_v_mask = cv2.inRange(right_eye_img_v, lower_red, upper_red)
	right_eye_img_v[np.where(right_v_mask==0)] = [0,0,0]
	right_eye_img_v[np.where(right_v_mask==255)] = [255, 255, 255]
	cv2.fillConvexPoly(right_eye_img_v, poly_right_eye, (255, 255, 255))
	right_eye_img_v = mask_image(right_eye_img_v.copy(), black_image_full_face.copy())
	
	mouth_v_mask = cv2.inRange(mouth_img_v, lower_red, upper_red)
	mouth_img_v[np.where(mouth_v_mask==0)] = [0,0,0]
	mouth_img_v[np.where(mouth_v_mask==255)] = [255, 255, 255]
	mouth_img_v = mask_image(mouth_img_v.copy(), black_image_full_face.copy())
	
	left_eye_img_v_smoothed = smooth_mask(left_eye_img_v.copy())
	right_eye_img_v_smoothed = smooth_mask(right_eye_img_v.copy())
	mouth_img_v_smoothed = smooth_mask(mouth_img_v.copy())
	

	left_eye_img_v_smoothed_dilated = dilate_mask(left_eye_img_v_smoothed.copy())
	right_eye_img_v_smoothed_dilated = dilate_mask(right_eye_img_v_smoothed.copy())
	mouth_img_v_smoothed_dilated = dilate_mask(mouth_img_v_smoothed.copy())
	
	left_eye_img_v_smoothed_eroded = erode_mask(left_eye_img_v_smoothed.copy())
	right_eye_img_v_smoothed_eroded = erode_mask(right_eye_img_v_smoothed.copy())
	mouth_img_v_smoothed_eroded = erode_mask(mouth_img_v_smoothed.copy())
	
	kernel_size = int(math.ceil(max(img.shape)/128.0) * 4.0) + 1
	kernel = np.ones((kernel_size,kernel_size),np.uint8)


	black_image_full_face_smoothed = smooth_mask(black_image_full_face.copy())
	black_image_full_face_smoothed_dilated = dilate_mask(black_image_full_face_smoothed.copy())
	black_image_full_face_smoothed_dilated_cut = black_image_full_face_smoothed_dilated.copy()
	black_image_full_face_smoothed_dilated_cut[np.where(left_eye_img_v_smoothed_eroded==[255])] = [0]
	black_image_full_face_smoothed_dilated_cut[np.where(right_eye_img_v_smoothed_eroded==[255])] = [0]
	black_image_full_face_smoothed_dilated_cut[np.where(mouth_img_v_smoothed_eroded==[255])] = [0]
	
	

	mask_crop_arr = []
	mask_crop_arr.append(left_eye_img_v_smoothed_dilated)
	mask_crop_arr.append(right_eye_img_v_smoothed_dilated)
	mask_crop_arr.append(mouth_img_v_smoothed_dilated)
	mask_crop_arr.append(black_image_full_face_smoothed_dilated_cut)
	output_masked_arr = []
	masks_arr = []
	cropped_arr_arr = []
	for mask_org in mask_crop_arr:
		mask = mask_org.copy()
		img = crop.copy()
	
		cropped_array = get_crop_array(mask[:,:,0]);
		cropped_img = crop_image(img, cropped_array)
		cropped_mask = crop_image(mask, cropped_array)	


		mask_val = cropped_mask.astype(np.float)/255.0
		gray_1024 = cv2.resize(average_image, (cropped_img.shape[1], cropped_img.shape[0]))

		foreground2 = cv2.multiply(mask_val, cropped_img.astype(np.float))
		background2 = cv2.multiply(1 - mask_val, gray_1024.astype(np.float))
		masked_image2 = cv2.add(foreground2, background2).astype(np.uint8)
		output_masked_arr.append(masked_image2.copy())
		masks_arr.append(cropped_mask.copy())
		cropped_arr_arr.append(cropped_array)
	return output_masked_arr, masks_arr, cropped_arr_arr

		# Show results

	# 		cv2.imshow('left_eye_img_v', left_eye_img_v)
	# 		cv2.imshow('right_eye_img_v', right_eye_img_v)
	# 		cv2.imshow('mouth_img_v', mouth_img_v)
	
	# 		cv2.imshow('left_eye_img_v_smoothed', left_eye_img_v_smoothed)
	# 		cv2.imshow('right_eye_img_v_smoothed', right_eye_img_v_smoothed)
	# 		cv2.imshow('mouth_img_v_smoothed', mouth_img_v_smoothed)
	# 		cv2.imshow('black_image_left_eye', black_image_left_eye)
	# 		cv2.imshow('black_image_right_eye', black_image_right_eye)
	# 		cv2.imshow('black_image_full_face', black_image_full_face)
	# 		if('.' in key):
	# 			left_eye_masked = mask_image(crop.copy(), left_eye_img_v_smoothed_dilated)
	# 			right_eye_masked = mask_image(crop.copy(), right_eye_img_v_smoothed_dilated)
	# 			mouth_masked = mask_image(crop.copy(),  mouth_img_v_smoothed_dilated)
	# 			face_masked = mask_image(crop.copy(),  black_image_full_face_smoothed_dilated_cut)
	# 
	# 			cv2.imshow('black_image_full_face_smoothed_dilated', black_image_full_face_smoothed_dilated)
	# 			cv2.imshow('black_image_full_face_smoothed_dilated_cut', black_image_full_face_smoothed_dilated_cut)
	# 			cv2.imshow('crop', crop)
	# 			cv2.imshow('left_eye_masked', left_eye_masked)
	# 			cv2.imshow('right_eye_masked', right_eye_masked)
	# 			cv2.imshow('mouth_masked', mouth_masked)
	# 			cv2.imshow('face_masked', face_masked)
	# 			cv2.waitKey(0)
	# 		

	# 		cv2.imwrite(left_eye_mask_save_name, left_eye_img_v_smoothed_dilated)
	# 		cv2.imwrite(right_eye_mask_save_name, right_eye_img_v_smoothed_dilated)
	# 		cv2.imwrite(mouth_mask_save_name, mouth_img_v_smoothed_dilated)
	# 		cv2.imwrite(face_mask_save_name, black_image_full_face_smoothed_dilated_cut)

		# cv2.imwrite(crop_save_name, crop)
	# 		cv2.imwrite(mask_save_name, black_crop)






args_dict = {}
gan_dict = {}

gan_path_name_eyes = '128_stgan_baddies_masked_eyes'
[gan_sess_model_4, gan_xa_sample_model_4, gan__b_sample_model_4, gan_raw_b_sample_model_4,  gan_x_sample_model_4,  D, xa_logit_gan, xa_logit_att] =  generate_session_and_graph_model_2(model_path=gan_path_name_eyes)
gan_dict[gan_path_name_eyes] = [gan_sess_model_4, gan_xa_sample_model_4, gan__b_sample_model_4, gan_raw_b_sample_model_4,  gan_x_sample_model_4]
with open('./face-cropping_output/%s/setting.txt' % gan_path_name_eyes) as f:
	args = json.load(f)
	args_dict[gan_path_name_eyes] = args

tf.reset_default_graph()
gan_path_name_mouth = '128_stgan_baddies_masked_mouth_rec10'
[gan_sess_model_4, gan_xa_sample_model_4, gan__b_sample_model_4, gan_raw_b_sample_model_4,  gan_x_sample_model_4,  D, xa_logit_gan, xa_logit_att] =  generate_session_and_graph_model_2(model_path=gan_path_name_mouth)
gan_dict[gan_path_name_mouth] = [gan_sess_model_4, gan_xa_sample_model_4, gan__b_sample_model_4, gan_raw_b_sample_model_4,  gan_x_sample_model_4]
with open('./face-cropping_output/%s/setting.txt' % gan_path_name_mouth) as f:
	args = json.load(f)
	args_dict[gan_path_name_mouth] = args

tf.reset_default_graph()
gan_path_name_face = '128_stgan_baddies_masked_face'
[gan_sess_model_4, gan_xa_sample_model_4, gan__b_sample_model_4, gan_raw_b_sample_model_4,  gan_x_sample_model_4,  D, xa_logit_gan, xa_logit_att] =  generate_session_and_graph_model_2(model_path=gan_path_name_face)
gan_dict[gan_path_name_face] = [gan_sess_model_4, gan_xa_sample_model_4, gan__b_sample_model_4, gan_raw_b_sample_model_4,  gan_x_sample_model_4]
with open('./face-cropping_output/%s/setting.txt' % gan_path_name_face) as f:
	args = json.load(f)
	args_dict[gan_path_name_face] = args

#calculate average image
average_image = cv2.imread('baddies4_average.png')


image_dir = 'face-cropping_cyclegan_data/original/baddies7'
output_dir = image_dir.replace('original', 'processed2')

image_files = []
image_files += sorted(Path(image_dir).glob('*.png'))
image_files += sorted(Path(image_dir).glob('**/*.png'))
image_files += sorted(Path(image_dir).glob('*.jpg'))
image_files += sorted(Path(image_dir).glob('**/*.jpg'))
image_files += sorted(Path(image_dir).glob('*.jpeg'))
image_files += sorted(Path(image_dir).glob('**/*.jpeg'))
image_files = list(set(image_files))
image_files = [str(path) for path in image_files]

# os.path.basename(your_path)

existing_files = []
existing_files += sorted(Path(output_dir).glob('*.png'))
existing_files += sorted(Path(output_dir).glob('**/*.png'))
existing_files += sorted(Path(output_dir).glob('*.jpg'))
existing_files += sorted(Path(output_dir).glob('**/*.jpg'))
existing_files += sorted(Path(output_dir).glob('*.jpeg'))
existing_files += sorted(Path(output_dir).glob('**/*.jpeg'))
existing_files = list(set(existing_files))
existing_files = [str(path) for path in existing_files]


existing_files_basenames = [os.path.basename(str(path)) for path in existing_files]
existing_files_basenames = set(existing_files_basenames)

image_files_copy = image_files.copy()

print("image files length: ")
print(len(image_files))

print("existing files length: ")
print(len(existing_files_basenames))


image_files_copy = image_files.copy()
for i, file in enumerate(image_files_copy):
	if(i % 1000 == 0):
		print(i)
	basename = os.path.basename(file)
	if basename in existing_files_basenames:
		image_files.remove(file)
	
json_save_name = '176K_all_landmarks.json'
landmarks_dict = {}
if(os.path.exists(json_save_name)):
	with open(json_save_name, 'r') as fp:
		landmarks_dict = json.load(fp)
landmarks_dict_basenames = {}
for key, value in 	landmarks_dict.items():
	for i, v in enumerate(value):
		file_base, file_ext = os.path.splitext(key)
		new_key = file_base.replace("faces/", "").replace("/", "_") + '_' + str(i) + file_ext
		landmarks_dict_basenames[new_key] = v
dec_time = time.time()



from random import shuffle

shuffle(image_files)

padding = 0.05;
crop_size_thresh = 128;
image_size_thresh = 512.0;
final_image_size = 512.0;

for key in image_files:
	if("/baddie/" in key):
		
		edited_image_save_name = key.replace(image_dir, output_dir)


		if os.path.isfile(edited_image_save_name):
			print("edited exists, skipping")
			continue
			
		crop = cv2.imread(key)
		if(min(crop.shape[0], crop.shape[1]) < image_size_thresh):
			print("image too small, skipping")
			continue
			
		if not os.path.exists(os.path.dirname(edited_image_save_name)):
			try:
				os.makedirs(os.path.dirname(edited_image_save_name))
			except OSError as exc: # Guard against race condition
				print(exc)
		resize_factor = min(1.0, final_image_size/max(crop.shape[0], crop.shape[1]))
		crop_resized = cv2.resize(crop, None, fx=resize_factor, fy=resize_factor, interpolation = cv2.INTER_AREA)	
		cv2.imwrite(edited_image_save_name,crop_resized)
		
		
		continue
	continue
	key_basename = os.path.basename(key)
	first_under = key_basename.split("_")[0]
	if(first_under.isdigit()):
		new_base_name = '_'.join(key_basename.split("_")[1:])
		key_basename = new_base_name
	value = landmarks_dict_basenames[key_basename]
	for i, landmarks in enumerate([value]):
		
		file_base, file_ext = os.path.splitext(key)
		
		edited_image_save_name = key.replace(image_dir, output_dir)


		if os.path.isfile(edited_image_save_name):
			print("edited exists, skipping")
			continue
		
		crop = cv2.imread(key)
		
		if(min(crop.shape[0], crop.shape[1]) < image_size_thresh):
			print("image too small, skipping")
			continue
		
		masked_images_arr, masks_arr, cropped_arr_arr = get_stgan_inputs(crop, landmarks)
# 		masked_eye_left = masked_images_arr[0]
# 		masked_eye_right = masked_images_arr[1]
# 		masked_mouth = masked_images_arr[2]
# 		masked_face = masked_images_arr[3]
		
		gan_paths = [gan_path_name_eyes, gan_path_name_eyes, gan_path_name_mouth, gan_path_name_face]
		ganned_images = []
		original_images = []
		for i, masked_eye_left in enumerate(masked_images_arr):
			gan_path_name = gan_paths[i]
			[sess, xa_sample, _b_sample, raw_b_sample, x_sample] = gan_dict[gan_path_name]
			args = args_dict[gan_path_name]
			atts = args['atts']
			
			print(args['experiment_name'])
			cropped_masked_image_rgb = cv2.cvtColor(masked_eye_left.copy(), cv2.COLOR_BGR2RGB)
			cropped_masked_image_rgb_tensor = tf.convert_to_tensor(np.asarray([cropped_masked_image_rgb.copy()]), np.float32)
			cropped_masked_image_rgb_tensor_resized = tf.image.resize_images(cropped_masked_image_rgb_tensor, (128, 128), tf.image.ResizeMethod.AREA)
			cropped_masked_image_rgb_tensor_resized_clipped = tf.clip_by_value(cropped_masked_image_rgb_tensor_resized, 0, 255) / 127.5 - 1
			with tf.Session() as sess_fuck:
				resized_image = cropped_masked_image_rgb_tensor_resized_clipped.eval()
	# 		resized_image = sess.run(cropped_masked_image_rgb_tensor_resized_clipped)
	

			xa_sample_ipt = resized_image
			b_sample_ipt = np.zeros(shape=(1,len(atts)))
			_b_sample_ipt = b_sample_ipt
			_b_sample_ipt_copy = _b_sample_ipt.copy()
			_b_sample_ipt_copy[..., atts.index(atts[0])] = -1.0
			_b_sample_ipt_copy[..., atts.index(atts[1])] = +1.0
			
# 			if(i == 2):
# 				_b_sample_ipt_copy[..., atts.index(atts[0])] = -2.0

			sess_result = sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
																	   _b_sample: _b_sample_ipt,
																	   raw_b_sample: _b_sample_ipt_copy})
																   
																   
			sess_return = sess_result.copy().squeeze(0)      
			return_image = (sess_return + 1.0) * 127.5;

	
	
	# 	Prepare cyclegan input
			final_return = cv2.cvtColor(return_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
			final_return_resized = cv2.resize(final_return, (masked_eye_left.shape[1], masked_eye_left.shape[0]))
			ganned_images.append(final_return_resized.copy())
			original_images.append(masked_eye_left.copy())
# 			cv2.imshow('final_return', final_return)
# 			cv2.waitKey(0)
		
		black = np.zeros(crop.shape)
		uncropped_gan_images = []
		uncropped_masks = []
		running_image = crop.copy()
		for num, (i, m, g, c, o) in enumerate(zip(masked_images_arr, masks_arr, ganned_images, cropped_arr_arr, original_images)):
# 			uncropped_i = uncrop_image(i.copy(), crop.copy(), c)
			
			uncropped_m = uncrop_image(m.copy(), black.copy(), c)
			uncropped_g = uncrop_image(g.copy(), crop.copy(), c)
# 			uncropped_o = uncrop_image(o.copy(), crop.copy(), c)
			eroded_mask = erode_mask(uncropped_m.copy())
			eroded_mask_extrasmall = erode_mask(uncropped_m.copy(), "extrasmall")
			
			blurred_eroded_mask = eroded_mask.copy()
			
			blurred_eroded_mask = blur_mask(blurred_eroded_mask.copy())
			blurred_eroded_mask = blur_mask(blurred_eroded_mask.copy())
			blurred_eroded_mask = blurred_eroded_mask.copy().astype(np.uint8)
			
			
			mask_val = eroded_mask_extrasmall.copy().astype(np.float)/255.0
			foreground2 = cv2.multiply(mask_val, uncropped_g.astype(np.float))
			background2 = cv2.multiply(1 - mask_val, crop.astype(np.float))
			masked_image2 = cv2.add(foreground2, background2).astype(np.uint8)
			uncropped_gan_images.append(masked_image2)
			uncropped_masks.append(uncropped_m)

			mask_val = blurred_eroded_mask.astype(np.float)/255.0
			
			foreground3 = cv2.multiply(mask_val, masked_image2.astype(np.float))
			background3 = cv2.multiply(1 - mask_val, running_image.astype(np.float))
			masked_image3 = cv2.add(foreground3, background3).astype(np.uint8)
			running_image = masked_image3.copy()
			
		
		if not os.path.exists(os.path.dirname(edited_image_save_name)):
			try:
				os.makedirs(os.path.dirname(edited_image_save_name))
			except OSError as exc: # Guard against race condition
				print(exc)
		resize_factor = min(1.0, final_image_size/max(masked_image3.shape[0], masked_image3.shape[1]))
		masked_image3_resized = cv2.resize(masked_image3, None, fx=resize_factor, fy=resize_factor)	
		cv2.imwrite(edited_image_save_name,masked_image3_resized)
# 			
# 			cv2.imshow('uncropped_o', uncropped_o)		
# 			cv2.imshow('uncropped_g', uncropped_g)
# 			cv2.imshow('eroded_mask', eroded_mask)
# 			cv2.imshow('blurred_eroded_mask', blurred_eroded_mask)
# 			cv2.imshow('crop', crop)
# 			cv2.imshow('masked_image2', masked_image2)
# 			cv2.imshow('masked_image3', masked_image3)
# 			cv2.waitKey(0)
		
		



	
