# from main import *
from flask import Flask, request, send_file
from random import randint
import requests, json
import cv2
import flask
from scipy import misc

import requests
from io import StringIO
import uuid

from urllib.request import urlopen

#Import image processing and serving
from PIL import Image, ExifTags, ImageOps
from sklearn.neighbors import BallTree
import scipy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

#Import image processing and serving
from io import StringIO
import base64
from io import BytesIO

from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime
import time
import math



import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tflib as tl
import numpy as np
from run_gans_on_images import  *
from db_utils import *
from face_parsing_utils import * 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  
def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)
    
app = Flask(__name__)
app.secret_key = str(randint(1000, 10000))


import tensorflow as tf
device_name = 'device_name'
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    device_name = tf.test.gpu_device_name()
else:
    print("Please install GPU version of TF")



demo_images = []

demo_images.append("static/2p5k0hlHeC9g.png")
demo_images.append("static/2PMEkUNdiYJT.png")
demo_images.append("static/47q8zHHipRAR.png")
demo_images.append("static/65faFO3zBLDw.png")
demo_images.append("static/5OuYhAx0GX0I.png")
demo_images.append("static/5ntmFRfTe3nF.png")
demo_images.append("static/cdn.cliqueinc.com__cache__posts__260459__short-curly-hairstyles-is-googled-over-60k-times-a-monththese-are-the-best-2820453.700x0c-9581ec97bac34196a60eeeb25dabf601.png")
demo_images.append("static/160510_kristen_stewart_before.jpg")
demo_images.append("static/alexis-bledel-blue-eyes-1363295545.jpg")
demo_images.append("static/blonde-hairstyles-with-bangs-lovely-teen-hairstyles-men-best-30-short-pixie-cuts-for-women-of-blonde-hairstyles-with-bangs.jpg")


#Returns the html index page
@app.route('/')
def index():
	return flask.render_template('index.html', has_result=False, demo_images = demo_images)



def run_oldify_on_image(image, sub_folder_save, intensity):
	
	print('start oldify')
	crop, image_bgr, landmarks, (bbox_min, bbox_max), replicate, border_array = get_crop_and_image_and_landmarks_from_upload(image, sub_folder_save)
	new_landmarks = get_new_landmarks_with_bounding_box(landmarks, padding_left = 0.1, padding_right = 0.1, padding_top = 0.5, padding_bottom = 0.33)
	
	mask_crop_arr = get_masks_from_new_landmarks(new_landmarks, crop.shape[0], crop.shape[1])
	eyes_mask = mask_crop_arr[6]
	full_face_mask = mask_crop_arr[7]
	full_face_mask = final_full_face_big_before_seamless_clone(full_face_mask.copy(), eyes_mask)
	
	crop_array = (bbox_min[1], bbox_max[1], bbox_min[0], bbox_max[0])
	
	old_0 = run_crop_through_oldify_preprocess_with_stgan(crop, '128_stgan_face_3_attrs_8', 'lst_wrinkled', intensity = intensity)
	cyc_old_0 = run_preprocessed_crop_through_cyclegan(old_0, path = 'stgan128_wrinkled_cyclegan_data_1_a_cyclegan_ouutput')

	replicate_cyc_old_0 = seamless_clone_image_with_mask(replicate, cv2.resize(cyc_old_0, (crop.shape[0], crop.shape[1])), full_face_mask, None, crop_array)
	unreplicated_cyc_old_0 = unreplicate_image_with_border_array(replicate_cyc_old_0, border_array)
	
	print('end oldify')
	return unreplicated_cyc_old_0

def run_makeup_on_image(image, sub_folder_save, type):
	
	crop, image_bgr, landmarks, (bbox_min, bbox_max), replicate, border_array = get_crop_and_image_and_landmarks_from_upload(image, sub_folder_save)
	new_landmarks = get_new_landmarks_with_bounding_box(landmarks, padding_left = 0.1, padding_right = 0.1, padding_top = 0.5, padding_bottom = 0.33)
	
	mask_crop_arr = get_masks_from_new_landmarks(new_landmarks, crop.shape[0], crop.shape[1])
	eyes_mask = mask_crop_arr[6]
	full_face_mask = mask_crop_arr[7]
	full_face_mask = final_full_face_big_before_seamless_clone(full_face_mask.copy(), eyes_mask)
	
	
	crop_array = (bbox_min[1], bbox_max[1], bbox_min[0], bbox_max[0])
	
	makeup_natural = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks, type=type)
	cyc_makeup_natural = run_preprocessed_crop_through_cyclegan(makeup_natural, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')

	
	replicate_cyc_makeup_natural = seamless_clone_image_with_mask(replicate, cv2.resize(cyc_makeup_natural, (crop.shape[0], crop.shape[1])), full_face_mask, None, crop_array)
	unreplicated_cyc_makeup_natural = unreplicate_image_with_border_array(replicate_cyc_makeup_natural, border_array)
	return unreplicated_cyc_makeup_natural
	
def run_hair_on_image(image, sub_folder_save, hair_num):

	crop_hair, image_bgr_hair, landmarks_hair, (bbox_min_hair, bbox_max_hair), replicate_hair, border_array_hair = get_crop_and_image_and_landmarks_from_upload(image, sub_folder_save, type='hair')
	crop_array_hair = (bbox_min_hair[1], bbox_max_hair[1], bbox_min_hair[0], bbox_max_hair[0])
	new_landmarks = get_new_landmarks_with_bounding_box(landmarks_hair, padding_left = 0.2, padding_right = 0.2, padding_top = 1.0, padding_bottom = 1.0)
	
	hair_0, hair_mask_align_0 = run_crop_through_hair_preprocess_with_landmarks(crop_hair, landmarks_hair, hair_num=hair_num)
	
	mask_crop_arr = get_masks_from_new_landmarks(new_landmarks, crop_hair.shape[0], crop_hair.shape[1])
	full_face_mask = mask_crop_arr[7]
	full_face_mask = erode_mask(full_face_mask.copy())
	full_face_mask = erode_mask(full_face_mask.copy())
	full_face_mask = erode_mask(full_face_mask.copy())
	full_face_mask = erode_mask(full_face_mask.copy())

	

	hair_mask_align_0 = erode_mask(hair_mask_align_0.copy())
	hair_mask_align_0 = erode_mask(hair_mask_align_0.copy())
	
	hair_mask_align_0 = final_full_face_big_before_seamless_clone(hair_mask_align_0.copy(), full_face_mask)
	
	
	cyc_hair_0 = run_preprocessed_crop_through_cyclegan(hair_0, path = 'hairstyle_cyclegan_data_2_a_cyclegan_output')
	replicate_cyc_hair_0 = seamless_clone_image_with_mask(replicate_hair, cv2.resize(cyc_hair_0, (crop_hair.shape[0], crop_hair.shape[1])), hair_mask_align_0, None, crop_array_hair)
	unreplicated_cyc_hair_0 = unreplicate_image_with_border_array(replicate_cyc_hair_0, border_array_hair)

	return unreplicated_cyc_hair_0


# Called by the HTML web app when a user uploads through index.html
@app.route('/get_face_transformation', methods=['POST'])
def get_face_transformation():
	print("test 1")
	start_time = time.time()
  	  
	time_dict = {}
	time_calc = datetime.now()
    
	image = None
	isFromApp = False;
	
	if request.method == "POST":
		file = request.files['file']
	if not file:
		print("did not find file")
    	
# 	image = Image.open(file).transpose(Image.TRANSPOSE)
	if(file):
		image = Image.open(file)
		try:
			for orientation in ExifTags.TAGS.keys():
				if ExifTags.TAGS[orientation]=='Orientation':
					break
			exif=dict(image._getexif().items())
			if exif[orientation] == 3:
				image=image.rotate(180, expand=True)
			elif exif[orientation] == 6:
				image=image.rotate(270, expand=True)
			elif exif[orientation] == 8:
				image=image.rotate(90, expand=True)
		except Exception as x:
			print("could not exif")
			print(x)
		if(image.mode == 'RGBA'):
			image = image.convert('RGB')
	else:
		image = None
	
	image =  cv2.cvtColor(load_image_into_numpy_array(image.copy()), cv2.COLOR_RGB2BGR)
	if("folder_name" in flask.request.form):
		sub_folder_save = flask.request.form["folder_name"]
	else:
		sub_folder_save = "from_mobile/saved_images/" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "-" + uuid.uuid4().hex
# 	if(sub_folder_save == ""):
# 		sub_folder_save = "from_mobile/saved_images/" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "-" + uuid.uuid4().hex
# 	sub_folder_save = "from_mobile/saved_images/" + "test_1_a"
		if not os.path.exists(sub_folder_save):
			os.makedirs(sub_folder_save, 0o777)

	print("flask.request.form")
	print(flask.request.form)
	filter_name = flask.request.form["filter_name"]
	filter_num = flask.request.form["filter_num"]
	print("time 1: ")
	print((datetime.now() - time_calc))
	time_dict["load_images"] = (datetime.now() - time_calc)
	time_calc = datetime.now()
	
	
	final_image = None
	
	if(filter_name == "oldify"):
		intensity = -1.0;
		if(filter_num == "0"):
			intensity = -0.5
		if(filter_num == "1"):
			intensity = -1.0
		if(filter_num == "2"):
			intensity = -1.5
		final_image = run_oldify_on_image(image, sub_folder_save, intensity)
	if(filter_name == "makeup"):
		makeup_name = "dark"
		if(filter_num == "0"):
			makeup_name = "dark"
		if(filter_num == "1"):
			makeup_name = "natural"
		if(filter_num == "2"):
			makeup_name = "stylish"
		final_image = run_makeup_on_image(image, sub_folder_save, makeup_name)
	if(filter_name == "hair_change"):
		final_image = run_hair_on_image(image, sub_folder_save, int(filter_num))


	# final_image_save_name = sub_folder_save + "/final_image_" + filter_name + "_" + filter_num + ".jpg"
# 	cv2.imwrite(final_image_save_name, final_image)
	
	
	final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
	
	
	
	time_dict["runProcess"] = (datetime.now() - time_calc)
	time_calc = datetime.now()
	
	final_image_pil = Image.fromarray(final_image.astype(np.uint8))
	img_io = BytesIO()
	final_image_pil.save(img_io, 'JPEG', quality=50)
	final_image_str = base64.b64encode(img_io.getvalue()).decode('utf-8')
	

	time_dict["save_images"] = (datetime.now() - time_calc)
	time_calc = datetime.now()

    
	return_data = {}
	return_data["image_string"] = final_image_str
	return_data["folder_name"] = sub_folder_save
	time_dict_return = {}
	for key, value in time_dict.items():
		time_dict_return[key] = value.total_seconds()
	return_data["time_profile"] = time_dict_return
	return_data["device_name"] = device_name
    
	print("\n\n\n")
	print("***TIMES***")
	print(return_data["time_profile"])
	print("\n\n\n")
    
	return flask.jsonify(data=return_data)
	
	
		

	
def test_full_process():
	image = cv2.imread('test-images/queen-prod-mobile-camera-uploads-ios-jpg/03MP3C78gHWk.jpg')
	f = 0.4
	image = cv2.resize(image,None,fx=f,fy=f)
	
# 	sub_folder_save = "from_mobile/saved_images/" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "-" + uuid.uuid4().hex
	sub_folder_save = "from_mobile/saved_images/" + "test_1_z"
	if not os.path.exists(sub_folder_save):
		os.makedirs(sub_folder_save, 0o777)
	

# 	unreplicated_cyc_old_0 = run_oldify_on_image(image, sub_folder_save, -1.0)
	unreplicated_cyc_makeup_natural = run_makeup_on_image(image, sub_folder_save, "natural")
	unreplicated_cyc_makeup_dark = run_makeup_on_image(image, sub_folder_save, "dark")
	unreplicated_cyc_makeup_stylish = run_makeup_on_image(image, sub_folder_save, "stylish")
	
# 	unreplicated_cyc_makeup_red = run_makeup_on_image(image, sub_folder_save, "red")
# 	unreplicated_cyc_makeup_blue = run_makeup_on_image(image, sub_folder_save, "blue")
# 	unreplicated_cyc_makeup_pink = run_makeup_on_image(image, sub_folder_save, "pink")
# 	unreplicated_cyc_makeup_silver = run_makeup_on_image(image, sub_folder_save, "silver")
# 	unreplicated_cyc_makeup_gold = run_makeup_on_image(image, sub_folder_save, "gold")
# 	unreplicated_cyc_makeup_black = run_makeup_on_image(image, sub_folder_save, "black")
# 	unreplicated_cyc_makeup_purple = run_makeup_on_image(image, sub_folder_save, "purple")
# 	unreplicated_cyc_makeup_none = run_makeup_on_image(image, sub_folder_save, "none")
# 	
# 	cv2.imshow('unreplicated_cyc_makeup_red', cv2.resize(unreplicated_cyc_makeup_red, (256, 256)))
# 	cv2.imshow('unreplicated_cyc_makeup_blue', cv2.resize(unreplicated_cyc_makeup_blue, (256, 256)))
# 	cv2.imshow('unreplicated_cyc_makeup_pink', cv2.resize(unreplicated_cyc_makeup_pink, (256, 256)))
# 	cv2.imshow('unreplicated_cyc_makeup_silver', cv2.resize(unreplicated_cyc_makeup_silver, (256, 256)))
# 	cv2.imshow('unreplicated_cyc_makeup_gold', cv2.resize(unreplicated_cyc_makeup_gold, (256, 256)))
# 	cv2.imshow('unreplicated_cyc_makeup_black', cv2.resize(unreplicated_cyc_makeup_black, (256, 256)))
# 	cv2.imshow('unreplicated_cyc_makeup_purple', cv2.resize(unreplicated_cyc_makeup_purple, (256, 256)))
# 	cv2.imshow('unreplicated_cyc_makeup_none', cv2.resize(unreplicated_cyc_makeup_none, (256, 256)))
# 	
# 	cv2.imshow('unreplicated_cyc_makeup_red', unreplicated_cyc_makeup_red)
# 	cv2.imshow('unreplicated_cyc_makeup_blue', unreplicated_cyc_makeup_blue)
# 	cv2.imshow('unreplicated_cyc_makeup_pink', unreplicated_cyc_makeup_pink)
# 	cv2.imshow('unreplicated_cyc_makeup_silver', unreplicated_cyc_makeup_silver)
# 	cv2.imshow('unreplicated_cyc_makeup_gold', unreplicated_cyc_makeup_gold)
# 	cv2.imshow('unreplicated_cyc_makeup_black', unreplicated_cyc_makeup_black)
# 	cv2.imshow('unreplicated_cyc_makeup_purple', unreplicated_cyc_makeup_purple)
# 	cv2.imshow('unreplicated_cyc_makeup_none', unreplicated_cyc_makeup_none)
	
	
# 	unreplicated_cyc_hair_0 = run_hair_on_image(image, sub_folder_save, 1)

# 	cv2.imshow('unreplicated_cyc_old_0', unreplicated_cyc_old_0)
# 	cv2.imshow('unreplicated_cyc_hair_0', unreplicated_cyc_hair_0)
	cv2.imshow('unreplicated_cyc_makeup_natural', unreplicated_cyc_makeup_natural)
	cv2.imshow('unreplicated_cyc_makeup_dark', unreplicated_cyc_makeup_dark)
	cv2.imshow('unreplicated_cyc_makeup_stylish', unreplicated_cyc_makeup_stylish)
	cv2.imshow('image', image)
	cv2.waitKey(0)
	
	
def test_full_process_inline():
	image = cv2.imread('test-images/queen-prod-mobile-camera-uploads-ios-jpg/0C44LGmeg8Dm.jpg')
	f = 0.4
	image = cv2.resize(image,None,fx=f,fy=f)
	
# 	sub_folder_save = "from_mobile/saved_images/" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "-" + uuid.uuid4().hex
	sub_folder_save = "from_mobile/saved_images/" + "test_1_a"
	if not os.path.exists(sub_folder_save):
		os.makedirs(sub_folder_save, 0o777)
	
	crop, image_bgr, landmarks, (bbox_min, bbox_max), replicate, border_array = get_crop_and_image_and_landmarks_from_upload(image, sub_folder_save)
	crop_hair, image_bgr_hair, landmarks_hair, (bbox_min_hair, bbox_max_hair), replicate_hair, border_array_hair = get_crop_and_image_and_landmarks_from_upload(image, sub_folder_save, type='hair')
	
	
	new_landmarks = get_new_landmarks_with_bounding_box(landmarks, padding_left = 0.1, padding_right = 0.1, padding_top = 0.5, padding_bottom = 0.33)
	mask_crop_arr = get_masks_from_new_landmarks(new_landmarks, crop.shape[0], crop.shape[1])
	
	eyes_mask = mask_crop_arr[6]
	full_face_mask = mask_crop_arr[7]
	full_face_mask = final_full_face_big_before_seamless_clone(full_face_mask.copy(), eyes_mask)
	
	crop_array = (bbox_min[1], bbox_max[1], bbox_min[0], bbox_max[0])
	crop_array_hair = (bbox_min_hair[1], bbox_max_hair[1], bbox_min_hair[0], bbox_max_hair[0])

	
	makeup_natural = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks)
	makeup_dark = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks, type="dark")
	makeup_sexy = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks, type="sexy")
	
	makeup_blue = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks, type="blue")
	makeup_red = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks, type="red")
	makeup_pink = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks, type="pink")
	makeup_silver = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks, type="silver")
	makeup_gold = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks, type="gold")
	makeup_black = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks, type="black")
	makeup_purple = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks, type="purple")
	makeup_none = run_crop_through_makeup_preprocess_with_landmarks(crop, landmarks, type="none")
	
	# elif(type == "blue"):
# 		eyeliner_color = blue
# 		eyeshadow_color = blue
# 		lips_color = blue
# 	elif(type == "red"):
# 		eyeliner_color = red
# 		eyeshadow_color = red
# 		lips_color = red
# 	elif(type == "pink"):
# 		eyeliner_color = pink
# 		eyeshadow_color = pink
# 		lips_color = pink
# 	elif(type == "silver"):
# 		eyeliner_color = silver
# 		eyeshadow_color = silver
# 		lips_color = silver
# 	elif(type == "gold"):
# 		eyeliner_color = gold
# 		eyeshadow_color = gold
# 		lips_color = gold
# 	elif(type == "black"):
# 		eyeliner_color = black
# 		eyeshadow_color = black
# 		lips_color = black
# 	elif(type == "purple"):
# 		eyeliner_color = purple
# 		eyeshadow_color = purple
# 		lips_color = purple
# 	elif(type == "none"):
# 		eyeliner_color = None
# 		eyeshadow_color = None
# 		lips_color = None
	
# 	old_0 = run_crop_through_oldify_preprocess_with_stgan(crop, '128_stgan_face_3_attrs_8', 'lst_wrinkled', intensity = -1.0)
# 	old_1 = run_crop_through_oldify_preprocess_with_stgan(crop, '128_stgan_face_3_attrs_8', 'lst_wrinkled', intensity = -1.5)
# 	old_2 = run_crop_through_oldify_preprocess_with_stgan(crop, '128_stgan_face_3_attrs_8', 'lst_wrinkled', intensity = -2.0)
	
# 	hair_0, hair_mask_align_0 = run_crop_through_hair_preprocess_with_landmarks(crop_hair, landmarks_hair, hair_num=0)
# 	hair_1 = run_crop_through_hair_preprocess_with_landmarks(crop_hair, landmarks_hair, hair_num=1)
# 	hair_2 = run_crop_through_hair_preprocess_with_landmarks(crop_hair, landmarks_hair, hair_num=2)
# 	hair_3 = run_crop_through_hair_preprocess_with_landmarks(crop_hair, landmarks_hair, hair_num=3)
# 	hair_4 = run_crop_through_hair_preprocess_with_landmarks(crop_hair, landmarks_hair, hair_num=4)
# 	hair_5 = run_crop_through_hair_preprocess_with_landmarks(crop_hair, landmarks_hair, hair_num=5)
# 	hair_6 = run_crop_through_hair_preprocess_with_landmarks(crop_hair, landmarks_hair, hair_num=6)
# 	hair_7 = run_crop_through_hair_preprocess_with_landmarks(crop_hair, landmarks_hair, hair_num=7)
# 	hair_8 = run_crop_through_hair_preprocess_with_landmarks(crop_hair, landmarks_hair, hair_num=8)

	
# 	cyc_old_0 = run_preprocessed_crop_through_cyclegan(old_0, path = 'stgan128_wrinkled_cyclegan_data_1_a_cyclegan_ouutput')
# 	cyc_old_1 = run_preprocessed_crop_through_cyclegan(old_1, path = 'stgan128_wrinkled_cyclegan_data_1_a_cyclegan_ouutput')
# 	cyc_old_2 = run_preprocessed_crop_through_cyclegan(old_2, path = 'stgan128_wrinkled_cyclegan_data_1_a_cyclegan_ouutput')
	
	cyc_makeup_natural = run_preprocessed_crop_through_cyclegan(makeup_natural, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	cyc_makeup_dark = run_preprocessed_crop_through_cyclegan(makeup_dark, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	cyc_makeup_sexy = run_preprocessed_crop_through_cyclegan(makeup_sexy, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	
	cyc_makeup_blue = run_preprocessed_crop_through_cyclegan(makeup_blue, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	cyc_makeup_red = run_preprocessed_crop_through_cyclegan(makeup_red, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	cyc_makeup_pink = run_preprocessed_crop_through_cyclegan(makeup_pink, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	cyc_makeup_silver = run_preprocessed_crop_through_cyclegan(makeup_silver, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	cyc_makeup_gold = run_preprocessed_crop_through_cyclegan(makeup_gold, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	cyc_makeup_black = run_preprocessed_crop_through_cyclegan(makeup_black, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	cyc_makeup_purple = run_preprocessed_crop_through_cyclegan(makeup_purple, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	cyc_makeup_none = run_preprocessed_crop_through_cyclegan(makeup_none, path = 'stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	
	
# 	cyc_hair_0 = run_preprocessed_crop_through_cyclegan(hair_0, path = 'hairstyle_cyclegan_data_2_a_cyclegan_output')
# 	cyc_hair_1 = run_preprocessed_crop_through_cyclegan(hair_1, path = 'hairstyle_cyclegan_data_2_a_cyclegan_output')
# 	cyc_hair_2 = run_preprocessed_crop_through_cyclegan(hair_2, path = 'hairstyle_cyclegan_data_2_a_cyclegan_output')
# 	cyc_hair_3 = run_preprocessed_crop_through_cyclegan(hair_3, path = 'hairstyle_cyclegan_data_2_a_cyclegan_output')
# 	cyc_hair_4 = run_preprocessed_crop_through_cyclegan(hair_4, path = 'hairstyle_cyclegan_data_2_a_cyclegan_output')
# 	cyc_hair_5 = run_preprocessed_crop_through_cyclegan(hair_5, path = 'hairstyle_cyclegan_data_2_a_cyclegan_output')
# 	cyc_hair_6 = run_preprocessed_crop_through_cyclegan(hair_6, path = 'hairstyle_cyclegan_data_2_a_cyclegan_output')
# 	cyc_hair_7 = run_preprocessed_crop_through_cyclegan(hair_7, path = 'hairstyle_cyclegan_data_2_a_cyclegan_output')
# 	cyc_hair_8 = run_preprocessed_crop_through_cyclegan(hair_8, path = 'hairstyle_cyclegan_data_2_a_cyclegan_output')
	
	
# 	replicate_cyc_makeup_natural = seamless_clone_image_with_mask(replicate, cv2.resize(cyc_makeup_natural, (crop.shape[0], crop.shape[1])), full_face_mask, None, crop_array)
# 	replicate_cyc_old_0 = seamless_clone_image_with_mask(replicate, cv2.resize(cyc_old_0, (crop.shape[0], crop.shape[1])), full_face_mask, None, crop_array)
# 	replicate_cyc_hair_0 = seamless_clone_image_with_mask(replicate_hair, cv2.resize(cyc_hair_0, (crop_hair.shape[0], crop_hair.shape[1])), hair_mask_align_0, None, crop_array_hair)
	
# 	replicate_cyc_makeup_natural_no_clone = combine_image_and_mask_without_seamelss_clone(replicate, cv2.resize(cyc_makeup_natural, (crop.shape[0], crop.shape[1])), full_face_mask, None, crop_array)
# 	replicate_cyc_old_0_no_clone = combine_image_and_mask_without_seamelss_clone(replicate, cv2.resize(cyc_old_0, (crop.shape[0], crop.shape[1])), full_face_mask, None, crop_array)
# 	replicate_cyc_hair_0_no_clone = combine_image_and_mask_without_seamelss_clone(replicate_hair, cv2.resize(cyc_hair_0, (crop_hair.shape[0], crop_hair.shape[1])), hair_mask_align_0, None, crop_array_hair)

# 	unreplicated_cyc_old_0 = unreplicate_image_with_border_array(replicate_cyc_old_0, border_array)
# 	unreplicated_cyc_makeup_natural = unreplicate_image_with_border_array(replicate_cyc_makeup_natural, border_array)
# 	unreplicated_cyc_hair_0 = unreplicate_image_with_border_array(replicate_cyc_hair_0, border_array_hair)
	
# 	unreplicated_cyc_old_0_no_clone = unreplicate_image_with_border_array(replicate_cyc_old_0_no_clone, border_array)
# 	unreplicated_cyc_makeup_natural_no_clone = unreplicate_image_with_border_array(replicate_cyc_makeup_natural_no_clone, border_array)
# 	unreplicated_cyc_hair_0_no_clone = unreplicate_image_with_border_array(replicate_cyc_hair_0_no_clone, border_array_hair)	
	
	
	
	cv2.imshow('crop', cv2.resize(crop, (256, 256)))
# 	cv2.imshow('makeup_natural', cv2.resize(makeup_natural, (256, 256)))
# 	cv2.imshow('makeup_dark', cv2.resize(makeup_dark, (256, 256)))
# 	cv2.imshow('makeup_sexy', cv2.resize(makeup_sexy, (256, 256)))
	cv2.imshow('cyc_makeup_natural', cv2.resize(cyc_makeup_natural, (256, 256)))
	cv2.imshow('cyc_makeup_dark', cv2.resize(cyc_makeup_dark, (256, 256)))
	cv2.imshow('cyc_makeup_sexy', cv2.resize(cyc_makeup_sexy, (256, 256)))

	
	# cv2.imshow('cyc_makeup_red', cv2.resize(cyc_makeup_red, (256, 256)))
# 	cv2.imshow('cyc_makeup_blue', cv2.resize(cyc_makeup_blue, (256, 256)))
# 	cv2.imshow('cyc_makeup_pink', cv2.resize(cyc_makeup_pink, (256, 256)))
# 	cv2.imshow('cyc_makeup_silver', cv2.resize(cyc_makeup_silver, (256, 256)))
# 	cv2.imshow('cyc_makeup_gold', cv2.resize(cyc_makeup_gold, (256, 256)))
# 	cv2.imshow('cyc_makeup_black', cv2.resize(cyc_makeup_black, (256, 256)))
# 	cv2.imshow('cyc_makeup_purple', cv2.resize(cyc_makeup_purple, (256, 256)))
# 	cv2.imshow('cyc_makeup_none', cv2.resize(cyc_makeup_none, (256, 256)))
	
# 	cv2.imshow('cyc_old_0', cv2.resize(cyc_old_0, (256, 256)))
# 	cv2.imshow('cyc_old_1', cv2.resize(cyc_old_1, (256, 256)))
# 	cv2.imshow('cyc_old_2', cv2.resize(cyc_old_2, (256, 256)))
	
# 	cv2.imshow('hair_0', cv2.resize(hair_0, (256, 256)))
# 	cv2.imshow('hair_1', cv2.resize(hair_1, (256, 256)))
# 	cv2.imshow('hair_2', cv2.resize(hair_2, (256, 256)))
# 	cv2.imshow('hair_3', cv2.resize(hair_3, (256, 256)))
# 	cv2.imshow('hair_4', cv2.resize(hair_4, (256, 256)))
# 	cv2.imshow('hair_5', cv2.resize(hair_5, (256, 256)))
# 	cv2.imshow('hair_6', cv2.resize(hair_6, (256, 256)))
# 	cv2.imshow('hair_7', cv2.resize(hair_7, (256, 256)))
# 	cv2.imshow('hair_8', cv2.resize(hair_8, (256, 256)))
	
# 	cv2.imshow('cyc_hair_0', cv2.resize(cyc_hair_0, (256, 256)))
# 	cv2.imshow('cyc_hair_1', cv2.resize(cyc_hair_1, (256, 256)))
# 	cv2.imshow('cyc_hair_2', cv2.resize(cyc_hair_2, (256, 256)))
# 	cv2.imshow('cyc_hair_3', cv2.resize(cyc_hair_3, (256, 256)))
# 	cv2.imshow('cyc_hair_4', cv2.resize(cyc_hair_4, (256, 256)))
# 	cv2.imshow('cyc_hair_5', cv2.resize(cyc_hair_5, (256, 256)))
# 	cv2.imshow('cyc_hair_6', cv2.resize(cyc_hair_6, (256, 256)))
# 	cv2.imshow('cyc_hair_7', cv2.resize(cyc_hair_7, (256, 256)))
# 	cv2.imshow('cyc_hair_8', cv2.resize(cyc_hair_8, (256, 256)))
	
	
# 	cv2.imshow('hair_mask_align_0', cv2.resize(hair_mask_align_0, (256, 256)))
	
# 	cv2.imshow('unreplicated_cyc_old_0', unreplicated_cyc_old_0)
# 	cv2.imshow('unreplicated_cyc_hair_0', unreplicated_cyc_hair_0)
# 	cv2.imshow('unreplicated_cyc_makeup_natural', unreplicated_cyc_makeup_natural)
	cv2.imshow('image', image)
	cv2.waitKey(0)
	
def get_crop_and_image_and_landmarks_from_upload(image_bgr, folder_name, type="not_hair"):
	original_image_str = folder_name + "/original_image.jpg"
	original_crop_str = folder_name + "/original_image_cropped_" + type + ".jpg"
	original_replicate_str = folder_name + "/original_image_replicated_" + type + ".jpg"
	original_landmarks_str = folder_name + "/original_image_landmarks_" + type + ".txt"
	original_bbox_str = folder_name + "/original_image_bbox_" + type + ".txt"
	original_border_array_str = folder_name + "/original_image_border_array_" + type + ".txt"
	
# 	
	all_files_exist = True;
	
	if not os.path.isfile(original_image_str):
		all_files_exist = False
	if not os.path.isfile(original_crop_str):
		all_files_exist = False
	if not os.path.isfile(original_replicate_str):
		all_files_exist = False
	if not os.path.isfile(original_landmarks_str):
		all_files_exist = False
	if not os.path.isfile(original_bbox_str):
		all_files_exist = False
	if not os.path.isfile(original_border_array_str):
		all_files_exist = False
	
	
	if(all_files_exist is True):
		image_bgr = cv2.imread(original_image_str)
		crop = cv2.imread(original_crop_str)
		replicate = cv2.imread(original_replicate_str)
		landmarks = np.loadtxt(original_landmarks_str).astype(int).tolist()
		(bbox_min, bbox_max) = np.loadtxt(original_bbox_str).astype(int).tolist()
		border_array = np.loadtxt(original_border_array_str).astype(int).tolist()
		
		print("SHIT!")
		print(landmarks)
		print((bbox_min, bbox_max))
		print(border_array)
		return crop, image_bgr, landmarks, (bbox_min, bbox_max), replicate, border_array
# 		landmarks = 
	else:
		faces = detector(image_bgr, 1)
		if(len(faces) == 0):
			return None, None, None
		face = faces[0]
		predicted_info = predictor(image_bgr, face)
		detected_landmarks = predicted_info.parts()
		landmarks = [[p.x, p.y] for p in detected_landmarks]
		if(type == "hair"):
			bbox_min, bbox_max, replicate, border_array = get_bounding_box_with_landmarks(landmarks, image_bgr, padding_left = 0.2, padding_right = 0.2, padding_top = 1.0, padding_bottom = 1.0)
		else:
			bbox_min, bbox_max, replicate, border_array = get_bounding_box_with_landmarks(landmarks, image_bgr, padding_left = 0.1, padding_right = 0.1, padding_top = 0.5, padding_bottom = 0.33)
		crop = crop_image_with_bounding_box(replicate, bbox_min, bbox_max)
		
		cv2.imwrite(original_image_str, image_bgr)
		cv2.imwrite(original_crop_str, crop)
		cv2.imwrite(original_replicate_str, replicate)
		
		np.savetxt(original_landmarks_str, landmarks)
		np.savetxt(original_bbox_str, (bbox_min, bbox_max))
		np.savetxt(original_border_array_str, border_array)
		
		print("SHIT!")
		print(landmarks)
		print((bbox_min, bbox_max))
		print(border_array)
		return crop, image_bgr, landmarks, (bbox_min, bbox_max), replicate, border_array
	return None, None, None
	
def run_crop_through_makeup_preprocess_with_landmarks(image, landmarks, type="natural"):
	new_landmarks = get_new_landmarks_with_bounding_box(landmarks, padding_left = 0.1, padding_right = 0.1, padding_top = 0.5, padding_bottom = 0.33)
	
	mask_crop_arr = get_masks_from_new_landmarks(new_landmarks, image.shape[0], image.shape[1])
	
	black_image_both_eyeshadow = mask_crop_arr[0]
	black_image_both_eyeliner = mask_crop_arr[1]
	black_image_lips_outer = mask_crop_arr[2]
# 	black_image_both_eyebrows_smoothed = mask_crop_arr[3]
# 	black_image_inner_mouth = mask_crop_arr[4]
	black_image_full_face = mask_crop_arr[5]
	black_image_both_eyes = mask_crop_arr[6]
	
	black_image_both_eyeshadow = blur_mask(black_image_both_eyeshadow.copy())
	black_image_both_eyeshadow[np.where(black_image_both_eyes==255)] = 0
	black_image_both_eyeshadow = black_image_both_eyeshadow * 1.5
	black_image_both_eyeshadow = np.clip(black_image_both_eyeshadow, a_min = 0, a_max = 255) 
	black_image_both_eyeshadow = black_image_both_eyeshadow.astype(np.uint8)

	
	black_image_both_eyeliner = blur_mask(black_image_both_eyeliner.copy(), isBig=False)

	black_image_lips_outer = blur_mask(black_image_lips_outer.copy(), isBig=False)

	
	red = (177, 244)
	blue = (int(252.0/360.0 * 179.0), int(92.0/100.0 * 255.0))
	pink = (int(288.0/360.0 * 179.0), int(38.5/100.0 * 255.0))
	silver = (int(0.0/360.0 * 179.0), int(0.0/100.0 * 255.0))
	gold = (int(50.6/360.0 * 179.0), int(100/100.0 * 255.0))
	black = (0, 0, 0)
	purple = (int(300.0/360.0 * 179.0), int(100/100.0 * 255.0))
	
	if(type == "natural"):
		eyeliner_color = black
		eyeshadow_color = None
		lips_color = pink
	elif(type == "dark"):
		eyeliner_color = black
		eyeshadow_color = red
		lips_color = red
	elif(type == "stylish"):
		eyeliner_color = black
		eyeshadow_color = gold
		lips_color = purple		
	elif(type == "blue"):
		eyeliner_color = blue
		eyeshadow_color = blue
		lips_color = blue
	elif(type == "red"):
		eyeliner_color = red
		eyeshadow_color = red
		lips_color = red
	elif(type == "pink"):
		eyeliner_color = pink
		eyeshadow_color = pink
		lips_color = pink
	elif(type == "silver"):
		eyeliner_color = silver
		eyeshadow_color = silver
		lips_color = silver
	elif(type == "gold"):
		eyeliner_color = gold
		eyeshadow_color = gold
		lips_color = gold
	elif(type == "black"):
		eyeliner_color = black
		eyeshadow_color = black
		lips_color = black
	elif(type == "purple"):
		eyeliner_color = purple
		eyeshadow_color = purple
		lips_color = purple
	elif(type == "none"):
		eyeliner_color = None
		eyeshadow_color = None
		lips_color = None
	
	

	# eyeliner_color = random.choice(colors)
# 	eyeshadow_color = random.choice(colors)
# 	lips_color = random.choice(colors)
	
	 
	
	smooth_skin_image_raw = cv2.bilateralFilter(image,int(image.shape[0]/32.0),75,75)
	smooth_skin_image_comb = combine_images_with_mask_no_preprocess(smooth_skin_image_raw, image, black_image_full_face)
	
	image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
	image_hsv_eyeliner = image_hsv.copy()
	if(eyeliner_color is not None):
		image_hsv_eyeliner[:,:,0] = int(eyeliner_color[0])
		image_hsv_eyeliner[:,:,1] = int(eyeliner_color[1])
		if(len(eyeliner_color) == 3):
			image_hsv_eyeliner[:,:,2] = eyeliner_color[2]
	image_bgr_eyeliner = cv2.cvtColor(image_hsv_eyeliner, cv2.COLOR_HSV2BGR)
	image_bgr_eyeliner_comb = combine_images_with_mask_no_preprocess(image_bgr_eyeliner, image, black_image_both_eyeliner)
	
	image_hsv_eyeshadow = image_hsv.copy()
	if(eyeshadow_color is not None):
		image_hsv_eyeshadow[:,:,0] = int(eyeshadow_color[0])
		image_hsv_eyeshadow[:,:,1] = int(eyeshadow_color[1])
		if(len(eyeshadow_color) == 3):
			image_hsv_eyeshadow[:,:,2] = eyeshadow_color[2]
	image_bgr_eyeshadow = cv2.cvtColor(image_hsv_eyeshadow, cv2.COLOR_HSV2BGR)
	image_bgr_eyeshadow_comb = combine_images_with_mask_no_preprocess(image_bgr_eyeshadow, image, black_image_both_eyeshadow)


	image_hsv_lips = image_hsv.copy()
	if(lips_color is not None):
		image_hsv_lips[:,:,0] = int(lips_color[0])
		image_hsv_lips[:,:,1] = int(lips_color[1])
		if(len(lips_color) == 3):
			image_hsv_lips[:,:,2] = lips_color[2]
	image_bgr_lips = cv2.cvtColor(image_hsv_lips, cv2.COLOR_HSV2BGR)
	image_bgr_lips_comb = combine_images_with_mask_no_preprocess(image_bgr_lips, image, black_image_lips_outer)
	
	final_image_comb = combine_images_with_mask_no_preprocess(image_bgr_eyeshadow_comb, smooth_skin_image_comb, black_image_both_eyeshadow)
	final_image_comb = combine_images_with_mask_no_preprocess(image_bgr_eyeliner_comb, final_image_comb.copy(), black_image_both_eyeliner)
	final_image_comb = combine_images_with_mask_no_preprocess(image_bgr_lips_comb, final_image_comb.copy(), black_image_lips_outer)
	
	return final_image_comb
	
def run_crop_through_oldify_preprocess_with_stgan(image, gan_path_name, attr_name, intensity = 1.0):
	stgan_image = run_stgan_on_image(image, gan_path_name=gan_path_name, attr_name=attr_name, intensity=intensity)
	return stgan_image
	
	
con = db_connect()
cur = con.cursor() 
known_id = "1194567"
known_marks_string = get_landmarks_string_from_id(con, known_id)
print(known_marks_string)
if(known_marks_string is None):
	print("FUCKING SHIT")
	
known_marks_list = convert_landmark_string_to_list(known_marks_string)

def run_crop_through_hair_preprocess_with_landmarks(image, landmarks, hair_num=None):
	
	hair_names = sorted(Path('face_transformations/hair').glob('hair*.png'))
	hair_names = [str(i) for i in hair_names]
	if(hair_num is None):
		hair_num = random.randrange(len(hair_names))
	
	hair_path = hair_names[hair_num]	
	marks_path = hair_path.replace('/hair_', '/marks_')
	mask_path = hair_path.replace('/hair_', '/mask_')
	
	mask_image = cv2.imread(mask_path)
	hair_image = cv2.imread(hair_path)
	

	f = image.shape[0]/mask_image.shape[0]
	mask_image = cv2.resize(mask_image,None,fx=f,fy=f)
	hair_image = cv2.resize(hair_image,None,fx=f,fy=f)
	
	
	known_new_landmarks = get_new_landmarks_with_bounding_box(known_marks_list, padding_left = 0.2, padding_right = 0.2, padding_top = 1.0, padding_bottom = 1.0)
	known_new_landmarks_5pt = convert_68pt_to_5pt(known_new_landmarks)
	
	new_landmarks = get_new_landmarks_with_bounding_box(landmarks, padding_left = 0.2, padding_right = 0.2, padding_top = 1.0, padding_bottom = 1.0)
# 	new_landmarks = landmarks.copy()
	new_landmarks_5pt = convert_68pt_to_5pt(new_landmarks)
	
	known_new_landmarks_5pt_resized = []
	for k in known_new_landmarks_5pt:
		new_k = [int(k[0]) * f, int(k[1] * f)]
		known_new_landmarks_5pt_resized.append(new_k)
	
	known_new_landmarks_5pt = known_new_landmarks_5pt_resized.copy()
	
	aligned_hair_5, mask_align =  align_crop_5pts_opencv(hair_image,
				   image,
				   mask_image,
				   known_new_landmarks_5pt,
				   new_landmarks_5pt,
				   align_type='affine',
				   order=3,
				   mode='edge')
# 	added_hair, aligned_hair_5, crop = add_hair_to_image_with_landmarks(image, landmarks_list, hair_num, known_marks_list, shouldAddHair)
	
	return aligned_hair_5, mask_align
	
def run_preprocessed_crop_through_cyclegan(crop, path):
	cyclegan_image = run_cyclegan_on_preprocessed_image(crop, path)
	
	return cyclegan_image
	
	
	
	
	

		
	
	

if __name__=="__main__":
# 	app.run(host="0.0.0.0", port=8000)
	load_stgan_with_path(path='128_stgan_face_3_attrs_8')
	reset_default_graph()
	load_cyclegan_with_path(path='stgan128_wrinkled_cyclegan_data_1_a_cyclegan_ouutput')
	reset_default_graph()
	load_cyclegan_with_path(path='stgan128_perfect_makeup_cyclegan_data_3_cyclegan_output')
	reset_default_graph()
	load_cyclegan_with_path(path='hairstyle_cyclegan_data_2_a_cyclegan_output')
	app.run(host="0.0.0.0", port=8000)
	
# 	test_full_process()




