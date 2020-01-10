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
from datetime import datetime

import cv2
import numpy as np
import random
from PIL import Image, ExifTags, ImageOps

from test_multi_stgan import generate_session_and_graph_model_2, generate_image_from_crop_stgan_from_preload_2
from test_multi_stgan import  generate_image_from_crop_with_fullstring, generate_image_from_crop_with_fullstring_baseimage

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
from util.visualizer import save_images
from util import html
import torchvision.transforms as transforms
import torch.utils.data

import tflib as tl
import tensorflow as tf

import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK']='True'


import dlib.cuda as cuda
print(cuda.get_num_devices())

dlib.DLIB_USE_CUDA = True
print(dlib.DLIB_USE_CUDA)




args_dict = {}
gan_dict = {}

cyclegan_dict = {}
cyclegan_opts = {}
# def setupCuda():

def load_stgan_with_path(path='128_stgan_baddies_masked_eyes'):
	gan_path_name_eyes = path
	[gan_sess_model_4, gan_xa_sample_model_4, gan__b_sample_model_4, gan_raw_b_sample_model_4,  gan_x_sample_model_4,  D, xa_logit_gan, xa_logit_att] =  generate_session_and_graph_model_2(model_path=gan_path_name_eyes)
	gan_dict[gan_path_name_eyes] = [gan_sess_model_4, gan_xa_sample_model_4, gan__b_sample_model_4, gan_raw_b_sample_model_4,  gan_x_sample_model_4]
	with open('./face-cropping_output/%s/setting.txt' % gan_path_name_eyes) as f:
		args = json.load(f)
		args_dict[gan_path_name_eyes] = args

def reset_default_graph():
	tf.reset_default_graph()

def load_cyclegan_with_path(path='cyclegan_baddies_c_1_a'):
	opt = TestOptions().parse() # get test options
	# hard-code some parameters for test
	opt.num_threads = 0   # test code only supports num_threads = 1
	opt.batch_size = 1    # test code only supports batch_size = 1
	opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
	opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
	opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
	opt.dataroot = "./test_image"
	opt.load_size = 256
	opt.crop_size = 256
	opt.gpu_ids = '0'
# 	opt.gpu_ids = ''
	opt.name = path
	opt.eval = True

	torch.cuda.set_device(0)
	#  create a dataset given opt.dataset_mode and other options
	model = create_model(opt)      # create a model given opt.model and other options
	model.setup(opt)  

	model.eval()
	
	cyclegan_dict[path] = model
	cyclegan_opts[path] = opt

dec_time = time.time()



def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform_list = []
    osize = [opt.load_size, opt.load_size]
    transform_list.append(transforms.Resize(osize, method))
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

#     __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

	
	
	
	
def run_stgan_on_image(masked_eye_left, gan_path_name='', attr_name='', intensity=1.0):
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
	_b_sample_ipt_copy[..., atts.index(attr_name)] = intensity


	sess_result = sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
															   _b_sample: _b_sample_ipt,
															   raw_b_sample: _b_sample_ipt_copy})
														   
	sess_return = sess_result.copy().squeeze(0)      
	return_image = (sess_return + 1.0) * 127.5;



# 	Prepare cyclegan input
	final_return = cv2.cvtColor(return_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
	return final_return
			
def run_cyclegan_on_preprocessed_image(processed_image, cyclegan_path = 'stgan128_wrinkled_cyclegan_data_1_a_cyclegan_ouutput'):

# 	image_size = 
# 	masked_image3_resized = cv2.resize(processed_image, (256, 256))
	masked_image3_resized = processed_image.copy()
	
	
	opt = cyclegan_opts[cyclegan_path]
	model = cyclegan_dict[cyclegan_path]
	
	print("cyclegan_opts")
	print(opt)
		
	A = get_transform(opt)(Image.fromarray(cv2.cvtColor(masked_image3_resized.copy(), cv2.COLOR_BGR2RGB)))
	data = {'A': A, 'A_paths': ''}
	
	
	
	dataloader = torch.utils.data.DataLoader(
            [data],
            batch_size=1,
            shuffle=False,
            num_workers=1)
            
            


	im = None
	
	for d in dataloader:
		this_d = d.copy()
		for i in range(1):
			model.set_input(this_d)  # unpack data from data loader
			model.test()           # run inference
			visuals = model.get_current_visuals()
# 		(label, im_data) = visuals.first()
			for label, im_data in visuals.items():
				this_d["A"] = im_data
				im = util.tensor2im(im_data).copy()
#     
	im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
	im_resize = cv2.resize(im, (processed_image.shape[1], processed_image.shape[0]))
	
	return im_resize




	
