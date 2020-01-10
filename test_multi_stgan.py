from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import traceback


import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl

import celeb_data as data
import models_stgan as models

from datetime import datetime


import cv2

import os

# ==============================================================================
# =                                    param                                   =
# ==============================================================================


step_val = 2.0;
use_cropped_img = True#args['use_cropped_img']



def get_save_name_from_attrs_and_ints(attrs, ints):
	save_string = ""
	for att in attrs:
		save_string += att
		save_string += "-"
	save_string += "++"
	for int in ints:
		save_string += str(int)
		save_string += "-"
	return save_string;
	
		
def generate_session_and_graph_model_2(model_path = "384_hair", should_finalize=False):
	
	parser = argparse.ArgumentParser()
	args_ = parser.parse_args()
	with open('./face-cropping_output/%s/setting.txt' % model_path) as f:
		args = json.load(f)

        # model
	atts = args['atts']
	n_att = len(atts)
	img_size = args['img_size']
	shortcut_layers = args['shortcut_layers']
	inject_layers = args['inject_layers']
	enc_dim = args['enc_dim']
	dec_dim = args['dec_dim']
	dis_dim = args['dis_dim']
	dis_fc_dim = args['dis_fc_dim']
	enc_layers = args['enc_layers']
	dec_layers = args['dec_layers']
	dis_layers = args['dis_layers']

	label = args['label']
	use_stu = args['use_stu']
	stu_dim = args['stu_dim']
	stu_layers = args['stu_layers']
	stu_inject_layers = args['stu_inject_layers']
	stu_kernel_size = args['stu_kernel_size']
	stu_norm = args['stu_norm']
	stu_state = args['stu_state']
	multi_inputs = args['multi_inputs']
	rec_loss_weight = args['rec_loss_weight']
	one_more_conv = args['one_more_conv']

	session_load_time = datetime.now()
	sess_2 = tl.session()

	Genc_2 = partial(models.Genc, dim=enc_dim, n_layers=enc_layers, multi_inputs=multi_inputs)
	Gdec_2 = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers,
               inject_layers=inject_layers, one_more_conv=one_more_conv)
	Gstu_2 = partial(models.Gstu, dim=stu_dim, n_layers=stu_layers, inject_layers=stu_inject_layers,
               kernel_size=stu_kernel_size, norm=stu_norm, pass_state=stu_state)
               
	
               

	# inputs
	xa_sample_2 = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
	_b_sample_2 = tf.placeholder(tf.float32, shape=[None, n_att])
	raw_b_sample_2 = tf.placeholder(tf.float32, shape=[None, n_att])
	test_label_2 = _b_sample_2 - raw_b_sample_2
	
	D = partial(models.D, n_att=n_att, dim=dis_dim, fc_dim=dis_fc_dim, n_layers=dis_layers)
	xa_logit_gan, xa_logit_att = D(xa_sample_2)

	# sample
	x_sample_2 = Gdec_2(Gstu_2(Genc_2(xa_sample_2, is_training=False),
                         test_label_2, is_training=False), test_label_2, is_training=False)
	ckpt_dir = './face-cropping_output/%s/checkpoints' % model_path
	print(ckpt_dir)
	try:
		tl.load_checkpoint(ckpt_dir, sess_2)
	except Exception as e:
		print(e)
		raise Exception(' [*] No checkpoint!')

	print("Time Session Load")
	print(datetime.now() - session_load_time)
	if(should_finalize == True):
		sess.graph.finalize()
	return [sess_2, xa_sample_2, _b_sample_2, raw_b_sample_2,  x_sample_2, D, xa_logit_gan, xa_logit_att]
	

def convertStringToAttrAndIntsList(str):
	
	final_atts = []
	final_ints = []
	cellList = str.split("***")[1:]
	print(cellList)
	for c in cellList:
		attrs = c.split("+++")[0]
		attr_list = attrs.split(",")
		ints = c.split("+++")[1]
		int_list = ints.split(",")
		print(attr_list, int_list)
		for att in attr_list:
			final_atts.append(att)
		for int_val in int_list:
			final_ints.append(float(int_val))
	final_list = zip(final_atts, final_ints)
	return final_list

def generate_image_from_crop_stgan_from_preload_2(path, all_crops_string="", sess=None, xa_sample=None, _b_sample=None, raw_b_sample=None,  x_sample=None):
	
	parser = argparse.ArgumentParser()
	args_ = parser.parse_args()
	with open('./face-cropping_output/%s/setting.txt' % "384_hair") as f:
		args = json.load(f)

	# model
	atts = args['atts']
	n_att = len(atts)
	img_size = args['img_size']
	shortcut_layers = args['shortcut_layers']
	inject_layers = args['inject_layers']
	enc_dim = args['enc_dim']
	dec_dim = args['dec_dim']
	dis_dim = args['dis_dim']
	dis_fc_dim = args['dis_fc_dim']
	enc_layers = args['enc_layers']
	dec_layers = args['dec_layers']
	dis_layers = args['dis_layers']
	thres_int = args['thres_int']

	label = args['label']
	use_stu = args['use_stu']
	stu_dim = args['stu_dim']
	stu_layers = args['stu_layers']
	stu_inject_layers = args['stu_inject_layers']
	stu_kernel_size = args['stu_kernel_size']
	stu_norm = args['stu_norm']
	stu_state = args['stu_state']
	multi_inputs = args['multi_inputs']
	rec_loss_weight = args['rec_loss_weight']
	one_more_conv = args['one_more_conv']
	
	
	session_load_time = datetime.now()
	sess_shit = tf.Session()
	te_data = data.Celeba(path, atts, img_size, 1, part='test', sess=sess_shit, crop=False)
	
	cell_string_list = all_crops_string.split("$$$$$")
	print("CELL  STRING LIST")
	print(cell_string_list)
	
	

	print("Time Session Load")
	print(datetime.now() - session_load_time)
	total_time = (datetime.now() - session_load_time)

	return_images = []
	print("SHIT1")

	for cell_string in cell_string_list:
	    print("SHIT2")
	    if(cell_string == ""):
	    	print("cell string is  blank")
	    	continue
	    final_list = convertStringToAttrAndIntsList(cell_string)
# 	    (attrs_list, int_changers) = final_list
# 	    print("attrs and ints")
# 	    print(attrs_list)
# 	    print(int_changers)
# 	    save_string = get_save_name_from_attrs_and_ints(test_atts, test_ints)
	    save_string = cell_string + "_picture"
	    for idx, batch in enumerate(te_data):
	        print("SHIT3")
	        xa_sample_ipt = batch[0]
	        a_sample_ipt = batch[1]
	        b_sample_ipt = np.array(a_sample_ipt, copy=True)

	        x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
	        x_sample_opt_list = []
	        _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
	        _b_sample_ipt_copy = _b_sample_ipt.copy()
# 	        _b_sample_ipt = b_sample_ipt
	        print("original attributes")
	        print(_b_sample_ipt)
	        print(final_list)
	        for a, i in final_list:
	        	print(i)
	        	print(thres_int)
	        	print(_b_sample_ipt_copy[..., atts.index(a)])
	        	_b_sample_ipt_copy[..., atts.index(a)] = _b_sample_ipt_copy[..., atts.index(a)] - ( i * thres_int)
	        print("changed attributes")
	        print(_b_sample_ipt_copy)
	        generation_time = datetime.now()
# 	        x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
	        x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
                                                                   _b_sample: _b_sample_ipt,
                                                                   raw_b_sample: _b_sample_ipt_copy}))
	        sample = np.concatenate(x_sample_opt_list, 2)
	        return_image = (sample.squeeze(0) + 1.0) * 127.5;
	        return_images.append(return_image)
	        
	        print("Time Generation")
	        print(datetime.now() - generation_time)
	        
	        
	
	        save_dir = './%s' % (path)
	        save_time = datetime.now()
	        im.imwrite(sample.squeeze(0), '%s/%s.jpg' % (save_dir, save_string))	        
	        print("Time Save")
	        print(datetime.now() - save_time)

	
	        print('%d.png done!' % (idx + 182638))
	        break
	sess_shit.close()
	return [return_images, total_time]


def get_changing_gan_index_from_string(all_crops_string = ''):
	cell_string_list = all_crops_string.split("$$$$$")
	crop_string_gan_dict = {}
	for cell_string in cell_string_list:
		separate_gans = cell_string.split("!!!")
		gan_index = 0
		for crop_and_gan_string in separate_gans:
			gan_name = crop_and_gan_string.split('@@@')[0]
			crop_string = crop_and_gan_string.split('@@@')[1]
			gan_dict_key = str(gan_index)
			if(gan_dict_key not in crop_string_gan_dict):
				crop_string_gan_dict[gan_dict_key] = [crop_string]
			else:
				crop_string_gan_dict[gan_dict_key].append(crop_string)
			gan_index += 1
	
	for (key, value) in crop_string_gan_dict.items():
		if(len( set( value ) ) > 1):
			return int(key)
	return -1

def get_base_image(path, all_crops_string="", gan_dict=None, args_dict=None, mask_image = None, gan_changing_index = 0):
	
	print("\n\n\n")
	print("**********************")
	print('starting get_base_image')
	print("all_crops_string:")
	print(all_crops_string)
	print("gan_changing_index")
	print(gan_changing_index)
	print("**********************")
	
	first_gan_string = all_crops_string.split("$$$$$")[0]
	separate_gans = first_gan_string.split("!!!")
	index_gan_string = separate_gans[gan_changing_index]
	index_gan_name = index_gan_string.split('@@@')[0]
	index_crop_string = index_gan_string.split('@@@')[1]	
	final_list = convertStringToAttrAndIntsList(index_crop_string)
	
	
	return_images = [];
	
	attr_string = ""
	int_string = ""
	for a, i in final_list:
		attr_string += a
		attr_string += ","
		int_string += "0"
		int_string += ","
	attr_string = attr_string[:-1]
	int_string = int_string[:-1]
	new_index_crop_string = index_gan_name + "@@@" + "***" + attr_string + '+++' + int_string
	
	separate_gans[gan_changing_index] = new_index_crop_string
	
	first_gan_string = first_gan_string.replace(index_gan_string, new_index_crop_string)
	cell_string_list = [first_gan_string]
	
	print("separate_gans:")
	print(separate_gans)
	
	print("cell_string_list")
	print(cell_string_list)
	
	for cell_string in cell_string_list:
		save_string = cell_string + "_picture"
		save_dir = './%s' % (path)
		
		full_save_string = '%s/%s.jpg' % (save_dir, save_string)
		print("checking for existance of base image")
		print(full_save_string)
		if os.path.isfile(full_save_string):
			print("base image already exists, returning image")
			image = np.array(cv2.cvtColor(cv2.imread(full_save_string), cv2.COLOR_BGR2RGB))
			return_images.append(image)
			return return_images
			
		
		

		previous_gan_output = None;
		gan_index = 0
		final_gan_index = 1;
		for crop_and_gan_string in separate_gans:
		
			gan_name = crop_and_gan_string.split('@@@')[0]
			crop_string = crop_and_gan_string.split('@@@')[1]
			
			[sess, xa_sample, _b_sample, raw_b_sample, x_sample] = gan_dict[gan_name]
			args = args_dict[gan_name]
			atts = args['atts']
			n_att = len(atts)
			img_size = args['img_size']
			shortcut_layers = args['shortcut_layers']
			inject_layers = args['inject_layers']
			enc_dim = args['enc_dim']
			dec_dim = args['dec_dim']
			dis_dim = args['dis_dim']
			dis_fc_dim = args['dis_fc_dim']
			enc_layers = args['enc_layers']
			dec_layers = args['dec_layers']
			dis_layers = args['dis_layers']
			thres_int = args['thres_int']

			label = args['label']
			use_stu = args['use_stu']
			stu_dim = args['stu_dim']
			stu_layers = args['stu_layers']
			stu_inject_layers = args['stu_inject_layers']
			stu_kernel_size = args['stu_kernel_size']
			stu_norm = args['stu_norm']
			stu_state = args['stu_state']
			multi_inputs = args['multi_inputs']
			rec_loss_weight = args['rec_loss_weight']
			one_more_conv = args['one_more_conv']


			final_list = convertStringToAttrAndIntsList(crop_string)
			
			sess_shit = tf.Session()
			te_data = data.Celeba(path, atts, img_size, 1, part='test', sess=sess_shit, crop=False)
			isallzeros = False;
			
			isallzeros = True;
			for a, i in final_list:
				if(i != 0):
					isallzeros = False;
						
			if(isallzeros == True):
				if(gan_index == final_gan_index):
					print("final gan attributes are all zeros, saving image and will continue without running session")
					sample = np.concatenate(previous_gan_output, 2)
					return_image = (sample + 1.0) * 127.5;
					return_images.append(return_image)
					save_dir = './%s' % (path)
					im.imwrite(sample, '%s/%s.jpg' % (save_dir, save_string))
				else:
					print("gan attributes are zero, will continue without running session")
					


			# if(gan_index != 0):
# 				isallzeros = True;
# 				for a, i in final_list:
# 					if(i != 0):
# 						isallzeros = False;
# 				if(isallzeros == True):
# 					if(gan_index == final_gan_index):
# 						print("continueing because new attr change is zero - final")
# 						sample = np.concatenate(previous_gan_output, 2)
# 						return_image = (sample + 1.0) * 127.5;
# 						return_images.append(return_image)
# 						
# 						im.imwrite(sample, '%s/%s.jpg' % (save_dir, save_string))
# # 						break
# 					else:
# 						print("continueing because new attr change is zero")
# # 						break;
# 			else:
# 				isallzeros = True;
# 				for a, i in final_list:
# 					if(i != 0):
# 						isallzeros = False;
# 				if(isallzeros == True):
# 					if(True == True):
# 						print("continueing because new attr change is zero - first")
# # 						this_img = None
# # 						for idx, batch in enumerate(te_data_2):
# # 							this_img = batch[0]
# # 							break
# # 						x_sample_opt_list = [this_img]
# # 						sample = np.concatenate(x_sample_opt_list, 2)
# # 						return_image = (sample.squeeze(0) + 1.0) * 127.5;
# # 						return_images.append(return_image)
# # 						save_dir = './%s' % (path)
# # 						im.imwrite(sample.squeeze(0), '%s/%s.jpg' % (save_dir, save_string))	 
# 					else:
# 						print("continueing because new attr change is zero")

			
			for idx, batch in enumerate(te_data):
				print('start iterating over data')
				if(isallzeros == True):
					print('gan attributes were zero, increasing gan index, continuing')
					gan_index += 1
					if(previous_gan_output is None):
						print('previous gan output was none, setting as batch data')
						previous_gan_output = batch[0].copy()
					break
				
				
				print('gan attributes were not zero, running session')
				final_list = convertStringToAttrAndIntsList(crop_string)
				xa_sample_ipt = batch[0]
				if(previous_gan_output is not None):
					print('previous gan output exists, using this intead of batch data')
					xa_sample_ipt = previous_gan_output.copy()
				
				a_sample_ipt = batch[1]
				b_sample_ipt = np.array(a_sample_ipt, copy=True)
				x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
				x_sample_opt_list = []
				
				_b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
				_b_sample_ipt_copy = _b_sample_ipt.copy()

				shouldUseMask = True;
				for a, i in final_list:
					if(a in attrs_without_masks):
						if(i != 0):
							print("setting should use mask to false")
							shouldUseMask = False;
					_b_sample_ipt_copy[..., atts.index(a)] = _b_sample_ipt_copy[..., atts.index(a)] - ( i * thres_int)

				generation_time = datetime.now()

				print("running session with values")
				print('starting attributes:')
				print(_b_sample_ipt)
				print('new attributes:')
				print(_b_sample_ipt_copy)
				print('crop_string:')
				print(crop_string)
				print('gan_name:')
				print(gan_name)
				print('final_list:')
				print(final_list)
				
				
				
				
				sess_result = sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
                                                                   _b_sample: _b_sample_ipt,
                                                                   raw_b_sample: _b_sample_ipt_copy})
				
# 	        x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
				
				
				
				
				sess_copy = sess_result.copy().squeeze(0)
				sess_return = sess_result.copy().squeeze(0)
				
				print('finished running session')

				if(mask_image is not None):
					if(shouldUseMask == True):
						
						print('found mask image and shouldUseMask is True, adding mask image')
						
						foreground = cv2.multiply(mask_image, sess_copy.astype(np.float))
						background = cv2.multiply(1 - mask_image, batch[0].squeeze(0).astype(np.float))
						img_return = cv2.add(foreground, background)
						sess_return = img_return.copy()
				
				
				x_sample_opt_list.append(sess_return.copy())
				sample = np.concatenate(x_sample_opt_list, 2)
				return_image = (sess_return + 1.0) * 127.5;


				if(gan_index == final_gan_index):
					print("final gan, saving image")
					return_images.append(return_image)
	        	
				
				previous_gan_output = x_sample_opt_list.copy()
	        	
	        
				print("Time Generation")
				print(datetime.now() - generation_time)
				
				save_dir = './%s' % (path)
				save_time = datetime.now()
				im.imwrite(sess_return, '%s/%s.jpg' % (save_dir, save_string))	        
				print("Time Save")
				print(datetime.now() - save_time)
				gan_index += 1	
				break
	
	sess_shit.close()
	return return_images
	
attrs_without_masks = ["Bangs"]
def generate_image_from_crop_with_fullstring_baseimage(path, all_crops_string="", gan_dict=None, args_dict=None, mask_image = None):
	
	print("\n\n\n")
	print("**********************")
	print("starting generate_image_from_crop_with_fullstring_baseimage")
	print("all_crops_string:")
	print(all_crops_string)
	print("**********************")
	
	session_load_time = datetime.now()
	cell_string_list = all_crops_string.split("$$$$$")
	
	gan_changing_index = get_changing_gan_index_from_string(all_crops_string)
	
	
	print("gan changing index")
	print(gan_changing_index)
	
	base_images = get_base_image(path, all_crops_string=all_crops_string, gan_dict=gan_dict, args_dict=args_dict, mask_image = mask_image, gan_changing_index = gan_changing_index)
	
	print("finished getting base image")
	base_image = [((base_images[0] / 127.5) - 1.0)]
 
	
	return_images = []
			
	for cell_string in cell_string_list:
		print('looping cell_string:')
		print(cell_string)
		
		separate_gans = cell_string.split("!!!")
		previous_gan_output = base_image;
		gan_index = 0
		final_gan_index = 1;
		for crop_and_gan_string in separate_gans:
			print('looping crop_and_gan_string:')
			print(crop_and_gan_string)
		
			gan_name = crop_and_gan_string.split('@@@')[0]
			crop_string = crop_and_gan_string.split('@@@')[1]
			
			[sess, xa_sample, _b_sample, raw_b_sample, x_sample] = gan_dict[gan_name]
			args = args_dict[gan_name]
			atts = args['atts']
			n_att = len(atts)
			img_size = args['img_size']
			shortcut_layers = args['shortcut_layers']
			inject_layers = args['inject_layers']
			enc_dim = args['enc_dim']
			dec_dim = args['dec_dim']
			dis_dim = args['dis_dim']
			dis_fc_dim = args['dis_fc_dim']
			enc_layers = args['enc_layers']
			dec_layers = args['dec_layers']
			dis_layers = args['dis_layers']
			thres_int = args['thres_int']

			label = args['label']
			use_stu = args['use_stu']
			stu_dim = args['stu_dim']
			stu_layers = args['stu_layers']
			stu_inject_layers = args['stu_inject_layers']
			stu_kernel_size = args['stu_kernel_size']
			stu_norm = args['stu_norm']
			stu_state = args['stu_state']
			multi_inputs = args['multi_inputs']
			rec_loss_weight = args['rec_loss_weight']
			one_more_conv = args['one_more_conv']
			
			final_list = convertStringToAttrAndIntsList(crop_string)
			save_string = cell_string + "_picture"
			sess_shit = tf.Session()
			te_data = data.Celeba(path, atts, img_size, 1, part='test', sess=sess_shit, crop=False)
			isallzeros = False;
			
			isallzeros = True;
			for a, i in final_list:
				if(i != 0):
					isallzeros = False;
						
			if(isallzeros == True):
				if(gan_index == final_gan_index):
					print("final gan attributes are all zeros, saving image and will continue without running session")
					sample = np.concatenate(previous_gan_output, 2)
					return_image = (sample + 1.0) * 127.5;
					return_images.append(return_image)
					save_dir = './%s' % (path)
					im.imwrite(sample, '%s/%s.jpg' % (save_dir, save_string))
				else:
					print("gan attributes are zero, will continue without running session")

# 			if(gan_index != 0):
# 				isallzeros = True;
# 				for a, i in final_list:
# 					if(i != 0):
# 						isallzeros = False;
# 				if(isallzeros == True):
# 					if(gan_index == final_gan_index):
# 						print("final gan attributes are all zeros, saving image")
# 						sample = np.concatenate(previous_gan_output, 2)
# 						return_image = (sample + 1.0) * 127.5;
# 						return_images.append(return_image)
# 						save_dir = './%s' % (path)
# 						im.imwrite(sample, '%s/%s.jpg' % (save_dir, save_string))
# 
# 					else:
# 						print("gan attributes are zero, will continue without running session")

# 			else:
# 				isallzeros = True;
# 				for a, i in final_list:
# 					if(i != 0):
# 						isallzeros = False;
# 				if(isallzeros == True):
# 					if(True == True):
# 						print("continueing because new attr change is zero - first")
# # 						this_img = None
# # 						for idx, batch in enumerate(te_data_2):
# # 							this_img = batch[0]
# # 							break
# # 						x_sample_opt_list = [this_img]
# # 						sample = np.concatenate(x_sample_opt_list, 2)
# # 						return_image = (sample.squeeze(0) + 1.0) * 127.5;
# # 						return_images.append(return_image)
# # 						save_dir = './%s' % (path)
# # 						im.imwrite(sample.squeeze(0), '%s/%s.jpg' % (save_dir, save_string))	 
# 					else:
# 						print("continueing because new attr change is zero")

			
			for idx, batch in enumerate(te_data):
				print('start iterating over data')
				if(isallzeros == True):
					print('gan attributes were zero, increasing gan index, continuing')
					gan_index += 1
					if(previous_gan_output is None):
						print('previous gan output was none, setting as batch data')
						previous_gan_output = batch[0].copy()
					break
				if(gan_changing_index != gan_index):
					print('we are not on the index we want to change, will continue without running session')
					if(gan_index == final_gan_index):
						print('final gan, saving image')
						sample = np.concatenate(previous_gan_output, 2)
						return_image = (sample + 1.0) * 127.5;
						return_images.append(return_image)
						save_dir = './%s' % (path)
						im.imwrite(sample, '%s/%s.jpg' % (save_dir, save_string))
					gan_index += 1
					break;
				
				print('gan attributes were not zero, and we are on the gan changing index, running session')
				final_list = convertStringToAttrAndIntsList(crop_string)
				xa_sample_ipt = batch[0]
				if(previous_gan_output is not None):
					print('previous gan output exists, using this intead of batch data')
					xa_sample_ipt = previous_gan_output.copy()
				
				a_sample_ipt = batch[1]
				b_sample_ipt = np.array(a_sample_ipt, copy=True)

				x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
				x_sample_opt_list = []
				_b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
				_b_sample_ipt_copy = _b_sample_ipt.copy()

				shouldUseMask = True;
				for a, i in final_list:
					if(a in attrs_without_masks):
						if(i != 0):
							print("setting should use mask to false")
							shouldUseMask = False;
					_b_sample_ipt_copy[..., atts.index(a)] = _b_sample_ipt_copy[..., atts.index(a)] - ( i * thres_int)
					
				generation_time = datetime.now()
				
				print("running session with values")
				print('starting attributes:')
				print(_b_sample_ipt)
				print('new attributes:')
				print(_b_sample_ipt_copy)
				print('crop_string:')
				print(crop_string)
				print('gan_name:')
				print(gan_name)
				print('final_list:')
				print(final_list)
				
				
				sess_result = sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
                                                                   _b_sample: _b_sample_ipt,
                                                                   raw_b_sample: _b_sample_ipt_copy})
				
				
				sess_copy = sess_result.copy().squeeze(0)
				sess_return = sess_result.copy().squeeze(0)
				
				print('finished running session')

				if(mask_image is not None):
					if(shouldUseMask == True):
						
						print('found mask image and shouldUseMask is True, adding mask image')
						
						foreground = cv2.multiply(mask_image, sess_copy.astype(np.float))
						background = cv2.multiply(1 - mask_image, batch[0].squeeze(0).astype(np.float))
						img_return = cv2.add(foreground, background)
						sess_return = img_return.copy()
				
				
				x_sample_opt_list.append(sess_return.copy())
				sample = np.concatenate(x_sample_opt_list, 2)
				return_image = (sess_return + 1.0) * 127.5;


				if(gan_index == final_gan_index):
					print("final gan, saving image")
					return_images.append(return_image)
	        	
				
				previous_gan_output = x_sample_opt_list.copy()
	        	
	        
				print("Time Generation")
				print(datetime.now() - generation_time)
				
				save_dir = './%s' % (path)
				save_time = datetime.now()
				im.imwrite(sess_return, '%s/%s.jpg' % (save_dir, save_string))	        
				print("Time Save")
				print(datetime.now() - save_time)
				gan_index += 1	
				
				break
	sess_shit.close()
	total_time = (datetime.now() - session_load_time)
	return [return_images, total_time]



def generate_image_from_crop_with_fullstring(path, all_crops_string="", gan_dict=None, args_dict=None, mask_image = None):
	
	session_load_time = datetime.now()
	cell_string_list = all_crops_string.split("$$$$$")
	
	gan_changing_index = get_changing_gan_index_from_string(all_crops_string)
	
	
	print("gan changing index")
	print(gan_changing_index)

	
	return_images = []
			
	for cell_string in cell_string_list:
		separate_gans = cell_string.split("!!!")
		previous_gan_output = None;
		gan_index = 0
		final_gan_index = 1;
		for crop_and_gan_string in separate_gans:
		
			gan_name = crop_and_gan_string.split('@@@')[0]
			crop_string = crop_and_gan_string.split('@@@')[1]
			
			[sess, xa_sample, _b_sample, raw_b_sample, x_sample] = gan_dict[gan_name]
			args = args_dict[gan_name]
			atts = args['atts']
			n_att = len(atts)
			img_size = args['img_size']
			shortcut_layers = args['shortcut_layers']
			inject_layers = args['inject_layers']
			enc_dim = args['enc_dim']
			dec_dim = args['dec_dim']
			dis_dim = args['dis_dim']
			dis_fc_dim = args['dis_fc_dim']
			enc_layers = args['enc_layers']
			dec_layers = args['dec_layers']
			dis_layers = args['dis_layers']
			thres_int = args['thres_int']

			label = args['label']
			use_stu = args['use_stu']
			stu_dim = args['stu_dim']
			stu_layers = args['stu_layers']
			stu_inject_layers = args['stu_inject_layers']
			stu_kernel_size = args['stu_kernel_size']
			stu_norm = args['stu_norm']
			stu_state = args['stu_state']
			multi_inputs = args['multi_inputs']
			rec_loss_weight = args['rec_loss_weight']
			one_more_conv = args['one_more_conv']
						

			
			print("SHIT1", path)

			final_list = convertStringToAttrAndIntsList(crop_string)
			save_string = cell_string + "_picture"
			sess_shit = tf.Session()
			te_data = data.Celeba(path, atts, img_size, 1, part='test', sess=sess_shit, crop=False)
			te_data_2 = data.Celeba(path, atts, img_size, 1, part='test', sess=sess_shit, crop=False)
			isallzeros = False;
			if(gan_index != 0):
				isallzeros = True;
				for a, i in final_list:
					if(i != 0):
						isallzeros = False;
				if(isallzeros == True):
					if(gan_index == final_gan_index):
						print("continueing because new attr change is zero - final")
						sample = np.concatenate(previous_gan_output, 2)
						return_image = (sample + 1.0) * 127.5;
						return_images.append(return_image)
						save_dir = './%s' % (path)
						im.imwrite(sample, '%s/%s.jpg' % (save_dir, save_string))
# 						break
					else:
						print("continueing because new attr change is zero")
# 						break;
			else:
				isallzeros = True;
				for a, i in final_list:
					if(i != 0):
						isallzeros = False;
				if(isallzeros == True):
					if(True == True):
						print("continueing because new attr change is zero - first")
# 						this_img = None
# 						for idx, batch in enumerate(te_data_2):
# 							this_img = batch[0]
# 							break
# 						x_sample_opt_list = [this_img]
# 						sample = np.concatenate(x_sample_opt_list, 2)
# 						return_image = (sample.squeeze(0) + 1.0) * 127.5;
# 						return_images.append(return_image)
# 						save_dir = './%s' % (path)
# 						im.imwrite(sample.squeeze(0), '%s/%s.jpg' % (save_dir, save_string))	 
					else:
						print("continueing because new attr change is zero")

			
			for idx, batch in enumerate(te_data):
				if(isallzeros == True):
					gan_index += 1
					if(previous_gan_output is None):
						previous_gan_output = batch[0].copy()
					break
				final_list = convertStringToAttrAndIntsList(crop_string)
				print("SHIT3")
				xa_sample_ipt = batch[0]
				if(previous_gan_output is not None):
					xa_sample_ipt = previous_gan_output.copy()
				
				a_sample_ipt = batch[1]
				b_sample_ipt = np.array(a_sample_ipt, copy=True)

				x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
				x_sample_opt_list = []
				_b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
				_b_sample_ipt_copy = _b_sample_ipt.copy()

				print("original attributes")
				print(_b_sample_ipt)
				print(final_list)
				shouldUseMask = True;
				for a, i in final_list:
					print(a)
					print(i)
					print(thres_int)
					if(a in attrs_without_masks):
						if(i != 0):
							print("setting should use mask to false")
							shouldUseMask = False;
					print(_b_sample_ipt_copy[..., atts.index(a)])
					_b_sample_ipt_copy[..., atts.index(a)] = _b_sample_ipt_copy[..., atts.index(a)] - ( i * thres_int)
				print("changed attributes")
				print(_b_sample_ipt_copy, _b_sample_ipt)
				generation_time = datetime.now()
				
				sess_result = sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
                                                                   _b_sample: _b_sample_ipt,
                                                                   raw_b_sample: _b_sample_ipt_copy})
				
# 	        x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
				
				sess_copy = sess_result.copy().squeeze(0)
				sess_return = sess_result.copy().squeeze(0)
				print("shit")

				if(mask_image is not None):
					if(shouldUseMask == True):
						
						
						foreground = cv2.multiply(mask_image, sess_copy.astype(np.float))
						background = cv2.multiply(1 - mask_image, batch[0].squeeze(0).astype(np.float))
						img_return = cv2.add(foreground, background)
						sess_return = img_return.copy()
						
# 						shit = ((sess_return + 1.0) * 127.5).astype(np.uint8)
# 						hsvShit = cv2.cvtColor(shit,cv2.COLOR_BGR2HSV)
# 						
# 						hsvShit_edit1 = hsvShit.copy()
# 						hsvShit_edit1[...,1] = hsvShit_edit1[...,1]*1.4
# 						edit1 = cv2.cvtColor(hsvShit_edit1,cv2.COLOR_HSV2BGR).astype(np.uint8)
# 						#multiple by a factor to change the saturation
# 						
# 						
# 						shit2 = (sess_copy + 1.0) * 127.5;
# 						cv2.imshow('output', cv2.cvtColor(shit, cv2.COLOR_BGR2RGB))
# 						cv2.imshow('edit 1', cv2.cvtColor(edit1, cv2.COLOR_BGR2RGB))
# 						cv2.waitKey(0)						
# 				
				x_sample_opt_list.append(sess_return.copy())
				sample = np.concatenate(x_sample_opt_list, 2)

				return_image = (sess_return + 1.0) * 127.5;

				

				
				
				
				if(gan_index == final_gan_index):
					return_images.append(return_image)
	        	
				previous_gan_output = x_sample_opt_list.copy()
	        	
	        
				print("Time Generation")
				print(datetime.now() - generation_time)
				
				save_dir = './%s' % (path)
				save_time = datetime.now()
				im.imwrite(sess_return, '%s/%s.jpg' % (save_dir, save_string))	        
				print("Time Save")
				print(datetime.now() - save_time)
				
				gan_index += 1
				
				

	        
	        
	
# 	        	save_dir = './%s' % (path)
# 	        	save_time = datetime.now()
# 	        	im.imwrite(sample.squeeze(0), '%s/%s.jpg' % (save_dir, save_string))	        
# 	        	print("Time Save")
# 	        	print(datetime.now() - save_time)

	
				break
	sess_shit.close()
	total_time = (datetime.now() - session_load_time)
	return [return_images, total_time]

