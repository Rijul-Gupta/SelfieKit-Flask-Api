import os
import sqlite3

# create a default path to connect to and create (if necessary) a database
# called 'database.sqlite3' in the same directory as this script
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'database.sqlite3')

def db_connect(db_path=DEFAULT_PATH):
    con = sqlite3.connect(db_path, timeout=300)
    return con
    
def create_raw_image(con, path):
    cur = con.cursor()
    exist_sql = "SELECT id, path FROM raw_images WHERE path = ?"
    cur.execute(exist_sql, (path,))
    list = cur.fetchall()
    if(len(list) > 0):
    	path_id = list[0][0]
    	return path_id
    sql = """
        INSERT INTO raw_images (path)
        VALUES (?)"""
    
    cur.execute(sql, (path,))
    return cur.lastrowid
    
def remove_landmarks_from_raw_images(con, raw_image_id):
	cur = con.cursor()
	exist_sql = "SELECT id, raw_image_id FROM landmarks WHERE raw_image_id = ?"
	cur.execute(exist_sql, (raw_image_id,))
	list = cur.fetchall()
	if(len(list) > 0):
		delete_sql = 'DELETE FROM landmarks WHERE raw_image_id='+str(raw_image_id)
		cur.execute(delete_sql)
		return;
    	
def create_landmark(con, raw_image_id, landmark_string):
    sql = """
        INSERT INTO landmarks (raw_image_id, landmark_string)
        VALUES (?, ?)"""
    cur = con.cursor()
    cur.execute(sql, (raw_image_id, landmark_string))
    return cur.lastrowid
    
    
def convert_landmark_list_to_string(landmark_list):
	landmark_string = ''
	for [x, y] in landmark_list:
		landmark_string += str(x)
		landmark_string += ", "
		landmark_string += str(y)
		landmark_string += ", "
	landmark_string = landmark_string[:-len(", ")]
	return landmark_string
	
def convert_landmark_string_to_list(landmark_string):
	split = landmark_string.split(", ")
	landmark_list = []
	for val in split:
		if(len(landmark_list) == 0):
			landmark_list.append([int(float(val))])
		else:
			if(len(landmark_list[-1]) == 1):
				landmark_list[-1].append(int(float(val)))
			else:
				landmark_list.append([int(float(val))])
	return landmark_list
	
	
	
def get_landmarks_string_from_id(con, id):
	cur = con.cursor()
	exist_sql = "SELECT id, raw_image_id, landmark_string FROM landmarks WHERE id = ?"
	cur.execute(exist_sql, (id,))
	list = cur.fetchall()

	
	if(len(list) > 0):
		return list[0][2]
	return None
	
def check_if_path_has_landmarks(con, raw_image_id):
	cur = con.cursor()
	exist_sql = "SELECT id, raw_image_id FROM landmarks WHERE raw_image_id = ?"
	cur.execute(exist_sql, (raw_image_id,))
	list = cur.fetchall()
	if(len(list) > 0):
		return True
	return False
	
		
def convert_raw_image_path_to_crop_path(raw_image_path, crop_folder, landmark_id):
	file_base, file_ext = os.path.splitext(raw_image_path)
	
	file_base = file_base.replace("/baddie_lips_scrapes_google/", "/blsg/")
	file_base = file_base.replace("/goth-scrapes-google/", "/gsg/")
	file_base = file_base.replace("/insta_scrapes/", "/insc/")
	file_base = file_base.replace("/vsco_ggle/", "/vscg/")
	file_base = file_base.replace("/smokey_eyes_scrapes_google/", "/sesg/")
	
	file_base = file_base.replace("/selfie-set/", "/sfst/")
	file_base = file_base.replace("/final_selections/", "/fnsl/")
	file_base = file_base.replace("/img_celeba/", "/imcl/")
	
	file_base = file_base.replace("/fat_man_scrapes_google/", "/fmsg/")
	file_base = file_base.replace("/fat_woman_scrapes_google/", "/fwsg/")
	
	file_base = file_base.replace("/old_person_scrapes_google/", "/opsg/")
	
	file_base = file_base.replace("/orange_tan_google_scrapes/", "/otgs/")
	
	file_base = file_base.replace("raw_images/", "")
	crop_save_name = crop_folder + file_base.replace("/", "_") + '_' + str(landmark_id) + file_ext
	return crop_save_name
	
	
def get_raw_images_that_have_not_been_cropped(con, cropped_images_list, crop_folder):
	cur = con.execute ('''\
            SELECT raw_images.path, landmarks.id, landmarks.landmark_string
            FROM raw_images
            INNER JOIN landmarks
            ON raw_images.id = landmarks.raw_image_id''')
	data = cur.fetchall()
	print("Number exists in database before stripping")
	print(len(data))
	cropped_images_set = set(cropped_images_list)
	
	output_tuple = []
	for dat in data:
		crop_string = convert_raw_image_path_to_crop_path(dat[0], crop_folder, dat[1])
		if(crop_string in cropped_images_set):
			continue;
		else:
# 			print((dat[0], crop_string, dat[2]))
			output_tuple.append((dat[0], crop_string, dat[2]))
	return output_tuple
	
def clean_list_with_current_landmarks(con, landmarks_list):
	cur = con.execute ('''\
            SELECT raw_images.path
            FROM raw_images
            INNER JOIN landmarks
            ON raw_images.id = landmarks.raw_image_id''')
	data = cur.fetchall()
	paths = [dat[0] for dat in data]
	print("total paths")
	print(len(paths))
	l3 = list(set(landmarks_list) - set(paths))
	print(len(l3))
	return l3
	
def clean_dict_with_current_landmarks(con, landmarks_dict):
	cur = con.execute ('''\
            SELECT raw_images.path
            FROM raw_images
            INNER JOIN landmarks
            ON raw_images.id = landmarks.raw_image_id''')
	data = cur.fetchall()
	paths = [dat[0] for dat in data]
	print("total paths")
	print(len(paths))
# 	l3 = [x for x in landmarks_dict.keys() if x not in paths]
	l3 = list(set(landmarks_dict.keys()) - set(paths))
	print(len(l3))
	new_dict = {}
	for l in l3:
		new_dict[l] = landmarks_dict[l]
	print("done")
	return new_dict