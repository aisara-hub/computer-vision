import os, time
import configs
from datetime import datetime

def watch(folder_path=configs.STORAGE_PATH,fromtime = datetime.now().timestamp(),totime = datetime.now().timestamp()):
	
	while 1:
		time.sleep (10)
		
		# the old one
		# fileList = [file.split('/')[-1].split('.')[0] for file in os.listdir (folder_path) if '.' in file]

		# the new one
		fileList = [file.split('/')[-1] for file in os.listdir (folder_path) if '.' in file]
		# print(fileList,fromtime,totime)
		fileList_filtered_bydatetime = [filename for filename in fileList if (int(int(fromtime)/1000)<=int("".join(filename.split('.')[:-2]).split('_')[1])<=int(int(totime)/1000))]
		# 1582598025
		# for filename in fileList:
		# 	print("what is this 1: ",filename.split('.')[:-2])
		# 	print("what the hell is this: ",int("".join(filename.split('.')[:-2]).split('_')[1]))
		# 	print(int(int(fromtime)/1000))
		# 	print(int(int(totime)/1000))
		# 	print(int(int(fromtime)/1000)<=int("".join(filename.split('.')[:-2]).split('_')[1])<=int(int(totime)/1000))
		# 	if ((int(fromtime)/1000)>=int("".join(filename.split('.')[:-2]).split('_')[1])>=(int(totime)/1000)):
		# 		pass
		# print("after filter",fileList_filtered_bydatetime,fromtime,totime)
		return fileList_filtered_bydatetime