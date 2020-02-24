import os, time
import configs
from datetime import datetime

def watch(folder_path=configs.STORAGE_PATH,fromtime = datetime.now().timestamp(),totime = datetime.now().timestamp()):
	
	while 1:
		time.sleep (10)

		fileList = [file.split('/')[-1].split('.')[0] for file in os.listdir (folder_path) if '.' in file]
		# fileList_filtered_bydatetime = [file.split('/')[-1].split('.')[0] for filename in fileList if (fromtime>=filename.split('_')[1]>=totime)]
		# print(fromtime>=filename.split('_')[1]>=totime)
		# print(fileList,fileList2)
		return fileList_filtered_bydatetime