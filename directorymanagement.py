import os, time
import configs


def watch(folder_path=configs.STORAGE_PATH):
	
	while 1:
		time.sleep (10)
		fileList = [file.split('/')[-1].split('.')[0] for file in os.listdir (folder_path) if '.' in file]
		print(fileList)
		return fileList