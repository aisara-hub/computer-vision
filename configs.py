import os 
RUNNER = "daus"
# RUNNER = "chikeong"
# RUNNER = "taufiq"


STORAGE_PATH = "storage"

def assure_path_exists(path):
	try:
		dir = os.path.dirname(path)
	except Exception as e:
		os.makedirs(dir)
        