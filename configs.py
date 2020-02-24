import os 
RUNNER = "daus"
# RUNNER = "chikeong"
# RUNNER = "taufiq"
HOST_IP = "http://localhost:5000"

STORAGE_PATH = "static/storage"
def assure_path_exists(path):
	try:
		dir = os.path.dirname(path)
	except Exception as e:
		os.makedirs(dir)