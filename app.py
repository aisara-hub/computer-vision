from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, Response, jsonify
import os
from camera_ck import VideoCamera
import directorymanagement
import configs
import time

app = Flask(__name__)

def multi_gen(camera_id):

    cam = find_camera(int(camera_id))
    cap=  cv2.VideoCapture(cam)                                                                 
    
    while True:
        # for cap in caps:
        # # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result  


@app.route('/dashboard')
def home():
	return render_template('dashboard.html',configs = configs)
	# if not session.get('logged_in'):
	# 	return render_template('login.html')
	# else:
	#	return "You're logged in"
def gen(camera):
    # times = []
    while True:
        # start = time.perf_counter()
        frame = camera.get_frame()
        # finish = time.perf_counter()
        # times.append(finish-start)
        # print(f"average processed in {sum(times)/len(times)} seconds" )
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('home'))
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template('register.html')

#Cam 1,2,3,4             
@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    return Response(gen(VideoCamera(src=cam_id)),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/storage')
def storage_files():
    args = request.args
    print(args)
    fromtime=request.args.get("fromtime")
    totime=request.args.get("to")
    print("time fileter",fromtime," : ",totime )
    return jsonify({'filelist': directorymanagement.watch(folder_path="static/unknown",fromtime=fromtime,totime=totime)})

@app.route("/logout")
def logout():
    return redirect(url_for('login'))
    #return render_template('login.html',configs = configs)

if __name__ == "__main__":
	app.secret_key = os.urandom(12)
	app.run(debug=True,host='0.0.0.0',threaded=True)

