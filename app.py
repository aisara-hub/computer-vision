from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, Response
import os
from camera import VideoCamera
# from camera import VideoCamera1
# from camera import VideoCamera2
# from camera import VideoCamera3

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
	 return render_template('dashboard.html')
	# if not session.get('logged_in'):
	# 	return render_template('login.html')
	# else:
	#	return "You're logged in"
def gen(camera):
    while True:
        frame = camera.get_frame()
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

#Cam 1             
@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
     return Response(gen(VideoCamera(src=cam_id)),mimetype='multipart/x-mixed-replace; boundary=frame')

# #Cam 2
# @app.route('/video_feed1')
# def video_feed1():
#      return Response(gen(VideoCamera1()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# #Cam 3
# @app.route('/video_feed2')
# def video_feed2():
#      return Response(gen(VideoCamera2()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# #Cam 4
# @app.route('/video_feed3')
# def video_feed3():
#      return Response(gen(VideoCamera3()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
    

if __name__ == "__main__":
	app.secret_key = os.urandom(12)
	app.run(debug=True,host='0.0.0.0')
