# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from flask import Response, Flask, render_template
import threading
import argparse
import imutils
from imutils.video import WebcamVideoStream
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

''' initialize the output frame and a lock used to ensure thread-safe
exchanges of the output frames (useful for multiple browsers/tabs
are viewing tthe stream)'''
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream 
vs = WebcamVideoStream(src=0).start()
time.sleep(2.0)

emotion =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = tf.keras.models.load_model("model.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_emotion(frameCount):
	# grab global references to the video stream, output frame, and lock variables
	global vs, outputFrame, lock

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream
		# convert the frame to grayscale
		frame = vs.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cas.detectMultiScale(gray, 1.3, 5)
		for (x, y, w, h) in faces:
			face_component = gray[y:y+h, x:x+w]
			# reshape into 48x48 pixel image input for model
			fc = cv2.resize(face_component, (48, 48))
			inpt = np.reshape(fc, (1, 48, 48, 1)).astype(np.float32)
			inpt = inpt/255.
			prediction = model.predict_proba(inpt)
			em = emotion[np.argmax(prediction)]
			score = np.max(prediction)
			if score > 0.50:
				cv2.putText(frame, em +"  "+ str(round(score*100, 1)) +'%', (x, y), font, 1, (0, 255, 0), 2)
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
		cv2.imshow("image", frame)
		
		# acquire the lock, set the output frame, and release the lock
		with lock:
			outputFrame = frame.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform emotion detection
	t = threading.Thread(target=detect_emotion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()