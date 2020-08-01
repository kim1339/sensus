from flask import Flask, render_template, Response
from utilis import *
import threading



app = Flask(__name__)

@app.route("/")
def index():
    data = {}
    data["message"] = "Hello, your app is working!"
    return  render_template("index.html")


@app.route("/FaceId")
def face_id():
    face_locations, face_names, frame = Face_ID()
    frame = Display(face_locations, face_names, frame)
    data = {}
    data['img'] = frame
    
    return Response(generate2(frame),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

#def index1():
    #return  render_template("index1.html")


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media type (mime type)
    t = threading.Thread(target=detect_emotion, args=(
    32,))
    t.daemon = True
    t.start()

    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

#def index2():
    #return  render_template("index.html")

# start a thread that will perform emotion detection




