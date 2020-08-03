from flask import Flask, render_template, Response, request
from utilis import *
import threading


app = Flask(__name__)

@app.route("/",methods=["GET", "POST"])
def index():
    data = {}
    data["message"] = "Hello, your app is working!"
    return  render_template("index2.html")


@app.route("/FaceId")
def face_id():
    face_locations, face_names, frame = Face_ID()
    frame = Display(face_locations, face_names, frame)
    
    return Response(generate2(frame),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media type (mime type)
    t = threading.Thread(target=detect_emotion, args=(
    32,))
    t.daemon = True
    t.start()
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame", )

@app.route("/plt")
def emo_fig():
    emo_im = emotion_plot()
    
    return Response(generate2(emo_im),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

          
@app.route("/re") 
def recom():
    prom_em, bg_color, emo_im = prom_color()
    face_locations, face_names, frame = Face_ID()
    recommendations = recommendation(face_names[0], prom_em)
    return render_template("index.html", prom_em=prom_em, bg_color=bg_color, recommendations=recommendations,name=face_names[0])






