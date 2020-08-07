import face_recognition
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import threading
from imutils.video import WebcamVideoStream
import time
import pyttsx3


outputFrame = None
lock = threading.Lock()
vs = WebcamVideoStream(src=0).start()
time.sleep(2.0)            
emotion =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_counts = {"Anger": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Sad": 0, "Surprise": 0, "Neutral": 0}
emotion_colors = {"Anger": "Lightcyan" ,"Disgust":"Lightcyan","Fear":"Lightcyan", "Happy":"Lavenderblush", "Sad":"Yellow", "Surprise":"Lightcyan","Neutral":"Turquoise"}

model = tf.keras.models.load_model("model.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
once = True

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        print("Runtime Error, passed.")

############################################################################
# Face ID Finctions  

def Face_ID():  
    # Get a reference to webcam #0 (the default one)
    
    
    # Load a sample picture and learn how to recognize it.
    image1 = face_recognition.load_image_file("Kimia.jpg")
    image2 = face_recognition.load_image_file("Wilder.jpg")
    #image3 = face_recognition.load_image_file("Daniel.jpg")
    image4 = face_recognition.load_image_file("Shreeya.png")
    
    
    data = {'images': [image1, image2, image4],'names': ['Kimia', 'Wilder', 'Shreeya']}
    Data_base = pd.DataFrame(data)
    
    known_face_encodings = []
    known_face_names = []
    for img, names in zip(Data_base['images'], Data_base['names']):
        # Create arrays of known face encodings and their names
        known_face_encodings.append(face_recognition.face_encodings(img)[0])
        known_face_names.append(names)
        
        
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    #########################################################################
    #------------------------image to recognize-------------------------------
    
    #1. Capture image from webcam

    frame = vs.read() 
    
    ###################################################################
        
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
        
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
    name = "Unknown"
    
    # If a match was found in known_face_encodings, just use the first one.
        
    face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    
        face_names.append(name)
        # Display the results    
        return face_locations, face_names, frame
    
def Display(face_locations, face_names, frame):
    global once
    for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
       # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
       # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Welcome " + name, (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)
        if name != "Unknown" and once:
            speak("Welcome " + name + ". My name is Sensus, and I'll be your in-car support system for today. Let's have a great drive!")
            once = False
    return frame 

def generate2(frame):
    # grab global references to the output frame and lock variables
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
            # encode the frame in JPEG format
        (flag2, encodedImage2) = cv2.imencode(".jpg", frame)

            # ensure the frame was successfully encoded
        if not flag2:
            continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage2) + b'\r\n')
        

#face_locations, face_names, frame = Face_ID()

#frame = Display(face_locations, face_names, frame)

###############################################################################


def detect_emotion(frameCount):
    # initialize the video stream 
    #Emotion recognition functions 
    i = 0
    first = True
    global prominent_em, bg_color
    # grab global references to the video stream, output frame, and lock variables
    global vs, outputFrame, lock
    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream
        # convert the frame to grayscale
        i += 1
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
                emotion_counts[em] += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #cv2.imshow("image", frame)
        
        # acquire the lock, set the output frame, and release the lock
        with lock:
            outputFrame = frame.copy()
        
        if i >= 60 and first:
            prominent_em = max(emotion_counts, key = emotion_counts.get)
            bg_color = emotion_colors[prominent_em]
            first = False
            print(prominent_em)
            print(bg_color)
            speak("Your most highly expressed emotion so far is " + prominent_em)

def return_emotion():
    global prominent_em, bg_color
    return prominent_em, bg_color

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

def recommendation(name, prominent_em):
    global recommendations, search
    if name == "Unknown":
        recommendations = ["User not recognized..."]
    else:
        # used pandas (pd) to read csv file of user preferences
        df = pd.read_csv("user_profiles.csv")
        search = "Music playback was not requested."
        # place the user's info in a dictionary
        user_dict = df[df['name'] == name].to_dict('records')[0]
        recommendations = []
        # if prominent emotion is Anger, Disgust, or Fear, and user wants music, play calm genre
        if (prominent_em in emotion[:3] or prominent_em == "Surprise")  and user_dict["suggestMusic"]:
            search = "non-copyrighted " + user_dict["calm genre"]
            speak("Playing" + search + "music...")
            recommendations.append("Personalized music/radio recommendations")
            recommendations.append("Calming background color")
            # now use Youtube API with the "search" variable
        
        if user_dict["controlClimate"]:
            recommendations.append("Climate control (suggestions for heating, cooling, temperature adjustments)")
            recommendations.append("Interior lighting, window, in-car scents, and/or volume adjustments")
            
        if prominent_em in emotion[:3] and user_dict["suggestRelax"]:
            recommendations.append("Suggested physical relaxation techniques to release tension in the body (ex: unclenching steering wheel, adjusting your seat/body position, shrugging of stress in the back/shoulders/neck, etc.)")
            recommendations.append("Suggested breathing/meditation techniques")
        
        if prominent_em in emotion[:3] and user_dict["suggestBreaks"]:
            recommendations.append("Suggestions to pull over, take a break from driving, take a walk, reroute, or try to clear your head")
        
        if prominent_em in ["Sad", "Neutral"] and user_dict["suggestMusic"]:
            search = "non-copyrighted " + user_dict["cheer-up genre"]  
            speak("To help cheer you up, I'll play some" + search + "music...")
            recommendations.append("Personalized music/radio recommendations")
            recommendations.append("Bright background color to boost your mood")
            # now use Youtube API with the "search" variable
    return recommendations, search