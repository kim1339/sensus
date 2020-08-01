import face_recognition
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import threading
from imutils.video import WebcamVideoStream
import time


outputFrame = None
lock = threading.Lock()
vs = WebcamVideoStream(src=0).start()
time.sleep(2.0)            
emotion =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = tf.keras.models.load_model("model.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

############################################################################
# Face ID Finctions  

def Face_ID():  
    # Get a reference to webcam #0 (the default one)
    
    
    # Load a sample picture and learn how to recognize it.
    image1 = face_recognition.load_image_file("Kimia.jpg")
    image2 = face_recognition.load_image_file("Wilder.jpg")
    image3 = face_recognition.load_image_file("joe.jpg")
    image4 = face_recognition.load_image_file("Shreeya.PNG")
    
    
    data = {'images':[image1,image2, image3,image4 ],'names':['Kimia','Wilder','Youssef',"Shreeya" ]}
    Data_base = pd.DataFrame(data)
    
    known_face_encodings = []
    known_face_names = []
    for img,names in zip(Data_base['images'], Data_base['names']):
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
    """
    video_capture = cv2.VideoCapture(0)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
    # Read picture. ret === True on success
    ret, frame = video_capture.read()
    # Close device
    video_capture.release()
    """
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
    for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
       # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
       # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Welcome " + name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
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
        #cv2.imshow("image", frame)
        
        
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



    
    

  