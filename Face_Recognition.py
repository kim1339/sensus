import face_recognition
import cv2
import numpy as np
import pandas as pd
import os



#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
image1 = face_recognition.load_image_file("Kimia.jpg")
image2 = face_recognition.load_image_file("Wilder.jpg")
image3 = face_recognition.load_image_file("joe.jpg")
image4 = face_recognition.load_image_file("Shreeya.jpg")


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

video_capture = cv2.VideoCapture(0)
# Check success
if not video_capture.isOpened():
    raise Exception("Could not open video device")
# Read picture. ret === True on success
ret, frame = video_capture.read()
# Close device
video_capture.release()

from matplotlib import pyplot as plt
frameRGB = frame[:,:,::-1] # BGR => RGB
plt.imshow(frameRGB)
###################################################################

######################################################################
#2. if you want to load you own image 
'''
path = ''
frame = cv2.imread(path)
'''
###########################################################################

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
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



Recommendations = {"Youssef":{"Music":"Amr diab", "Movies":"Avengers"}}
print(Recommendations["Youssef"])


video_capture.release()
cv2.destroyAllWindows()