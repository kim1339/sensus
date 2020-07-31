#!/usr/bin/env python
# coding: utf-8

import pyttsx3
import speech_recognition as sr
import pyaudio
import numpy as np

r = sr.Recognizer()

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def response():    
        try: 
            with sr.Microphone() as source2: 
                # let the recognizer adjust the energy threshold based on the surrounding noise level  
                r.adjust_for_ambient_noise(source2, duration = 0.2) 
                # listen for the user input  
                audio2 = r.listen(source2) 
                # using google to recognize audio 
                input_text = r.recognize_google(audio2).lower()
                return input_text

        except (sr.RequestError, sr.UnknownValueError):
            speak("Please try again.")
            response()


speak("Hello there, my name is Sensus. I do not recognize you yet. Would you like to set up a new driver profile?")
print("Please answer with 'Yes' or 'No'")

users = {"name": [], 
         "readEmotion": [],
         "suggestMusic": [],
         "calm genre": [],
         "cheer-up genre": [],
         "controlClimate": [],
         "suggestRelax": [],
         "suggestBreaks": [],
         "weeklyReport": []}


if response() == 'yes': #if 'yes' in response() or 'yeah' in response(): else: speak("Sorry, I didn't get that")
    speak("Ok, wonderful. I will ask you a few questions about yourself, and let you make selections to personalize your driving experience. First, what is your name?")
    name = response()
    users["name"].append(name)
    speak("Ok, " + name + " are you comfortable with me monitoring your emotions while you are driving?")
    print("Please answer with 'Yes' or 'No'")
    if response() == "yes":
        users["readEmotion"].append(True)
        speak("Great, would you like me to make music recommendations based on your mood?")
        print("Please answer with 'Yes' or 'No'")
        if response() == "yes":
            users["suggestMusic"].append(True)
            speak("Awesome, what genre of music calms you down?")
            calm_genre = response()
            users["calm genre"].append(calm_genre)
        else:
            users["suggestMusic"].append(False)
    else:
        speak("Ok, no worries, have a safe drive!")
        users["readEmotion"].append(False)
else:
    speak("Ok, no worries, have a safe drive!")

print(users)




