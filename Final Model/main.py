# Face Recognition

# Importing the libraries
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
import speech_recognition as sr
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from googlesearch import search
from GoogleNews import GoogleNews
import pyttsx3
import os
from datetime import datetime
import pytz
from bs4 import BeautifulSoup
import requests
import pywhatkit as py




stopwords = set(stopwords.words('english'))


model = load_model('facefeatures_new_model.h5')
flag = False

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    #canvas = detect(gray, frame)
    #image, face =face_detector(frame)
    
    face=face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
           #Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)
                     
        name="None matching"
        
        

        if(pred[0][0]>0.9):
            name='Face Found'
            flag = True
        if(pred[0][1]>0.9):
            name='Face Found'
            flag = True
        if(pred[0][2]>0.9):
            name='Face Found'
            flag = True
        if(pred[0][3]>0.9):
            name='Face Found'
            flag = True   
        # else:
        #     flag = False
                     
        cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        flag = False

    cv2.imshow('Video', frame)
    if (cv2.waitKey(1) & 0xFF == ord('q') or (flag is True)):
        break
video_capture.release()
cv2.destroyAllWindows()
#print(flag)
#print(name)



if(flag is True):
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("SPEAK: ")
        command_audio = r.listen(source)
    command_text = r.recognize_google(command_audio)
    print(command_text)
    classification_model = load_model('finalmodel2.h5')
    vocab_size = 30000
    embedding_dim  = 32
    max_length = 256
    with open('tokenizer.json') as f:
        tokens_json = json.load(f)
        tokenizer = tokenizer_from_json(tokens_json)

    for word in stopwords:
        token = " " + word + " "
        command_text = command_text.replace(token, " ")
        command_text = command_text.replace("  ", " ")
    sentence_test_seq = tokenizer.texts_to_sequences([command_text])
    sentence_test_padded = pad_sequences(sentence_test_seq, padding='post', maxlen=max_length)
    labels_pred = np.argmax(classification_model.predict(sentence_test_padded),axis = -1) 
    #print(labels_pred)
    #print(labels_pred.shape)


##### FEATURES #####


if(labels_pred == np.array([1]) or labels_pred == np.array([3])):
        for j in search(command_text, tld='com', num=10, stop=10, pause=2):
            print(j)



elif(labels_pred == np.array([2])):

    headers = {
	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}


    def weather(city):
	    city = city.replace(" ", "+")
	    res = requests.get(
		f'https://www.google.com/search?q={city}&oq={city}&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=chrome&ie=UTF-8', headers=headers)
	    print("Searching...\n")
	    soup = BeautifulSoup(res.text, 'html.parser')
	    location = soup.select('#wob_loc')[0].getText().strip()
	    time = soup.select('#wob_dts')[0].getText().strip()
	    info = soup.select('#wob_dc')[0].getText().strip()
	    weather = soup.select('#wob_tm')[0].getText().strip()
	    print(location)
	    print(time)
	    print(info)
	    print(weather+"°C")


    city = input("Enter the Name of City -> ")
    city = city+" weather"
    weather(city)


elif(labels_pred == np.array([4]) or labels_pred ==  np.array([5])):

    IST = pytz.timezone('Asia/Kolkata')
  
    print("IST in Default Format : ", 
        datetime.now(IST))
  
    datetime_ist = datetime.now(IST)
    print("Date & Time in IST : ", datetime_ist.strftime('%Y:%m:%d %H:%M:%S %Z %z'))


elif(labels_pred == np.array([7])):

    googlenews = GoogleNews()
    googlenews.search(command_text)
    result = googlenews.result()
    print(len(result))

    for n in range(len(result)):
        print(n)
        for index in result[n]:
            print(index, '\n', result[n][index])
            exit()

    

elif(labels_pred == np.array([6])):
      
    print("\n\t 1.MICROSOFT WORD \t 2.MICROSOFT POWERPOINT \n\t 3.MICROSOFT EXCEL \t 4.GOOGLE CHROME \n\t 5.VLC PLAYER	 \t 6.ADOBE ILLUSTRATOR \n\t 7.ADOBE PHOTOSHOP \t 8.MICROSOFT EDGE \n\t 9.NOTEPAD	 \t 10.CAMERA \n\n\t\t	 0. FOR EXIT")
    while True:
        p = input()
        p = p.upper()
        
        if ("DONT" in p) or ("DON'T" in p) or ("NOT" in p):
            pyttsx3.speak("Type Again")
            print(".")
            print(".")
            continue

	# assignements for different applications in the menu
        elif ("GOOGLE" in p) or ("SEARCH" in p) or ("WEB BROWSER" in p) or ("CHROME" in p) or ("BROWSER" in p) or ("4" in p):
            pyttsx3.speak("Opening")
            pyttsx3.speak("GOOGLE CHROME")
            print(".")
            print(".")
            os.system("start chrome.exe")

        elif ("IE" in p) or ("MSEDGE" in p) or ("EDGE" in p) or ("8" in p):
            pyttsx3.speak("Opening")
            pyttsx3.speak("MICROSOFT EDGE")
            print(".")
            print(".")
            os.system("start msedge.exe")
            
        elif ("NOTE" in p) or ("NOTES" in p) or ("NOTEPAD" in p) or ("EDITOR" in p) or ("9" in p):
            pyttsx3.speak("Opening")
            pyttsx3.speak("NOTEPAD")
            print(".")
            print(".")
            os.system("Notepad")

        elif ("VLCPLAYER" in p) or ("PLAYER" in p) or ("VIDEO PLAYER" in p) or ("5" in p):
            pyttsx3.speak("Opening")
            pyttsx3.speak("VLC PLAYER")
            print(".")
            print(".")
            os.system("start vlc.exe")

        elif ("EXCEL" in p) or ("MSEXCEL" in p) or ("SHEET" in p) or ("WINEXCEL" in p) or ("3" in p):
            pyttsx3.speak("Opening")
            pyttsx3.speak("MICROSOFT EXCEL")
            print(".")
            print(".")
            os.system("start excel.exe")
            
        elif ("SLIDE" in p) or ("MSPOWERPOINT" in p) or ("PPT" in p) or ("POWERPNT" in p) or ("2" in p):
            pyttsx3.speak("Opening")
            pyttsx3.speak("MICROSOFT POWERPOINT")
            print(".")
            print(".")
            os.system("start powerpnt.exe")

        elif ("WORD" in p) or ("MSWORD" in p) or ("1" in p):
            pyttsx3.speak("Opening")
            pyttsx3.speak("MICROSOFT WORD")
            print(".")
            print(".")
            os.system("start winword.exe")

        # elif ("camera" in p) or ("photo" in p) or ("10" in p):
        #     pyttsx3.speak("Opening")
        #     pyttsx3.speak("CAMERA")
        #     print(".")
        #     print(".")
        #     import subprocess,os
        #     subprocess.run('start microsoft.windows.camera:', shell=True)
            

        

	# close the program
        elif ("EXIT" in p) or ("QUIT" in p) or ("CLOSE" in p) or ("0" in p):
            pyttsx3.speak("Exiting")
            break

	# for ivalid input
        else:
            pyttsx3.speak(p)
            print("Is Invalid,Please Try Again")
            pyttsx3.speak("is Invalid,Please try again")
            print(".")
            print(".")

# import os

# os.system("camera")


elif(labels_pred == np.array([8])):
    import pywhatkit
    mobile = input("Enter reciever's mobile number with country code: ")
    message = input("Enter message you want to send: ")
    hours = int(input("Enter the hours when you want to send: "))
    minutes = int(input("Enter the minutes when you want to send: "))
    pywhatkit.sendwhatmsg(mobile, message, hours, minutes, wait_time=0)

else:
    print("Try again")


