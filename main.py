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
import streamlit as st
st.title('Hello, I am Cookie')
st.sidebar.title("Developers' Contacts")
st.sidebar.write("Aarush Pal: aarush.pal@learner.manipal.edu")
st.sidebar.write("Deepesh Garg: deepesh.garg1@learner.manipal.edu")
st.sidebar.write("Mihir Agarwal: mihir.agarwal@learner.manipal.edu")
st.sidebar.write("Saatvik Maheshwari: saatvik.maheshwari@learner.manipal.edu")

aud_text = ""
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
        break

    cv2.imshow('Video', frame)
    if (cv2.waitKey(1) & 0xFF == ord('q') or (flag is True)):
        break
video_capture.release()
cv2.destroyAllWindows()
#print(flag)
#print(name)


cookie = True
if(flag == True):
    while(cookie == True):
        print("Say 'Yes' to Continue, 'No' to Exit")
        pyttsx3.speak("Say 'Yes' to Continue, 'No' to Exit")
        st.write("Say 'Yes' to Continue, 'No' to Exit")
        r = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            print("Listening..")
            st.write("Listening..")
            aud = r.listen(source)
        aud_text = r.recognize_google(aud)
        aud_text = aud_text.upper()
        st.write(aud_text)
        print(aud_text)
        if(aud_text == "NO"):
            print("Shutting Down, Thank you")
            st.write("Shutting Down, Thank You")
            pyttsx3.speak("Shutting Down, Thank You")
            quit()
        st.write("Authentication Successful")
        print("Authentication Successful")
        r = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            print("Hello, my name is Cookie. How may I help you?")
            pyttsx3.speak("Hello, my name is Cookie. How may I help you?")
            st.write("Hello, my name is Cookie. How may I help you?")
            command_audio = r.listen(source)
        command_text = r.recognize_google(command_audio)
        print(command_text)
        st.write(command_text)
        classification_model = load_model('finalmodel2.h5')
        vocab_size = 30000
        embedding_dim  = 32
        max_length = 128
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

    ##### FEATURES #####


        if(labels_pred == np.array([1]) or labels_pred == np.array([3])):
                st.write("Here are the results for your query")
                pyttsx3.speak("Here are the results for your query")
                for j in search(command_text, tld='com', num=10, stop=10, pause=2):
                    print(j)
                    st.write(j)

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
                st.write(location) 
                print(time)
                st.write(time)
                print(info)
                st.write(info)
                print(weather+"°C")
                st.write(weather + "°C")

            r = sr.Recognizer()
            mic = sr.Microphone()
            with mic as source:
                pyttsx3.speak("Tell me the city for weather details")
                st.write("Tell me the city for weather details")
                print("Tell me the city for weather details")
                city_audio = r.listen(source)
            city = r.recognize_google(city_audio)
            #city = input("Enter the Name of City -> ")
            city = city+" weather"
            weather(city)

        elif(labels_pred == np.array([4]) or labels_pred ==  np.array([5])):

            IST = pytz.timezone('Asia/Kolkata')
        
            print("IST in Default Format : ", 
                datetime.now(IST))
        
            datetime_ist = datetime.now(IST)
            st.write("Date and Time in I S T" + datetime_ist.strftime('%Y:%m:%d %H:%M:%S %Z %z'))
            pyttsx3.speak("Date and Time in I S T" + datetime_ist.strftime('%Y:%m:%d %H:%M:%S %Z %z'))
            print("Date & Time in IST : ", datetime_ist.strftime('%Y:%m:%d %H:%M:%S %Z %z'))

        elif(labels_pred == np.array([7])):

            googlenews = GoogleNews()
            googlenews.search(command_text)
            result = googlenews.result()
            print(len(result))

            for n in range(len(result)):
                print(n)
                for index in result[n]:
                    st.write(index, '\n', result[n][index])
                    print(index, '\n', result[n][index])

            

        elif(labels_pred == np.array([6])):
            
            print("\n\t 1.MICROSOFT WORD \t 2.MICROSOFT POWERPOINT \n\t 3.MICROSOFT EXCEL \t 4.GOOGLE CHROME \n\t 5.VLC PLAYER \t 6.MICROSOFT EDGE \n\t 7.NOTEPAD \n\n\t\t	 0. FOR EXIT")
            st.write("1.MICROSOFT WORD      2.MICROSOFT POWERPOINT") 
            st.write("3.MICROSOFT EXCEL     4.GOOGLE CHROME")
            st.write("5.VLC PLAYER      6.MICROSOFT EDGE")
            st.write("7.NOTEPAD     0. FOR EXIT")
            while True:
                r = sr.Recognizer()
                mic = sr.Microphone()
                with mic as source:
                    print("Which app would you like to launch")
                    st.write("Which app would you like to launch")
                    pyttsx3.speak("Which app would you like to launch")
                    input_audio = r.listen(source)
                p = r.recognize_google(input_audio)
                p = p.upper()
                st.write(p)
                
                if ("DONT" in p) or ("DON'T" in p) or ("NOT" in p):
                    st.write("Type Again")
                    pyttsx3.speak("Type Again")
                    print(".")
                    print(".")
                    continue

            # assignements for different applications in the menu
                elif ("GOOGLE" in p) or ("SEARCH" in p) or ("WEB BROWSER" in p) or ("CHROME" in p) or ("BROWSER" in p) or ("4" in p):
                    st.write("Opening Google Chrome")
                    pyttsx3.speak("Opening Google Chrome")
                    print(".")
                    print(".")
                    os.system("start chrome.exe")

                elif ("IE" in p) or ("MSEDGE" in p) or ("EDGE" in p) or ("8" in p):
                    st.write("Opening Microsoft Edge")
                    pyttsx3.speak("Opening Microsoft Edge")
                    
                    #pyttsx3.speak("MICROSOFT EDGE")
                    print(".")
                    print(".")
                    os.system("start msedge.exe")
                    
                elif ("NOTE" in p) or ("NOTES" in p) or ("NOTEPAD" in p) or ("EDITOR" in p) or ("9" in p):
                    st.write("Opening Notepad")
                    pyttsx3.speak("Opening Notepad")
                    
                    #pyttsx3.speak("NOTEPAD")
                    print(".")
                    print(".")
                    os.system("Notepad")

                elif ("VLCPLAYER" in p) or ("PLAYER" in p) or ("VIDEO PLAYER" in p) or ("5" in p):
                    st.write("Opening VLC Media Player")
                    pyttsx3.speak("Opening VLC Media Player")
                    print(".")
                    print(".")
                    os.system("start vlc.exe")

                elif ("EXCEL" in p) or ("MSEXCEL" in p) or ("SHEET" in p) or ("WINEXCEL" in p) or ("3" in p):
                    pyttsx3.speak("Opening Microsoft Excel")
                    st.write("Opening Microsoft Excel")
                    print(".")
                    print(".")
                    os.system("start excel.exe")
                    
                elif ("SLIDE" in p) or ("MSPOWERPOINT" in p) or ("PPT" in p) or ("POWERPNT" in p) or ("2" in p):
                    st.write("Opening Microsoft Powerpoint")
                    pyttsx3.speak("Opening Microsoft Powerpoint")
                    
                    print(".")
                    print(".")
                    os.system("start powerpnt.exe")

                elif ("WORD" in p) or ("MSWORD" in p) or ("1" in p):
                    st.write("Opening Microsoft Word")
                    pyttsx3.speak("Opening Microsoft Word")
                    
                    print(".")
                    print(".")
                    os.system("start winword.exe")
                    

                

            # close the program
                elif ("EXIT" in p) or ("QUIT" in p) or ("CLOSE" in p) or ("0" in p):
                    st.write("Exiting")
                    pyttsx3.speak("Exiting")
                    break

            # for ivalid input
                else:
                    st.write(p + "Is Invalid, Please Try Again")
                    pyttsx3.speak(p)
                    pyttsx3.speak("Is Invalid, Please Try Again")
                    
                    print("Is Invalid, Please Try Again")
                    
                    pyttsx3.speak("is Invalid,Please try again")
                    print(".")
                    print(".")


        elif(labels_pred == np.array([8])):
            import pywhatkit
            r = sr.Recognizer()
            mic = sr.Microphone()
            with mic as source:
                print("Tell me the reciever's mobile number: ")
                st.write("Tell me the reciever's mobile number: ")
                pyttsx3.speak("Tell me the reciever's mobile number: ")
                mobile_audio = r.listen(source)
            mobile_text = r.recognize_google(mobile_audio)
            mobile = "+91" + mobile_text
            mobile = mobile.replace(" ", "")
            print(mobile)
            st.write(mobile)
            #mobile = input("Enter reciever's mobile number with country code: ")
            with mic as soure:
                st.write("Dictate the message")
                pyttsx3.speak("Dictate the message:")
                print("Dictate the message:")
                message_audio = r.listen(source)
            message = r.recognize_google(message_audio)
            st.write(message)
            print(message)
            #message = input("Enter message you want to send: ")
            with mic as soure:
                st.write("Tell me the hour at which you want to send in the 24 hour format:")
                pyttsx3.speak("Tell me the hour at which you want to send in the 24 hour format:")
                print("Tell me the hour at which you want to send (0 - 23):")
                hours_audio = r.listen(source)
            hours = int(r.recognize_google(hours_audio))
            print(hours)
            st.write(hours)
            #hours = int(input("Enter the hours when you want to send: "))
            with mic as soure:
                st.write("Tell me the minute at which you want to send: ")
                pyttsx3.speak("Tell me the minute at which you want to send: ")
                print("Tell me the minute at which you want to send (0 - 59):")
                minutes_audio = r.listen(source)
            minutes = int(r.recognize_google(minutes_audio))
            print(minutes)
            st.write(minutes)
            #minutes = int(input("Enter the minutes when you want to send: "))
            st.write("Please press Enter key to send the message when whatsapp opens")
            pyttsx3.speak("Please press Enter key to send the message when whatsapp opens")
            pywhatkit.sendwhatmsg(mobile, message, hours, minutes, wait_time=0)

        else:
            print("Try again")
            st.write("try Again")
            pyttsx3.speak("Try Again")


quit()