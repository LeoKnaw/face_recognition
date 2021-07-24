import PySimpleGUI as sg
import cv2
import numpy as np
import threading
import re
import os, shutil
import pyrebase
import face_recognition
from threadedStream import StreamGet
import dlib
import sys

datFile =  r"C:\Users\edafe\OneDrive\Desktop\abuadProject\shape_predictor_68_face_landmarks.dat"
dlib.shape_predictor(datFile)


total = 1

# Firebase initiation
firebaseConfig = {
    "apiKey": "AIzaSyD4CNu1ohJIX8L3T08tG8cl4-j7YLoj0wY",
    "authDomain": "awesome-cda6c.firebaseapp.com",
    "projectId": "awesome-cda6c",
    "storageBucket": "awesome-cda6c.appspot.com",
    "messagingSenderId": "710482971561",
    "appId": "1:710482971561:web:821a81fd139911ea8712b8",
    "measurementId": "G-LMT0SMJBCQ",
    "databaseURL": "https://awesome-cda6c-default-rtdb.firebaseio.com/"
    }

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()



def main():
    matric = "None"
    sg.theme("LightGreen")

    # Define the window layout
    layout = [
        [sg.Text("Matric Number", size=(12,1)), sg.Input(key="matric")],
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Button("Capture", size=(10, 1)), sg.Button("Exit", size=(10,1))],
    ]

    # Create the window and show it without the plot
    window = sg.Window("Face Training Model", layout, location=(800, 400))
    cap = StreamGet(src=0).start() # intialising the camera

    detect = cv2.CascadeClassifier(r"C:\Users\edafe\OneDrive\Desktop\abuadProject\haarcascade_frontalface_default.xml") # Viola jones cascade classifier
    total = 1

    # This function converts the numpy array to normal arrays because firebase doesnt like numpy arrays
    def convertToJson(array):
        dbArr = []
        for enc in array:
            # converting each numpy array list ot a normal list
            # because firebase doesnt accept numpy arrays
            ndList = enc.tolist()
            dbArr.append(ndList)
        return dbArr

    # This function checks whether a face is detected
    def detectFace(image):
        rects = detect.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5,
                                        minSize=(50, 50))

        if len(rects) == 1:

            return True
        return False

    def takePicture():
        global total
        print("Finding face...")
        while total <= 10: # continues this block of code 10 times
            if detectFace(cap.frame): # if a face is detected
                cv2.imwrite(f"{str(total).zfill(4)}.jpg", cap.frame) # write to a jpg file
                print("taken")
                total += 1
        total=1

        encode = encodeFaces() # here we have an array of 10 numpy arrays.
        # Each array representing a sample of a face taken
        print(encode)
        print(len(encode))
        if len(encode) != 10: # If the number of numpy array is not up to 10, then we retake the pictures
            takePicture()
        else:
            os.chdir("..") # going back by 1 directory
            shutil.rmtree("arbitrary") # removing that arbitrary folder we created earlier
            encodeJson = convertToJson(encode) #caonverts to normal array because firebase doesnt like numpy arrays

            # sending the arrays to the database
            data = {matric: encodeJson}
            db.child("students").child(matric).set(data)


# 18/SCI01/109
    def encodeFaces(): # encodes the faces
        encodedFaces = []
        print(os.listdir()) # the current directory
        for dir in os.listdir():

             face_image = face_recognition.load_image_file(f"{dir}") # loads the image to encode
             try:
                 encoding = face_recognition.face_encodings(face_image)[0] # gets the encoding of the loaded face
                 encodedFaces.append(encoding) # adds the encoded face to the empty list above
             except:
                 continue
        return encodedFaces

    def encodeFaces_threaded():  # still encodes the faces but in a new thread
        global total
        encode = encodeFaces()
        print(encode)

        print(len(encode))

        # here we are removing the arbitrary directory we created earlier
        os.chdir("..")
        shutil.rmtree("arbitrary")
        # passing the encoding of the face to the database
        total = 1
        print(total)

    # main GUI loop
    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            cap.stop()
            break
        elif event == "Capture": # if capture is clicked
            print(values)
            matric = values["matric"]
            regex = re.match("^\d\d/[SCI]*[LAW]*[SMS]*[MHS]*[ENG]*0\d/\d\d\d", matric)

            if regex is None:  # if the matric number regex is None
                sg.popup("Error", "The matric number is not in the right format")
            else:
                oldMatric = matric[regex.span()[0]:regex.span()[1]]
                matric = oldMatric.replace("/", "-")
                try:
                    os.mkdir("arbitrary")  # create an arbitrary folder
                    os.chdir("arbitrary")  # change directory to the arbitrary folder
                except FileExistsError:  # if the folder already exists...
                    shutil.rmtree("arbitrary")  # remove the directory
                    os.mkdir("arbitrary")  # add it again
                    os.chdir("arbitrary")  # Changes to that path

                x = threading.Thread(target=takePicture, args=())
                x.daemon = True
                x.start()


                sg.popup("Success", f"Samples for {oldMatric} have been registered")
            #print(type(bit))


        frame = cap.frame
        rects = detect.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                        scaleFactor=1.1,
                                        minNeighbors=3,
                                        minSize=(50, 50))

        for x, y, w, h in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    window.close()


if __name__ == "__main__":
    main()
