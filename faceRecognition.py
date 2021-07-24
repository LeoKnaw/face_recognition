from threadedStream import StreamGet
from datetime import datetime
from numpy import array
import PySimpleGUI as sg
import face_recognition
import cv2
import pyrebase
import winsound
import threading
import dlib


datFile =  r"C:\Users\edafe\OneDrive\Desktop\abuadProject\shape_predictor_68_face_landmarks.dat"
dlib.shape_predictor(datFile)


# Firebase intiation
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
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Button("Take Attendance", size=(15, 1), key="take"), sg.Button("Exit", size=(10,1))],
    ]

    # Create the window and show it without the plot
    window = sg.Window("Facial Recognition", layout, location=(800, 400))
    cap = StreamGet(src=0).start()

    detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def encodeForEach():
        stud = db.child("students").get()  # get values from database in form of a dictionary
        new = {}
        for mat, matricData in stud.val().items():  # loop through key-value pairs at the same time
            # the encodings are a list of 128 lists so we have to turn each of them to a numpy array
            newEnc = []
            for m, enc in matricData.items():
                for i in enc:
                    newEnc.append(array(i))
                    new[mat] = newEnc
        return new

    def recognise():
        initialiseAttendanceFile()
        present = []
        faces = []

        while True:
            location = face_recognition.face_locations(frame, model="hog")
            encoding = face_recognition.face_encodings(frame, location)
            for matric, enc in encodeForEach().items():  # looping through the key-value pairs in the dictionary at the same time
                for encode, loc in zip(encoding, location):
                    # compare the encoding of the frame to the encodings in the database

                    result = face_recognition.compare_faces(enc, encode, 0.4)
                    # comparing the encodings of the faces in the database to the encoding of the frame

                    if True in result:
                        # print("Match found in " , result.index(True))
                        newMatric = matric.replace("-", "/")
                        # if the matric is not in the list
                        # if the attendance of the matric has not yet been taken
                        if newMatric not in present:
                            tl = (loc[3], loc[0]) # top left of the rectangle
                            br = (loc[1], loc[2]) #bottom right of the rectangle
                            cv2.rectangle(frame, tl, br, (48, 200, 43), 2)
                            takeAttendance(newMatric)
                            present.append(newMatric)
                            print(present)
                        # if the attendance of the matric has been taken
                        else:
                            tl = (loc[3], loc[0])
                            br = (loc[1], loc[2])
                            cv2.rectangle(frame, tl, br, (200, 0, 0), 2)
                    else:
                        tl = (loc[3], loc[0])
                        br = (loc[1], loc[2])

    def initialiseAttendanceFile():
        now = datetime.now()
        date = now.strftime("%d %B, %Y")
        with open(date + ".csv", "w") as f:
            f.write("Name,Attendance")

    def takeAttendance(name):
        now = datetime.now()
        date = now.strftime("%d %B, %Y")
        # initialiseAttendanceFile()
        with open(date + ".csv", "r+") as f:
            dataList = f.readlines() # puts a line into an index in a list
            names = []
            for line in dataList:
                entry = line.split(",")
                names.append(entry[0])
            if name not in names:
                f.writelines(f"\n{name},Present")
                winsound.Beep(4000, 500)

    # main GUI loop
    while True:
        frame = cap.frame

        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            cap.stop()
            break
        elif event == "take":
            rec = threading.Thread(target = recognise, args=())
            rec.daemon = True
            rec.start()


        # rects = detect.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=3,
        #                                 minSize=(50, 50))
        # for x, y, w, h in rects:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    window.close()

if __name__ == "__main__":
    main()



