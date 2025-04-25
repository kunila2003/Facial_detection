# -------------------- Importing Libraries --------------------

from sklearn.neighbors import KNeighborsClassifier  # For face recognition using KNN
import cv2                                         
import pickle                                        
import numpy as np                                   
import os                                            
import csv                                           # To read/write CSV attendance files
import time                                         
from datetime import datetime                       

from win32com.client import Dispatch                 # To use the text-to-speech feature on Windows

# -------------------- Text to Speech Function --------------------

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))               # Initialize Windows text-to-speech engine
    speak.Speak(str1)                                # Speak the input string

# -------------------- Initialize Video Capture --------------------

video = cv2.VideoCapture(0)                          # Start capturing video from webcam
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')  # Load Haar Cascade for face detection

# -------------------- Load Trained Data --------------------

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)                          # Load saved names (labels)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)                           # Load saved face data

print('Shape of Faces matrix --> ', FACES.shape)     # Print shape of loaded face data

# -------------------- Train KNN Model --------------------

knn = KNeighborsClassifier(n_neighbors=5)            # Initialize KNN classifier with k=5
knn.fit(FACES, LABELS)                               # Train the model with data and labels

# -------------------- Load Background Image --------------------

imgBackground = cv2.imread("background.png")         # Load a background image (for fancy UI)

# Column names for CSV attendance file
COL_NAMES = ['NAME', 'TIME']

# -------------------- Real-Time Face Detection & Recognition Loop --------------------

while True:
    ret, frame = video.read()                        # Read a frame from webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Convert the frame to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]            # Crop face from the frame
        resized_img = cv2.resize(crop_img, (50, 50)) # Resize to match trained image size
        resized_img = resized_img.flatten().reshape(1, -1)  # Flatten image for prediction

        output = knn.predict(resized_img)            # Predict the person's name

        # Get current timestamp
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")   # Date in format: dd-mm-yyyy
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")  # Time in format: hh:mm:ss

        # Check if attendance file for the date exists
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        # Draw rectangles and display name on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        attendance = [str(output[0]), str(timestamp)]  # Prepare attendance entry

    # Place the frame into the background image
    imgBackground[162:162 + 480, 55:55 + 640] = frame

    # Show the combined image
    cv2.imshow("Frame", imgBackground)

    k = cv2.waitKey(1)

    # Press 'o' to record attendance
    if k == ord('o'):
        speak("Attendance Taken..")                 # Speak out confirmation
        time.sleep(5)                               # Wait for 5 seconds before continuing

        if exist:
            # If file exists, just add the new entry
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            # If file doesn't exist, write column headers first
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()

    # Press 'q' to quit the program
    if k == ord('q'):
        break

# Release the webcam and close windows
video.release()
cv2.destroyAllWindows()
