# Importing necessary libraries
import cv2              # OpenCV library for image processing
import pickle           # For saving data in binary format
import numpy as np      # For numerical operations
import os               # For interacting with the operating system

# Start capturing video from the default webcam (index 0)
video = cv2.VideoCapture(0)

# Load the Haar Cascade model for face detection
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# List to store captured face images
faces_data = []

# Counter for frame processing
i = 0

# Ask user to input their name (this will be used to label the data)
name = input("Enter Your Name: ")

# Start video capture loop
while True:
    ret, frame = video.read()  # Read a frame from webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame

    # Process each detected face
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        crop_img = frame[y:y+h, x:x+w, :]

        # Resize the face image to 50x50 pixels
        resized_img = cv2.resize(crop_img, (50, 50))

        # Append face data every 10 frames (up to 100 images)
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        
        # Increment frame counter
        i += 1

        # Display count of collected face images on the screen
        cv2.putText(frame, str(len(faces_data)), (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Show the frame in a window named "Frame"
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed or 100 face samples are collected
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

# Release the webcam and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Convert the list of face images to a NumPy array
faces_data = np.asarray(faces_data)

# Reshape it to (100, 7500), where 7500 = 50*50*3 (flattening the image)
faces_data = faces_data.reshape(100, -1)

# ------------------ Saving Name Labels ---------------------

# If names.pkl does not exist, create it with current user's name 100 times
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    # Load existing names and add new entries
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100  # Add new user's name
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# ------------------ Saving Face Data ---------------------

# If faces_data.pkl does not exist, create it with current face data
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    # Load existing face data and append new data
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)  # Combine datasets
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
