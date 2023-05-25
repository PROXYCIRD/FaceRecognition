import cv2
import face_recognition
import numpy as np
from keras.models import load_model
import os

# Set up variables for face recognition
known_face_encodings = []
known_face_names = []

# Load the known faces from a folder of images
known_faces_folder = 'known_faces'
for filename in os.listdir(known_faces_folder):
    image = face_recognition.load_image_file(os.path.join(known_faces_folder, filename))
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Load the pre-trained emotion recognition model
model = load_model('emotion_model.h5')

# Define the emotions labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read the video frame
    frame: object
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # Iterate over each detected face for emotion recognition
    for (x, y, w, h) in faces:
        # Preprocess the face image for emotion recognition
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64))
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0

        # Predict the emotion
        emotions = model.predict(face)[0]
        emotion_index = np.argmax(emotions)
        emotion_label = emotion_labels[emotion_index]

        # Display the emotion label
        cv2.putText(frame, emotion_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    # Convert the frame from BGR to RGB for face recognition
    rgb_frame = frame[:, :, ::-1]

    # Detect all faces in the frame for face recognition
    face_locations = face_recognition.face_locations(rgb_frame)

    # Loop through each face and encode it for face recognition
    for (top, right, bottom, left) in face_locations:
        face_image = rgb_frame[top:bottom, left:right]
        face_encodings = face_recognition.face_encodings(face_image)

        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]

            # Check if the face is a match with any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If we have a match, use the name of the known face
            if True in matches:

                match_index = matches.index(True)
                name = known_face_names[match_index]

        # Draw a rectangle around the face and label it with the name for face recognition
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face and Emotion Recognition', frame)

# Check if the user wants to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
