import cv2
import numpy as np
import face_recognition
import dlib
import pickle
from gtts import gTTS
from io import BytesIO
import pygame
import time
import vosk
import pyaudio
import json
import threading

# CUDA usage verification
use_cuda_dlib = dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() > 0
use_cuda_opencv = cv2.cuda.getCudaEnabledDeviceCount() > 0

print("DLIB_USE_CUDA:", use_cuda_dlib)
print("Number of CUDA devices for dlib:", dlib.cuda.get_num_devices())
print("OpenCV CUDA Support: ", cv2.cuda.getCudaEnabledDeviceCount())

# Initialize lists for known face encodings and names
known_face_encodings = []
known_face_names = []
welcomed_people = {}
welcome_timeout = 10  # 10 seconds timeout for welcome message

# Initialize pygame for audio playback
pygame.init()
pygame.mixer.init()

# Load the Vosk model for speech recognition
model_path = "vosk-model-en-us-0.42-gigaspeech"
model = vosk.Model(model_path)
rec = vosk.KaldiRecognizer(model, 16000)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=2048)  # Reduce buffer size

# Flag to indicate if the program should close
close_program = False

# Function to listen for the "close" command
def listen_for_close():
    global close_program
    while True:
        try:
            data = stream.read(2048, exception_on_overflow=False)  # Adjust buffer size and handle overflow
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                recognized_text = result['text']
                print(recognized_text)
                if "close" in recognized_text.lower():
                    print("Close command detected. Stopping...")
                    close_program = True
                    break
                elif "what can you do" in recognized_text.lower():
                    sound = speak(f"Now I can recognize people, but I could do some voice commands and generate actions.")
                    sound.seek(0)
                    pygame.mixer.music.load(sound, "mp3")
                    pygame.mixer.music.play()
        except Exception as e:
            print(f"Error in audio stream: {e}")
            break

# Function to load known faces from file
def load_known_faces():
    global known_face_encodings, known_face_names
    try:
        with open("face_encodings.pkl", "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)
    except FileNotFoundError:
        # If the file does not exist, create it
        with open("face_encodings.pkl", "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)

# Function to register a new person
def register_person(name, frames):
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)
        if face_encodings:
            face_encoding = face_encodings[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"Person {name} registered successfully with {len(frames)} images.")

# Function to capture multiple photos from camera and register person
def capture_and_register():
    video_capture = cv2.VideoCapture(0)
    name = input("Enter the name of the person: ")
    print("Press 'SPACE' to capture a photo. Press 'ESC' when done.")

    frames = []
    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            frames.append(frame)
            print(f"Captured image {len(frames)}")
        elif key == 27:  # ESC key
            if frames:
                register_person(name, frames)
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to draw a rotated rectangle
def draw_rotated_rectangle(image, rect, angle, color, thickness):
    # Get the center of the rectangle
    center = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
    # Get the size of the rectangle
    size = (rect[2] - rect[0], rect[3] - rect[1])
    # Create the rotation matrix
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Get the coordinates of the rectangle
    box = cv2.boxPoints((center, size, angle))
    box = np.intp(box)
    # Draw the rectangle
    cv2.drawContours(image, [box], 0, color, thickness)

# Function to speak a welcome message
def speak(text):
    mp3_fp = BytesIO()
    tts = gTTS(text, lang="en")
    tts.write_to_fp(mp3_fp)
    return mp3_fp

# Function to recognize faces in video
def recognize_faces_in_video():
    global close_program
    video_capture = cv2.VideoCapture(0)

    # Start the speech recognition thread
    threading.Thread(target=listen_for_close).start()

    while True:
        ret, frame = video_capture.read()

        if use_cuda_opencv:
            # Upload frame to the GPU
            gpu_frame = cv2.cuda.GpuMat()
            gpu_frame.upload(frame)

            # Convert the frame from BGR to RGB on the GPU
            gpu_rgb_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)

            # Download the frame back to the CPU
            rgb_frame = gpu_rgb_frame.download()
        else:
            # Convert the frame from BGR to RGB on the CPU
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the face locations and face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn" if use_cuda_dlib else "hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        current_time = time.time()

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            similarity_percentage = 0

            # Calculate the face distance to find the best match
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                similarity_percentage = (1 - face_distances[best_match_index]) * 100

                # Welcome the person if they haven't been welcomed recently
                if name not in welcomed_people or (current_time - welcomed_people[name]) > welcome_timeout:
                    welcomed_people[name] = current_time
                    sound = speak(f"Welcome, {name}")
                    sound.seek(0)
                    pygame.mixer.music.load(sound, "mp3")
                    pygame.mixer.music.play()
                    time.sleep(2)  # Wait for the message to finish playing
                elif name in welcomed_people and (current_time - welcomed_people[name]) < welcome_timeout:
                    welcomed_people[name] = current_time

            # Get facial landmarks for face rotation
            face_landmarks = face_recognition.face_landmarks(rgb_frame, [(top, right, bottom, left)])[0]
            # Calculate the angle of rotation based on the eye coordinates
            left_eye = np.array(face_landmarks['left_eye'])
            right_eye = np.array(face_landmarks['right_eye'])
            eye_center = np.mean([left_eye, right_eye], axis=1)
            angle = np.arctan2(eye_center[0, 1] - eye_center[1, 1], eye_center[0, 0] - eye_center[1, 0]) * 180.0 / np.pi

            # Draw the rotated rectangle
            draw_rotated_rectangle(frame, (left, top, right, bottom), angle, (0, 0, 255), 2)

            # Draw a label with a name and similarity percentage below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{name} ({similarity_percentage:.2f}%)"
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Check if the close command was detected
        if close_program:
            break

        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

# Load known faces from file
load_known_faces()

# Start video recognition or registration based on user input
while True:
    choice = input("Enter 'r' to register a new person or 'v' to start video recognition: ").lower()
    if choice == 'r':
        capture_and_register()
    elif choice == 'v':
        recognize_faces_in_video()
        break  # Exit the loop when the video recognition ends
    else:
        print("Invalid choice. Please enter 'r' or 'v'.")

# Close the microphone stream
stream.stop_stream()
stream.close()

# Terminate the PyAudio object
p.terminate()
