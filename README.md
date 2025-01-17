# Face Recognition and Voice Command Program

This project is a Python-based application that performs facial recognition using dlib with CUDA support, and responds to voice commands using the Vosk speech recognition library. The program can register new faces, recognize known faces, and respond to voice commands to close the application or describe its capabilities.

## Features
- Facial recognition with CUDA acceleration
- Voice command recognition
- Registration of new faces with multiple images
- Real-time video processing
- Welcome message for recognized faces
- Voice command to close the application

## Prerequisites
- Python 3.6 or higher
- CUDA-enabled GPU (for CUDA acceleration)

## Installation
1. Clone the Repository
```` 
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build . --config Release
cd ..
python setup.py install
```` 
2. Install Required Packages
````
pip install numpy
pip install face_recognition
pip install pygame
pip install gtts
pip install vosk
pip install pyaudio
````
3. Download Vosk Model
Download the Vosk speech recognition model from the following link and place it in the root directory of the project [Link](https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip). 
Unzip the downloaded file in the project root. Others languages supported in [Link](https://alphacephei.com/vosk/models)


## Usage
### Running the Program
To start the program, run the following command:

````
python register_person.py
````

### Program Options

#### Register a New Person

Enter 'r' to register a new person.
Follow the instructions to capture multiple images of the person's face.
Press 'SPACE' to capture an image and 'ESC' to finish the registration.

#### Start Video Recognition

Enter 'v' to start video recognition.
The program will recognize known faces and display a welcome message.
* Say "close" to terminate the application.
* Say "what can you do" to hear a description of the program's capabilities.

## File Structure
* register_person.py: Main script to run the program.
* face_encodings.pkl: File to store face encodings and names.
* vosk-model-en-us-0.42-gigaspeech: Directory containing the Vosk speech recognition model.

## Acknowledgements
* dlib: A toolkit for making real-world machine learning and data analysis applications.
* face_recognition: The world's simplest facial recognition API for Python and the command line.
* vosk: Offline speech recognition API based on Kaldi and Vosk.

## License
This project is licensed under the MIT License - see the LICENSE file for details.