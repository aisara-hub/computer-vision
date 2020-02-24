import os
from os import urandom
import numpy as np
import cv2

from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model,model_from_json
import configs

# Load pretrained Inception-ResNet-v1 model
# Update model and weights path according to your working environment

model_path = "Models/Inception_ResNet_v1.json"
weights_path = "Models/facenet_keras.h5"

json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
enc_model = model_from_json(loaded_model_json)
enc_model.load_weights(weights_path)

mtcnn_detector = MTCNN()

known_faces_encodings = []
known_faces_ids = []

known_faces_path = "Face_database/"



class faceRecog():

    def __init__(self, src=0):
        for filename in os.listdir(known_faces_path):
            print("load Keras Model")
            # Detect faces
            face = self.detect_face(known_faces_path + filename, normalize=True)

            # Compute face encodings

            feature_vector = enc_model.predict(face.reshape(1, 160, 160, 3))
            feature_vector /= np.sqrt(np.sum(feature_vector ** 2))
            known_faces_encodings.append(feature_vector)

            # Save Person IDs
            label = filename.split('.')[0]
            known_faces_ids.append(label)

        self.known_faces_encodings = np.array(known_faces_encodings).reshape(len(known_faces_encodings), 128)
        self.known_faces_ids = np.array(known_faces_ids)



    # Function to detect and extract face from a image
    def detect_face(self, filename, required_size=(160, 160), normalize=True):
        img = Image.open(filename)

        # convert to RGB
        img = img.convert('RGB')

        # convert to array
        pixels = np.asarray(img)

        # detect faces in the image
        results = mtcnn_detector.detect_faces(pixels)

        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']

        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face = pixels[y1:y2, x1:x2]

        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)

        if normalize == True:

            mean = np.mean(face_array, axis=(0, 1, 2), keepdims=True)
            std = np.std(face_array, axis=(0, 1, 2), keepdims=True)
            std_adj = np.maximum(std, 1.0)
            return (face_array - mean) / std

        else:
            return face_array



    # Function to recognize a face (if it is in known_faces)

    def recognize(self, img, known_faces_encodings, known_faces_ids, threshold=0.75):
        scores = np.zeros((len(known_faces_ids), 1), dtype=float)

        enc = enc_model.predict(img.reshape(1, 160, 160, 3))
        enc /= np.sqrt(np.sum(enc ** 2))

        scores = np.sqrt(np.sum((enc - known_faces_encodings) ** 2, axis=1))

        match = np.argmin(scores)

        if scores[match] > threshold:

            return ("UNKNOWN", 0)

        else:

            return (known_faces_ids[match], scores[match])


    # Function to perform real-time face recognition through a webcam

    def face_recognition(self, mode, frame,detector='haar', threshold=0.75):
        if detector == 'haar':
            # Load the cascade
            face_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')

        if mode == 'stream':

            # To capture webcam feed. Change argument for differnt webcams
            self.frame = frame

        elif mode == 'video':
            # To capture video feed
            self.frame = frame

        #while True:

        # Detect from frame
        if detector == 'haar':

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        elif detector == 'mtcnn':

            results = mtcnn_detector.detect_faces(frame)

            """if (len(results) == 0):
                continue"""

            faces = []

            for i in range(len(results)):
                x, y, w, h = results[i]['box']
                x, y = abs(x), abs(y)
                faces.append([x, y, w, h])

        # Draw the rectangle around each face and save
        count = 0
        for (x, y, w, h) in faces:
            image = Image.fromarray(frame[y:y + h, x:x + w])
            image = image.resize((160, 160))
            face_array = asarray(image)

            # Normalize
            mean = np.mean(face_array, axis=(0, 1, 2), keepdims=True)
            std = np.std(face_array, axis=(0, 1, 2), keepdims=True)
            std_adj = np.maximum(std, 1.0)
            face_array_normalized = (face_array - mean) / std

            # Recognize
            print("checking storage")
            configs.assure_path_exists(configs.STORAGE_PATH)
            count += 1
            label = self.recognize(face_array_normalized, self.known_faces_encodings, self.known_faces_ids, threshold)
            if label[0] != "UNKNOWN":
                # Save the captured image into the datasets folder if detected face
                print("saving to storage")
                # if using haarcascade
                # cv2.imwrite(configs.STORAGE_PATH+"/User_" + str(urandom(7).hex()) + '_' + str(count) + ".jpg", image[y:y+h,x:x+w])

                cv2.imwrite(configs.STORAGE_PATH + "/%s_" % (label[0]) + str(urandom(7).hex()) + '_' + str(count) + ".jpg",frame[y:y + h, x:x + w])
                #put rectangle is not needed
                """cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                cv2.putText(frame, label[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)"""

            # Display
            #cv2.imshow('Face_Recognition', img)





