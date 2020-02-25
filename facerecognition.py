import os
from os import urandom
import numpy as np
import cv2

from PIL import Image
from numpy import asarray
from mtcnn import MTCNN
from tensorflow.keras.models import load_model,model_from_json
import configs
from datetime import datetime

detector = MTCNN()
class FaceRecog():

    def __init__(self):
        # Load pretrained Inception-ResNet-v1 model
        # Update model and weights path according to your working environment

        self.known_faces_encodings = []
        self.known_faces_ids = []
        model_path = "Models/Inception_ResNet_v1.json"
        weights_path = "Models/facenet_keras.h5"
        known_faces_path = "Face_database/"
        mtcnn_detector = MTCNN()
        for filename in os.listdir(known_faces_path):
            print("load Keras Model")
            json_file = open(model_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.enc_model = model_from_json(loaded_model_json)
            self.enc_model.load_weights(weights_path)

            # Detect faces from image database to create vector based on keras model
            face = self.detect_face(known_faces_path + filename, normalize=True)

            # Compute face encodings
            feature_vector = self.enc_model.predict(face.reshape(1, 160, 160, 3))
            feature_vector /= np.sqrt(np.sum(feature_vector ** 2))
            self.known_faces_encodings.append(feature_vector)

            # Save Person IDs
            label = filename.split('.')[0]
            self.known_faces_ids.append(label)

        self.known_faces_encodings = np.array(self.known_faces_encodings).reshape(len(self.known_faces_encodings), 128)
        self.known_faces_ids = np.array(self.known_faces_ids)

    # Function to detect and extract face from a image
    def detect_face(self, filename, required_size=(160, 160), normalize=True):
        img = Image.open(filename)

        # convert to RGB
        img = img.convert('RGB')

        # convert to array
        pixels = np.asarray(img)

        # detect faces in the image
        results = detector.detect_faces(pixels)

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

        enc = self.enc_model.predict(img.reshape(1, 160, 160, 3))
        enc /= np.sqrt(np.sum(enc ** 2))

        scores = np.sqrt(np.sum((enc - known_faces_encodings) ** 2, axis=1))

        match = np.argmin(scores)

        if scores[match] > threshold:

            return ("UNKNOWN", 0)

        else:

            return (known_faces_ids[match], scores[match])

    # save the image, default saving to unknown faces
    def save_image_to(self, imageface, known="UNKNOWN", imagepath=configs.STORAGE_PATH, timestamp=datetime.now().timestamp()):
        cv2.imwrite(imagepath+"/"+known+"_" + str(timestamp)  + ".jpg", imageface)

    # Function to perform real-time face recognition through a webcam
    def face_recognition(self, faces, threshold=0.75):
        for face in faces:
            face_array = asarray(face)
            # Normalize face
            mean = np.mean(face_array, axis=(0, 1, 2), keepdims=True)
            std = np.std(face_array, axis=(0, 1, 2), keepdims=True)
            face_array_normalized = (face_array - mean) / std
            label = self.recognize(face_array_normalized, self.known_faces_encodings, self.known_faces_ids, threshold)
            if label[0] != "UNKNOWN":
                # Save the captured image into the datasets folder if detected face
                print("saving to storage")
                self.save_image_to(face_array, imagepath="static/known", known=label[0], timestamp=datetime.now().timestamp())
            else:
                self.save_image_to(face_array, imagepath="static/unknown", timestamp=datetime.now().timestamp())







