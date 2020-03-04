# libraries
import cv2
import os
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from tensorflow.keras.models import load_model, model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from datetime import datetime

import configs

# https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

detector = MTCNN()

# steps
# 1. Build system for list of know faces
#       a. process face
#       b. embedding of face
#       c. save locally for retrieval

class ProcessingFace:
    '''
    Preprocessing images of faces given
    '''
    def __init__(self, directory=None, required_size=(160, 160)):
        if directory:
            self.directory = directory
        else:
            self.directory = "Face_database/"
        self.required_size = required_size
        self.model_path = "Models/Inception_ResNet_v1.json"
        self.weights_path = "Models/facenet_keras.h5"
        # loading FaceNet model from Hiroki Taniai (should be able to directly load)
        self.model = load_model(self.weights_path)

    def load_dataset(self):
        X, y = list(), list()
        # enumerate folders, on per class
        for subdir in os.listdir(self.directory):
            # path
            path = self.directory + subdir + '/'
            # skip any files that might be in the dir
            if not os.path.isdir(path):
                continue
            # load all faces in the subdirectory
            faces = self.load_images(path)
            # create labels
            labels = [subdir for _ in range(len(faces))]
            # summarize progress
            print('>loaded %d examples for class: %s' % (len(faces), subdir))
            # store
            X.extend(faces)
            y.extend(labels)
        return np.asarray(X), np.asarray(y)

    def load_images(self, subdir):
        faces = list()
        # enumerate files
        for filename in os.listdir(subdir):
            # path
            path = subdir + filename
            # get face
            try:
                face = self.processing_face(path)
                faces.append(face)
            except Exception as e:
                pass
            # store
        return faces

    def processing_face(self, path):
        # load image from file
        image = Image.open(path)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(self.required_size)
        face_array = np.asarray(image)
        return face_array

    def embedding_face(self, face):
        # scale pixel values
        face = face.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face, axis=0)
        # make prediction to get embedding
        yhat = self.model.predict(samples)
        return yhat[0]

    def save_face(self):
        X, y = self.load_dataset()
        trainX = list()
        for face in X:
            embedding = self.embedding_face(face)
            trainX.append(embedding)
        trainX = np.asarray(trainX)
        # normaliser - vector comparison
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        np.savez_compressed('faces-embeddings.npz', trainX, y)
        return {'message':True}

# 2. Build system for faces detected by system
#       a. process face
#       b. embeddings

# 3. Predictive model
#       a. load known embeddings
#       b. normalisation
#       c. encoding
#       d. choose a model
#       e. train and predict

class PredictingFaces:
    """
    Predicting input faces from known inputs

    Assumptions:
        - processing of face done in detector
            - resize
            - numpy array
    """
    def __init__(self, known_embedding=None):
        if known_embedding:
            self.known_embedding = known_embedding
        else:
            self.known_embedding = "faces-embeddings.npz"
        data = np.load(self.known_embedding)
        self.X, self.y = data['arr_0'], data['arr_1']
        # normalizer
        self.in_encoder = Normalizer(norm='l2')
        # label encode targets
        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(self.y)
        self.y_encoded = self.out_encoder.transform(self.y)
        # predictive model
        self.predictive_model = SVC(kernel='linear', probability=True)
        self.predictive_model.fit(self.X, self.y_encoded)
        # face embedding model
        self.model_path = "Models/Inception_ResNet_v1.json"
        self.weights_path = "Models/facenet_keras.h5"
        # loading FaceNet model from Hiroki Taniai (should be able to directly load)
        self.model = load_model(self.weights_path)

    def embedding_face(self, face):
        # scale pixel values
        face = face.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face, axis=0)
        # make prediction to get embedding
        yhat = self.model.predict(samples)
        # normalisation - vector comparison
        predicted = self.in_encoder.transform([yhat[0]])
        return predicted

    def saving_image(self, imageface, known="UNKNOWN", imagepath=configs.STORAGE_PATH, timestamp=datetime.now().timestamp()):
        # save the image, default saving to unknown faces
        cv2.imwrite(imagepath+"/"+known+"_" + str(timestamp)  + ".jpg", imageface)


    def predict_face_svm(self, face, threshold=0.5):
        # getting embeddings
        face_embedding = self.embedding_face(face)
        # prediction using embedding
        yhat_class = self.predictive_model.predict(face_embedding)
        yhat_probability = self.predictive_model.predict_proba(face_embedding)
        class_index = yhat_class[0]
        class_probability = yhat_probability[0, class_index]
        predict_names = self.out_encoder.inverse_transform(yhat_class)
        # print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        if class_probability > threshold:
            self.saving_image(face, imagepath="static/known", known=predict_names, timestamp=datetime.now().timestamp())
        else:
            self.saving_image(face, imagepath="static/unknown", timestamp=datetime.now().timestamp())

    def predict_face_distance(self, face, threshold=0.75):
        # getting embeddings
        face_embedding = self.embedding_face(face)
        # face_embedding /= np.sqrt(np.sum(face_embedding ** 2))
        scores = np.sum((face_embedding - self.X) ** 2, axis=1)
        match = np.argmin(scores)
        # print(self.y[match], "....",scores[match])
        if scores[match] < threshold:
            self.saving_image(face, imagepath="static/known", known=self.y[match], timestamp=datetime.now().timestamp())
        else:
            self.saving_image(face, imagepath="static/unknown", timestamp=datetime.now().timestamp())

# uncomment if running manually to update face training
# ProcessingFace().save_face()



