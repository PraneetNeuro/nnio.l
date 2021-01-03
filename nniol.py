from tensorflow import keras as tf
import cv2
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import os


class Dataset:
    def __init__(self, arch, path_of_dataset=None):
        self.n_classes = 0
        self.bad_data = []
        self.X = []
        self.Y = []
        self.classes = []
        self.maxOccuringShape = None
        if path_of_dataset is not None:
            self.path_of_dataset = path_of_dataset
            self.populate_dataset()
            self.one_hot_encoding()
            self.getMaxOccuringShape()
            self.normalize()
            if arch == "dense":
                self.flatten()
            self.convertToArray()

    def populate_dataset(self):
        for directory in os.listdir(self.path_of_dataset):
            if directory.startswith('.'):
                continue
            self.classes.append(directory)
            self.n_classes += 1
            for img in os.listdir(os.path.join(self.path_of_dataset, directory)):
                if img.startswith('.') or img.startswith('_'):
                    continue
                try:
                    self.X.append(cv2.imread(os.path.join(self.path_of_dataset, directory, img)))
                    self.Y.append(directory)
                except:
                    pass
        print('Classes: ', self.classes)

    def one_hot_encoding(self):
        encoder = LabelEncoder()
        self.Y = encoder.fit_transform(self.Y)
        self.Y = tf.utils.to_categorical(self.Y)

    def getMaxOccuringShape(self):
        shapes = []
        for i in range(len(self.X)):
            self.X[i] = np.array(self.X[i])
            if len(self.X[i].shape) > 1:
                shapes.append(self.X[i].shape)
        self.maxOccuringShape = Counter(shapes).most_common()
        print('Shape: ', self.maxOccuringShape[0][0])

    def normalize(self):
        for i in range(len(self.X)):
            try:
                self.X[i] = cv2.resize(self.X[i], self.maxOccuringShape[0][0][:2], self.X[i])
                self.X[i] = cv2.normalize(self.X[i], self.X[i], 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            except:
                self.bad_data.append(i)

    def flatten(self):
        for i in range(len(self.X)):
            self.X[i] = np.array(self.X[i]).ravel()

    def convertToArray(self):
        for i in self.bad_data:
            np.delete(self.Y, i)
            np.delete(self.X, i)
        self.X = np.asarray(self.X, dtype=np.float)
        self.Y = np.array(self.Y)


class DenseNet:
    def __init__(self, use_pretrained_model, path_of_dataset=None, neurons_per_layer=None, activations=None,
                 model_path=None, epochs=None):
        self.model_path = model_path
        self.model = tf.models.Model()
        self.use_pretrained_model = use_pretrained_model
        if use_pretrained_model:
            self.dataset = Dataset(arch='dense')
            items = list(os.listdir(model_path))
            if 'nnio.l.cfg' not in items:
                print('Err: Not a valid model path, Configuration missing')
                return
            with open(os.path.join(model_path, 'nnio.l.cfg'), 'r') as f:
                config = f.readlines()
                self.dataset.n_classes = int(config[1].replace('\n', ''))
                self.dataset.maxOccuringShape = config[2].replace('\n', '').replace('[', '').replace(']', '').replace(
                    '(', '').replace(')', '').split(',')[:2]
                self.dataset.maxOccuringShape = [int(i.replace(' ', '')) for i in self.dataset.maxOccuringShape]
                self.dataset.classes = config[0].replace('\n', '')
                print(
                    'Model initialized with:\n{}\n{}\n{}'.format(self.dataset.n_classes, self.dataset.maxOccuringShape,
                                                                 self.dataset.classes))
        else:
            assert path_of_dataset is not None and neurons_per_layer is not None and activations is not None and model_path is not None and epochs is not None, "Err: Required args not passed for object initialization"
            self.path_of_dataset = path_of_dataset
            self.neurons_per_layer = neurons_per_layer
            self.activations = activations
            self.epochs = epochs
            self.dataset = Dataset('dense', path_of_dataset)
            self.DenseNet()
            self.fit()

    def DenseNet(self):
        self.model = tf.models.Sequential()
        self.model.add(tf.Input([np.prod(self.dataset.maxOccuringShape[0][0])]))
        for i in range(len(self.neurons_per_layer)):
            self.model.add(tf.layers.Dense(self.neurons_per_layer[i], activation=self.activations[i]))
        self.model.add(tf.layers.Dense(self.dataset.n_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def summary(self):
        self.model.summary()

    def fit(self):
        self.model.fit(self.dataset.X, self.dataset.Y, epochs=self.epochs)
        self.model.save(self.model_path)
        with open(os.path.join(self.model_path, 'nnio.l.cfg'), 'w') as f:
            f.write(str(self.dataset.classes) + '\n')
            f.write(str(self.dataset.n_classes) + '\n')
            f.write(str(self.dataset.maxOccuringShape) + '\n')

    def predict(self, x):
        img = cv2.imread(x)
        if self.use_pretrained_model:
            self.model = tf.models.load_model(self.model_path)
            img = cv2.resize(img, tuple(self.dataset.maxOccuringShape), img)
        else:
            img = cv2.resize(img, self.dataset.maxOccuringShape[0][0][:2], img)
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        img = np.array(img)
        img = img.ravel()
        img = np.expand_dims(img, 0)
        print("Prediction: ", np.array(self.model.predict(img)).argmax())


class ConvNet:
    def __init__(self, use_pretrained_model, path_of_dataset=None, filters_per_layer=None, activations=None,
                 model_path=None, epochs=None):
        self.model_path = model_path
        self.model = tf.models.Model()
        self.use_pretrained_model = use_pretrained_model
        if use_pretrained_model:
            self.dataset = Dataset(arch='conv')
            items = list(os.listdir(model_path))
            if 'nnio.l.cfg' not in items:
                print('Err: Not a valid model path, Configuration missing')
                return
            with open(os.path.join(model_path, 'nnio.l.cfg'), 'r') as f:
                config = f.readlines()
                self.dataset.n_classes = int(config[1].replace('\n', ''))
                self.dataset.maxOccuringShape = config[2].replace('\n', '').replace('[', '').replace(']', '').replace(
                    '(', '').replace(')', '').split(',')[:2]
                self.dataset.maxOccuringShape = [int(i.replace(' ', '')) for i in self.dataset.maxOccuringShape]
                self.dataset.classes = config[0].replace('\n', '')
                print(
                    'Model initialized with:\n{}\n{}\n{}'.format(self.dataset.n_classes, self.dataset.maxOccuringShape,
                                                                 self.dataset.classes))
        else:
            assert path_of_dataset is not None and filters_per_layer is not None and activations is not None and model_path is not None and epochs is not None, "Err: Required args not passed for object initialization"
            self.path_of_dataset = path_of_dataset
            self.filters_per_layer = filters_per_layer
            self.activations = activations
            self.epochs = epochs
            self.dataset = Dataset('conv', path_of_dataset)
            self.ConvNet()
            self.summary()
            self.fit()

    def ConvNet(self):
        self.model = tf.models.Sequential()
        self.model.add(tf.Input((self.dataset.maxOccuringShape[0][0])))
        for i in range(len(self.filters_per_layer)):
            self.model.add(tf.layers.Conv2D(self.filters_per_layer[i], kernel_size=(3, 3), activation=self.activations[i]))
        self.model.add(tf.layers.Flatten())
        self.model.add(tf.layers.Dense(self.dataset.n_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def summary(self):
        self.model.summary()

    def fit(self):
        self.model.fit(self.dataset.X, self.dataset.Y, epochs=self.epochs)
        self.model.save(self.model_path)
        with open(os.path.join(self.model_path, 'nnio.l.cfg'), 'w') as f:
            f.write(str(self.dataset.classes) + '\n')
            f.write(str(self.dataset.n_classes) + '\n')
            f.write(str(self.dataset.maxOccuringShape) + '\n')

    def predict(self, x):
        img = cv2.imread(x)
        if self.use_pretrained_model:
            self.model = tf.models.load_model(self.model_path)
            img = cv2.resize(img, tuple(self.dataset.maxOccuringShape), img)
        else:
            img = cv2.resize(img, self.dataset.maxOccuringShape[0][0][:2], img)
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        img = np.array(img)
        img = np.expand_dims(img, 0)
        print("Prediction: ", np.array(self.model.predict(img)).argmax())
