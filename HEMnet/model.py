import sys
import importlib
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model



class HEMnetModel(object):
    """

    """
    __name__ = "HEMnet model"
    model_base_info = {"resnet50": ("ResNet50", (224, 224, 3)),
                       "vgg16": ("VGG16", (224, 224, 3)),
                       "vgg19": ("VGG19", (224, 224, 3)),
                       "inception_v3": ("InceptionV3", (224, 224, 3)),
                       "xception": ("Xception", (299, 299, 3))}

    def __init__(self, cnn_base="xception", num_gpus=2):
        self.cnn_base = cnn_base
        self.num_gpus = num_gpus
        self.keras_model = self.get_model_base()
        self.input_shape = self.get_input_shape()
        self.data_format = K.image_data_format()
        self.keras_application = self.import_keras_module()
        self.parallel_model, self.model = self.build_model()
        self.history = None
        self.total_val_loss = None
        self.total_train_accuracy = None
        self.total_train_loss = None
        self.total_val_accuracy = None
        self.y_pred = None
        self.y_true = None
        self.file_name = None

    def get_model_base(self):
        if self.cnn_base in self.model_base_info:
            return self.model_base_info[self.cnn_base][0]
        else:
            print("ERROR: Wrong Model: " + self.cnn_base + ".\n"
                  "Please choose one from: " + ', '.join([key for key in self.model_base_info]))
            sys.exit(1)

    def get_input_shape(self):
        return self.model_base_info[self.cnn_base][1]

    def import_keras_module(self):
        try:
            keras_module = importlib.import_module("tensorflow.keras.applications." + self.cnn_base)
        except ModuleNotFoundError as err:
            print("ERROR: Model not found in Keras application")
            sys.exit(1)
        return keras_module

    def build_model(self):
        model_base = getattr(self.keras_application, self.keras_model)(include_top=False,
                                                                       weights='imagenet',
                                                                       pooling="max",
                                                                       input_shape=self.input_shape)
        model = model_base.output
        model = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(model)
        model = Dense(256, activation='relu')(model)
        preds = Dense(1, activation='sigmoid')(model)
        model = Model(inputs=model_base.input, outputs=preds)
        try:
            parallel_model = multi_gpu_model(model, gpus=self.num_gpus, cpu_merge=False)
            print("Model using multiple GPUs..")
        except ValueError:
            parallel_model = model
            print("Model using single GPU or CPU..")
        parallel_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        return parallel_model, model

    def train(self, train_generator, valid_generator, epochs):
        steps_per_epoch = int(len(train_generator.classes)/train_generator.batch_size)
        validation_steps = int(len(valid_generator.classes)/valid_generator.batch_size)
        self.history = self.parallel_model.fit_generator(generator=train_generator,
                                                         steps_per_epoch=steps_per_epoch,
                                                         epochs=epochs,
                                                         validation_data=valid_generator,
                                                         validation_steps=validation_steps)
        self.total_train_loss = self.history.history['loss']
        self.total_val_loss = self.history.history['val_loss']
        self.total_train_accuracy = self.history.history['accuracy']
        self.total_val_accuracy = self.history.history['val_accuracy']


    @staticmethod
    def save_results(result_list, out_path):
        result_np = np.array(result_list)
        np.save(out_path, result_np)

    def save_model(self, out_path):
        self.model.save(out_path)

    def save_training_results(self, out_path):
        self.save_results(self.total_train_loss, out_path.joinpath("total_train_loss.npy"))
        self.save_results(self.total_val_loss, out_path.joinpath("total_val_loss.npy"))
        self.save_results(self.total_train_accuracy, out_path.joinpath("total_train_accuracy.npy"))
        self.save_results(self.total_val_accuracy, out_path.joinpath("total_val_accuracy.npy"))

    def load_model(self, model_path):
        self.parallel_model = load_model(model_path)

    def predict(self, test_generator):
        self.y_pred = self.parallel_model.predict_generator(generator=test_generator, verbose=1)
        self.y_true = test_generator.classes
        self.file_name = test_generator.filenames

    def save_test_results(self, out_path):
        self.save_results(self.y_pred, out_path.joinpath("total_predicted.npy"))
        self.save_results(self.y_true, out_path.joinpath("total_actual.npy"))
        self.save_results(self.file_name, out_path.joinpath("file_names.npy"))













