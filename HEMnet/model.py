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
    HEMnet model: convolutional neural network (CNN) model for automated cancer segmentation using molecular labelled
    Haematoxylin and Eosin (H&E) stained tissue.

    Attributes
    ----------
    cnn_base : str
        Name of established CNN base.
    num_gpus : int
        Number of GPUs implemented
    transfer_learning : bool
        Use CNN base trained from ImageNet or not
    fine_tuning : bool
        Fine-tuning CNN base or not
    input_shape : tuple int
        Input image shape (width, height, channels)
    parallel_model : keras Model object
        Parallel model for GPU computing
    model : keras Model object
        Compiled model
    history : dict
        Model training results


    Methods
    -------
    get_model_base()
        Get name of CNN base
    get_input_shape()
        Get input shae
    build_model()
        Build model
    train(train_generator, valid_generator, epochs)
        Train model on training data generator, valid on validation data generator
    save_model(out_path)
        Save trained model to path
    load_model(model_path)
        Load trained model from path
    predict(test_generator)
        Predict labels on test data generator
    save_training_results(out_path)

    """
    __name__ = "HEMnet model"
    model_base_info = {"resnet50": ("ResNet50", (224, 224, 3)),
                       "vgg16": ("VGG16", (224, 224, 3)),
                       "vgg19": ("VGG19", (224, 224, 3)),
                       "inception_v3": ("InceptionV3", (224, 224, 3)),
                       "xception": ("Xception", (299, 299, 3))}

    def __init__(self, cnn_base="xception", num_gpus=2, transfer_learning=True, fine_tuning=False):
        self.cnn_base = cnn_base
        self.num_gpus = num_gpus
        self.transfer_learning = transfer_learning
        self.fine_tuning = fine_tuning
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
        """Check CNN base

        Parameters
        ----------
        self

        Returns
        -------
        cnn_base or exit
        """
        if self.cnn_base in self.model_base_info:
            return self.model_base_info[self.cnn_base][0]
        else:
            print("ERROR: Wrong Model: " + self.cnn_base + ".\n"
                  "Please choose one from: " + ', '.join([key for key in self.model_base_info]))
            sys.exit(1)

    def get_input_shape(self):
        """Get tile input shape

        Parameters
        ----------
        self

        Returns
        -------
        tile input shape tuple(width, height, channels)
        """
        return self.model_base_info[self.cnn_base][1]

    def import_keras_module(self):
        """Import keras application module
        Parameters
        ----------
        self

        Returns
        -------
        keras application module
        """
        try:
            keras_module = importlib.import_module("tensorflow.keras.applications." + self.cnn_base)
        except ModuleNotFoundError as err:
            print("ERROR: Model not found in Keras application")
            sys.exit(1)
        return keras_module

    def build_model(self):
        """Build HEMnet model
        Parameters
        ----------
        self

        Returns
        -------
        gpu parallel model, compiled model
        """
        if self.transfer_learning:
            weights = 'imagenet'
        else:
            weights = None
        model_base = getattr(self.keras_application, self.keras_model)(include_top=False,
                                                                       weights=weights,
                                                                       pooling="max",
                                                                       input_shape=self.input_shape)
        if not self.fine_tuning:
            model_base.layers.trainable = False
        model = model_base.output
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
        """Train HEMnet model
        Parameters
        ----------
        train_generator: Keras ImageDataGenerator for generating training tile batches
        valid_generator: Keras ImageDataGenerator for generating validation tile batches
        epochs: Number of training iteration

        Returns
        -------
        None
        """
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
        """Save trained HEMnet model
        Parameters
        ----------
        out_path: Path for saving model

        Returns
        -------
        None
        """
        self.model.save(out_path)

    def save_training_results(self, out_path):
        """Save model training results
        Parameters
        ----------
        out_path: Path for saving Save model training results

        Returns
        -------
        None
        """
        self.save_results(self.total_train_loss, out_path.joinpath("total_train_loss.npy"))
        self.save_results(self.total_val_loss, out_path.joinpath("total_val_loss.npy"))
        self.save_results(self.total_train_accuracy, out_path.joinpath("total_train_accuracy.npy"))
        self.save_results(self.total_val_accuracy, out_path.joinpath("total_val_accuracy.npy"))

    def load_model(self, model_path):
        """Load saved model
        Parameters
        ----------
        model_path: Path for loading saved model

        Returns
        -------
        None
        """
        self.parallel_model = load_model(model_path)

    def predict(self, test_generator):
        """Predict labels for test dataset
        Parameters
        ----------
        test_generator: Keras ImageDataGenerator for generating test tile batches

        Returns
        -------
        None
        """
        self.y_pred = self.parallel_model.predict_generator(generator=test_generator, verbose=1)
        self.y_true = test_generator.classes
        self.file_name = test_generator.filenames

    def save_test_results(self, out_path):
        """Save model prediction results
        Parameters
        ----------
        out_path: Path for saving Save model prediction results

        Returns
        -------
        None
        """
        self.save_results(self.y_pred, out_path.joinpath("total_predicted.npy"))
        self.save_results(self.y_true, out_path.joinpath("total_actual.npy"))
        self.save_results(self.file_name, out_path.joinpath("file_names.npy"))













