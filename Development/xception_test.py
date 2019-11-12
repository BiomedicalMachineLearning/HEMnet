from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.utils import multi_gpu_model
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import xception
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
os.environ['QT_QPA_PLATFORM']='offscreen'
from tensorflow.keras.models import load_model

def get_class_weights(generator):
    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(generator.classes),
        generator.classes)
    return class_weights


def xception_model(tile_shape):
    xception_base = Xception(weights='imagenet', include_top=False)
    x = xception_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=xception_base.input, outputs=preds)
    parallel_model = multi_gpu_model(model, gpus=2, cpu_merge=False)
    parallel_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return parallel_model, model


def mkdirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def save_result(result_list, out_path):
    result_np = np.array(result_list)
    np.save(out_path + ".npy", result_np)


if __name__ == '__main__':
    train_input_path = '/scratch/imb/Xiao/HE_test/10x/'
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    total_predicted = []
    total_actual = []
    test_datagen = ImageDataGenerator(preprocessing_function=xception.preprocess_input)
    test_generator = test_datagen.flow_from_directory(train_input_path + 'test',
                                                      target_size=(299, 299),
                                                      color_mode='rgb',
                                                      batch_size=1,
                                                      class_mode='binary',
                                                      shuffle=False)
    #parallel_model, model = xception_model((299, 299, 3))
    parallel_model = load_model("./HE_xception_model.h5")
    y_pred = parallel_model.predict_generator(generator=test_generator, verbose=1)
    total_predicted.append(y_pred)
    y_true = test_generator.classes
    total_actual.append(y_true)
    out_path = os.path.join("/scratch/imb/Andrew", "HE_results_2171")
    mkdirs(out_path)
    save_result(total_actual, os.path.join(out_path, "total_actual"))
    save_result(total_predicted, os.path.join(out_path, "total_predicted"))
    print(test_generator)
    file_name = test_generator.filenames
    save_result(file_name, os.path.join(out_path, "file_name"))
    





























