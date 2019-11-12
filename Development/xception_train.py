from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.utils import multi_gpu_model
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import xception
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
os.environ['QT_QPA_PLATFORM']='offscreen'


def get_class_weights(generator):
    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(generator.classes),
        generator.classes)
    return class_weights


def xception_model(tile_shape):
    xception_base = Xception(weights='imagenet', include_top=False)
    #for i in range(len(xception_base.layers)):
    #    xception_base.layers[i].trainable = False
    x = xception_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
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
    total_val_loss = []
    total_train_accuracy = []
    total_train_loss = []
    total_val_accuracy = []
    total_actual = []
    total_predicted = []
    train_datagen = ImageDataGenerator(preprocessing_function=xception.preprocess_input)
    valid_datagen = ImageDataGenerator(preprocessing_function=xception.preprocess_input)
    train_generator = train_datagen.flow_from_directory(train_input_path + 'train',
                                                        target_size=(299, 299),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='binary',
                                                        shuffle=True)
    valid_generator = valid_datagen.flow_from_directory(train_input_path + 'valid',
                                                        target_size=(299, 299),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='binary',
                                                        shuffle=True)
    parallel_model, model = xception_model((299, 299, 3))
    train_history = parallel_model.fit_generator(generator=train_generator,
                                  steps_per_epoch=int(len(train_generator.classes) / train_generator.batch_size),
                                  epochs=500,
                                  validation_data=valid_generator,
                                  validation_steps=int(len(valid_generator.classes) / valid_generator.batch_size))
    loss = train_history.history['loss']
    total_train_loss.append(loss)
    val_loss = train_history.history['val_loss']
    total_val_loss.append(val_loss)
    acc = train_history.history['acc']
    total_train_accuracy.append(acc)
    val_acc = train_history.history['val_acc']
    total_val_accuracy.append(val_acc)
    model.save('HE_xception_model.h5')
    out_path = os.path.join(train_input_path, "results")
    mkdirs(out_path)
    save_result(total_train_loss, os.path.join(out_path, "total_train_loss"))
    save_result(total_val_loss, os.path.join(out_path, "total_val_loss"))
    save_result(total_train_accuracy, os.path.join(out_path, "total_train_accuracy"))
    save_result(total_val_accuracy, os.path.join(out_path, "total_val_accuracy"))





























