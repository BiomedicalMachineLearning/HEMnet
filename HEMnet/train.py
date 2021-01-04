# Example command:
# python /scratch/imb/Xiao/HEMnet/HEMnet/train.py -b /scratch/imb/Xiao/HE_test/10x/ -t train_dataset_10x_19_12_19_strict_Reinhard/tiles_10x/ -l valid_Reinhard/tiles_10x -o HEMnet_14_01_2020 -g 2 -e 10 -s -m vgg16 -a 64 -v
import argparse
import numpy as np
import os
import time
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from model import HEMnetModel


def get_class_weights(generator):
    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(generator.classes),
        generator.classes)
    return class_weights


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type = Path,
                        help = 'Base Directory')
    parser.add_argument('-t', '--train_dir', type = Path,
                        help = 'Directory containing training input tiles - relative to base directory')
    parser.add_argument('-l', '--valid_dir', type=Path,
                        help='Directory containing validation input tiles - relative to base directory')
    parser.add_argument('-o', '--out_dir', type = Path, default=Path(),
                        help = 'Output Directory - relative to base directory')
    parser.add_argument('-m', '--cnn_base', type=str, default="xception",
                        help='pre-trained convolutional neural network base')
    parser.add_argument('-g', '--num_gpus', type=int, default=2,
                        help='Number of GPUs for training model')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs for training model')
    parser.add_argument('-a', '--batch_size', type=int, default=32,
                        help='Number of tiles for each batch')
    parser.add_argument('-s', '--save_model', action = 'store_true',
                        help = 'save model weights')
    parser.add_argument('-v', '--verbosity', action = 'store_true',
                        help = 'Increase output verbosity')
    parser.add_argument('-w', '--transfer_learning', action='store_true',
                        help='Use CNN base pre-trained from ImageNet')
    parser.add_argument('-f', '--fine_tuning', action='store_true',
                        help='Fine-tuning pre-trained model')


    args = parser.parse_args()

    ####################
    # Paths and Inputs #
    ####################
    # Paths
    BASE_PATH = args.base_dir
    TRAIN_INPUT_PATH = BASE_PATH.joinpath(args.train_dir)
    VALID_INPUT_PATH = BASE_PATH.joinpath(args.valid_dir)
    OUTPUT_PATH = BASE_PATH.joinpath(args.out_dir)

    # User selectable parameters
    SAVE_MODEL = args.save_model
    CNN_BASE = args.cnn_base
    NUM_GPUS = args.num_gpus
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    TRANSFER_LEARNING = args.transfer_learning
    FINE_TUNING = args.fine_tuning
    VERBOSE = args.verbosity


    # Verbose functions
    if VERBOSE:
        verbose_print = lambda *args: print(*args)
        verbose_save_img = lambda img, path, img_type: img.save(path, img_type)
        verbose_save_fig = lambda fig, path, dpi=300: fig.savefig(path, dpi=dpi)
    else:
        verbose_print = lambda *args: None
        verbose_save_img = lambda *args: None
        verbose_save_fig = lambda *args: None

    HEMnet = HEMnetModel(cnn_base=CNN_BASE,
                         num_gpus=NUM_GPUS,
                         transfer_learning=TRANSFER_LEARNING,
                         fine_tuning=FINE_TUNING)
    input_size = (HEMnet.get_input_shape()[0], HEMnet.get_input_shape()[1])

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=360,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2)
    train_generator = train_datagen.flow_from_directory(TRAIN_INPUT_PATH,
                                                        classes=['cancer', 'non-cancer'],
                                                        target_size=input_size,
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='binary',
                                                        shuffle=True)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_generator = valid_datagen.flow_from_directory(VALID_INPUT_PATH,
                                                        classes=['cancer', 'non-cancer'],
                                                        target_size=input_size,
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='binary',
                                                        shuffle=True)

    HEMnet.train(train_generator, valid_generator, EPOCHS)

    OUTPUT_PATH = OUTPUT_PATH.joinpath('training_results')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    HEMnet.save_training_results(OUTPUT_PATH)
    if SAVE_MODEL:
        model_save_path = OUTPUT_PATH.joinpath("trained_model.h5")
        HEMnet.save_model(model_save_path)




