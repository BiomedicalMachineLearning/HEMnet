# Example command:
# python /scratch/imb/Xiao/HEMnet/HEMnet/test.py  -b /scratch/imb/Xiao/HE_test/10x/ -t 1957_T_Reinhard/tiles_10x -o 14_01_2020 -w 14_01_2020/training_results/trained_model_14_Jan_2020.h5 -m vgg16 -g 2 -v
import argparse
import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import HEMnetModel



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type = Path,
                        help = 'Base Directory')
    parser.add_argument('-t', '--test_dir', type = Path,
                        help = 'Directory containing test input tiles - relative to base directory')
    parser.add_argument('-o', '--out_dir', type = Path,
                        help = 'Output Directory - relative to Base Directory')
    parser.add_argument('-w', '--saved_model', type=Path,
                        help='Path to saved_model - relative to Base Directory')
    parser.add_argument('-m', '--cnn_base', type=str, default="xception",
                        help='pre-trained convolutional neural network base')
    parser.add_argument('-g', '--num_gpus', type=int, default=2,
                        help='Number of GPUs for training model')
    parser.add_argument('-v', '--verbosity', action = 'store_true',
                        help = 'Increase output verbosity')

    args = parser.parse_args()

    ####################
    # Paths and Inputs #
    ####################
    # Paths
    BASE_PATH = args.base_dir
    TEST_INPUT_PATH = BASE_PATH.joinpath(args.test_dir)
    OUTPUT_PATH = BASE_PATH.joinpath(args.out_dir)
    SAVED_MODEL = BASE_PATH.joinpath(args.saved_model)

    # User selectable parameters
    CNN_BASE = args.cnn_base
    NUM_GPUS = args.num_gpus
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

    HEMnet = HEMnetModel(cnn_base=CNN_BASE, num_gpus=NUM_GPUS)
    input_size = (HEMnet.get_input_shape()[0], HEMnet.get_input_shape()[1])

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(TEST_INPUT_PATH,
                                                      classes=['cancer', 'non-cancer'],
                                                      target_size=input_size,
                                                      batch_size=1,
                                                      class_mode='binary',
                                                      shuffle=False)

    HEMnet.load_model(SAVED_MODEL)
    HEMnet.predict(test_generator)

    OUTPUT_PATH = OUTPUT_PATH.joinpath('prediction_results')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    HEMnet.save_test_results(OUTPUT_PATH)

