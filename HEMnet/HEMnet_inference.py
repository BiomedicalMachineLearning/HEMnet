# ------------------------------------------------------------------------
# Predict cancer on H&E slides using a previously trained neural network
#
# Example command:
# python HEMnet_inference.py -s '/QRISdata/Q1139/ST_Projects/HEMnet/HEMnet_Data/TCGA/COAD' -o '/gpfs1/scratch/90days/s4436005/TCGA/02_08_20' -t '/gpfs1/scratch/90days/s4436005/Slides/1957_T_9668_3_HandE.svs' -nn '/QRISdata/Q1139/ST_Projects/HEMnet/Results/model/HE_vgg16_model_10x_23_07_20_strict_vahadane_224px.h5' -v
# ------------------------------------------------------------------------

import argparse
import cv2 as cv
import importlib
import io
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import multiprocessing as mp
import numpy as np
import openslide
from openslide import open_slide
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageChops, ImageDraw
import seaborn as sns
import sys
import time
import timeit
from tensorflow import keras

# Allow Pillow to open very big images
Image.MAX_IMAGE_PIXELS = None

# Import HEMnet package
BASE_DIR = Path().resolve()
HEMNET_DIR = BASE_DIR.joinpath('HEMnet')
sys.path.append(str(HEMNET_DIR))

from slide import *
from utils import *
from normaliser import IterativeNormaliser

#############
# Functions #
#############


def restricted_float(x):
    # Restrict argument to float between 0 and 1 (inclusive)
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError('{0} not a floating point literal'.format(x))
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError('{0} not in range [0.0, 1.0]'.format(x))
    return x


def tile_from_slide(x, y, wsi, mag, tile_size_at_mag):
    """Read a tile from a Whole Slide Image (WSI) based on tile x and y coordinates
    """
    # Calculate maximum number tiles
    width_mag, height_mag = get_size_for_mag(wsi, mag)
    x_tiles = max_tiles(width_mag, tile_size_at_mag, overlap=0)
    y_tiles = max_tiles(height_mag, tile_size_at_mag, overlap=0)
    assert x <= x_tiles, "Tile x-coordinate outside image bounds"
    assert y <= y_tiles, "Tile y-coordinate outside image bounds"
    # Tile is first read from WSI at max magnification, then downscaled to specified mag
    scale = highest_mag(wsi)/mag
    scaled_tile_size = scale * tile_size_at_mag
    # Calculate x and y pixel coordinates on WSI at max magnification
    x_coord = int(round(x * scaled_tile_size))
    y_coord = int(round(y * scaled_tile_size))
    tile = wsi.read_region((x_coord, y_coord), 0, (round(scaled_tile_size), round(scaled_tile_size)))
    return tile.resize((round(tile_size_at_mag), round(tile_size_at_mag)), resample=Image.BICUBIC)


def plot_predictions(img, tile_predictions, tile_size):
    """Plot cancer and non-cancer predictions over Pillow Image

    Red tile border is cancer | Green tile border is non-cancer
    """
    img_overlay = img.copy()
    d = ImageDraw.Draw(img_overlay)
    width = int(np.round(tile_size * 0.07))
    for row, column, prediction in tile_predictions:
        x_top_left, y_top_left = np.round(column * tile_size), np.round(row * tile_size)
        x_bottom_right, y_bottom_right = np.round(x_top_left + tile_size), np.round(y_top_left + tile_size)
        if prediction >= 0.5:
            outline = 'lime'
        elif prediction < 0.5:
            outline = 'red'
        d.rectangle([(x_top_left, y_top_left), (x_bottom_right, y_bottom_right)], outline=outline, width=width)
    return img_overlay


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('-b', '--base_dir', type = Path, required = True,
    #                     help = 'Base Directory')
    parser.add_argument('-s', '--slides_dir', type=Path, default=Path(),
                        help = 'Directory containing H&E and TP53 slides - relative to base directory')
    parser.add_argument('-o', '--out_dir', type=Path, default=Path(),
                        help = 'Output Directory - relative to base directory')
    parser.add_argument('-t', '--template_path', type = Path,
                        help = 'Path to normalisation template slide - relative to base directory')
    parser.add_argument('-nn', '--neural_network_path', type=Path,
                        help='Path to trained neural network - relative to base directory')
    parser.add_argument('-m', '--tile_mag', type = float, default = 10,
                        help = 'Magnification for generating tiles')
    parser.add_argument('-ts', '--tile_size', type = int, default = 224,
                        help = 'Tile size in pixels')
    parser.add_argument('-n', '--normaliser', type = str, default = 'vahadane',
                        choices=['none','reinhard','macenko','vahadane'], help = 'H&E normalisation method')
    parser.add_argument('-std', '--standardise_luminosity', action='store_false',
                        help = 'Disable luminosity standardisation')
    parser.add_argument('-p', '--processing_mag', type = float, default = 5,
                        help = 'Magnification at which to process slide images' )
    parser.add_argument('-v', '--verbosity', action = 'store_true',
                        help = 'Increase output verbosity')

    args = parser.parse_args()

    ####################
    # Paths and Inputs #
    ####################
    # Paths
    # BASE_PATH = args.base_dir
    # SLIDES_PATH = BASE_PATH.joinpath(args.slides_dir)
    # OUTPUT_PATH = BASE_PATH.joinpath(args.out_dir)
    # TEMPLATE_SLIDE_PATH = BASE_PATH.joinpath(args.template_path)
    SLIDES_PATH = args.slides_dir
    OUTPUT_PATH = args.out_dir
    TEMPLATE_SLIDE_PATH = args.template_path
    MODEL_PATH = args.neural_network_path

    # User selectable parameters
    PROCESSING_MAG = args.processing_mag
    TILE_MAG = args.tile_mag
    VERBOSE = args.verbosity
    NORMALISER_METHOD = args.normaliser
    STANDARDISE_LUMINOSITY = args.standardise_luminosity
    NN_TILE_SIZE = args.tile_size

    # Verbose functions
    if VERBOSE:
        verbose_print = lambda *args: print(*args)
        verbose_save_img = lambda img, path, img_type: img.save(path, img_type)
        verbose_save_fig = lambda fig, path, dpi=300: fig.savefig(path, dpi=dpi, bbox_inches='tight')
    else:
        verbose_print = lambda *args: None
        verbose_save_img = lambda *args: None
        verbose_save_fig = lambda *args: None

    # Find Slides
    slide_paths = []
    for slide_path in SLIDES_PATH.glob('**/*.svs'):
        relative_path = slide_path.relative_to(SLIDES_PATH)
        slide_paths.append(relative_path)
    slide_paths.sort()

    slide_info_df = pd.DataFrame({}, columns=['Slide_Name',
                                              'WSI_Area_mm^2',
                                              'Tissue_Area_mm^2',
                                              'Tiles',
                                              'Cancer_Tiles',
                                              'Cancer_Tile_Proportion',
                                              'Average_Sigmoid_Value',
                                              'Weighted_Average_Sigmoid_Value',
                                              'Cancer_Area_Proportion'])

    verbose_print(f'Found {len(slide_paths)} slides')

    # Load template slide
    template_slide = open_slide(str(TEMPLATE_SLIDE_PATH))
    template_img = read_slide_at_mag(template_slide, PROCESSING_MAG).convert('RGB')

    # Fit normaliser to template image
    normaliser = IterativeNormaliser(NORMALISER_METHOD, STANDARDISE_LUMINOSITY)
    normaliser.fit_target(template_img)

    # Load and compile trained neural network
    model = keras.models.load_model(MODEL_PATH)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    verbose_print(model.summary())

    for SLIDE_NUM in range(len(slide_paths)):
    # for SLIDE_NUM in [9]:
        slide_path_absolute = SLIDES_PATH.joinpath(slide_paths[SLIDE_NUM])
        slide_stem = slide_path_absolute.stem
        verbose_print(f'Processing Slide: {slide_stem}')
        SAMPLE_NAME = slide_stem.split('.')[0]
        PREFIX = f'{SAMPLE_NAME}_{PROCESSING_MAG}x'
        slide_info = {'Slide_Name': slide_path_absolute.name}

        # Load Slides
        he_slide = open_slide(str(slide_path_absolute))
        # Handles exceptions where slide max. magnification is not specified
        try:
            he = read_slide_at_mag(he_slide, PROCESSING_MAG).convert('RGB')
        except KeyError:
            print(f'KeyError for slide : {slide_stem}')
            print('Continuing to next slide')
            continue
        verbose_save_img(he, OUTPUT_PATH.joinpath(f'{PREFIX}.jpeg'), 'JPEG')

        # Normalise H&E slide against the template slide
        normaliser.fit_source(he)
        he_norm = normaliser.transform_tile(he)
        verbose_save_img(he_norm, OUTPUT_PATH.joinpath(f'{PREFIX}_normalised.jpeg'), 'JPEG')

        ###############
        # H&E Masking #
        ###############
        he_filtered = filter_green(he_norm)
        he_filtered = filter_grays(he_filtered, tolerance=15)
        he_mask = he_filtered.convert('L')

        he_cv = np.array(he_norm)[:, :, ::-1]  # Convert RGB to BGR

        # Use the filtered H&E image to form the initial mask for Grabcut
        he_mask_initial = (np.array(he_mask) != 255).astype(np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        he_cut = he_cv.copy()
        cv.grabCut(he_cut, he_mask_initial, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
        he_mask_final = np.where((he_mask_initial == 2) | (he_mask_initial == 0), 0, 1).astype('uint8')

        # Remove small debris using a rough 'filled in' mask of the tissue
        kernal_size = round(PROCESSING_MAG * 32)
        kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernal_size, kernal_size))
        he_mask_closed = cv.morphologyEx(he_mask_final, cv.MORPH_CLOSE, kernal)
        he_mask_opened = cv.morphologyEx(he_mask_closed, cv.MORPH_OPEN, kernal)

        he_mask_cleaned = cv.bitwise_and(he_mask_final, he_mask_final, mask=he_mask_opened)
        he_mask_cleaned_pil = Image.fromarray(he_mask_cleaned.astype(np.bool))
        verbose_save_img(he_mask_cleaned_pil, OUTPUT_PATH.joinpath(f'{PREFIX}_tissue_mask.jpeg'), 'JPEG')

        # Calculate Area occupied by each pixel
        # he_slide.associated_images['macro']
        MPP = float(he_slide.properties['aperio.MPP'])  # Microns Per Pixel at max. magnification
        px_area_mm = MPP * MPP * 1e-6
        # Pixel Area (mm^2) for a pixel at processing mag
        equiv_px_area_mm = px_area_mm * (highest_mag(he_slide) / PROCESSING_MAG) ** 2

        # Calculate Whole-Slide-Image Area
        wsi_px_area = he.size[0] * he.size[1]
        wsi_area = wsi_px_area * equiv_px_area_mm
        slide_info['WSI_Area_mm^2'] = wsi_area

        # Calculate Tissue Area
        total_tissue_pixels = np.count_nonzero(he_mask_cleaned.ravel())
        tissue_area = total_tissue_pixels * equiv_px_area_mm
        slide_info['Tissue_Area_mm^2'] = tissue_area
        verbose_print(f'WSI area is {wsi_area :.2f}mm^2 of which tissue occupies {tissue_area :.2f}mm^2')

        #########################
        # Tiling and Prediction #
        #########################

        # Find tile pixel size for processing mag image that is equivalent to the tile mag image
        tile_size = NN_TILE_SIZE * PROCESSING_MAG / TILE_MAG

        # Calculate the proportion of each tile occupied by tissue
        tgen = tile_gen(he_mask_cleaned_pil, tile_size)
        width, height = next(tgen)
        counts = []
        proportions = []
        for tile in tgen:
            tile_np = np.array(tile)
            tile_count = np.count_nonzero(tile_np)
            total_pixels = tile_np.size
            counts.append(tile_count)
            proportions.append(tile_count / total_pixels)
        hist = np.reshape(np.array(counts), (height, width))
        hist_proportion = np.reshape(np.array(proportions), (height, width))

        # Crop normalised H&E image to tiled area for heatmap overlay
        width_mag, height_mag = get_size_for_mag(he_slide, TILE_MAG)
        width_tiled = round((width_mag - width_mag % NN_TILE_SIZE) * PROCESSING_MAG / TILE_MAG)
        height_tiled = round((height_mag - height_mag % NN_TILE_SIZE) * PROCESSING_MAG / TILE_MAG)
        he_norm_cropped = he_norm.crop((0, 0, width_tiled, height_tiled))

        # Generate a semi-transparent colourmap for plotting heatmaps over images
        cmap = plt.cm.jet
        my_cmap = cmap(np.arange(cmap.N))  # Get the colormap colours
        my_cmap[:, -1] = np.linspace(0.2, 0.7, cmap.N)  # Set alpha as a linear gradient
        my_cmap = ListedColormap(my_cmap)  # Create new colourmap

        # Overlay translucent heatmap over H&E tissue mask
        tissue_proportion_heatmap = plt.figure(figsize=(20, 10))
        hmax = sns.heatmap(hist_proportion, vmin=0, vmax=1, square=True, xticklabels=False, yticklabels=False,
                           cmap=my_cmap, mask=(hist_proportion == 0),
                           cbar_kws={'pad': 0.01, 'shrink': 0.9})
        cbar_axes = hmax.figure.axes[-1]
        # cbar_axes.yaxis.label.set_size(20)
        cbar_axes.tick_params(labelsize=20)
        hmax.imshow(np.array(he_norm_cropped),
                    cmap='gray',
                    aspect=hmax.get_aspect(),
                    extent=hmax.get_xlim() + hmax.get_ylim(),
                    interpolation='bicubic',
                    zorder=0)
        hmax.set_title(f'Proportion of Tile Area occupied by Tissue for {SAMPLE_NAME}', fontsize=20)
        tissue_proportion_heatmap.tight_layout()
        verbose_save_fig(tissue_proportion_heatmap,
                         OUTPUT_PATH.joinpath(f'{SAMPLE_NAME}_tissue_proportion_heatmap.jpeg'))

        tissue_tile_coords = np.argwhere(hist_proportion > 0).astype(np.uint16)

        # Get a list of tissue tile images
        tissue_tile_imgs = []
        for y, x in tissue_tile_coords:
            tissue_tile_imgs.append(tile_from_slide(x, y, he_slide, TILE_MAG, NN_TILE_SIZE).convert('RGB'))

        tissue_tile_imgs_norm = [x for x in map(normaliser.transform_tile, tissue_tile_imgs)]

        tissue_tiles_number = len(tissue_tile_imgs_norm)
        slide_info['Tiles'] = tissue_tiles_number
        verbose_print(f'Found {tissue_tiles_number} tissue tiles')

        predictions = []
        for i in range(len(tissue_tile_imgs_norm)):
            predictions.append(model.predict(np.expand_dims(np.array(tissue_tile_imgs_norm[i]) / 255, axis=0))[0][0])

        tile_predictions = [(x, y, z) for (x, y), z in zip(tissue_tile_coords, predictions)]

        # Overlay binary predictions onto the H&E image
        predictions_overlay = plot_predictions(he_norm, tile_predictions, tile_size)
        verbose_save_img(predictions_overlay, OUTPUT_PATH.joinpath(f'{PREFIX}_predictions.jpeg'), 'JPEG')
        thumbnail(predictions_overlay)

        cancer_tiles = np.count_nonzero(np.array(predictions) < 0.5)
        cancer_tile_proportion = cancer_tiles / len(predictions)
        slide_info['Cancer_Tiles'] = cancer_tiles
        slide_info['Cancer_Tile_Proportion'] = cancer_tile_proportion
        verbose_print(f'{cancer_tile_proportion * 100 :.2f}% of tissue tiles were predicted as cancer')

        # Create an array of tile predictions such that cancer = 1 and non-cancer = 0
        tile_predictions_array = np.zeros(hist_proportion.shape)
        for row, column, prediction in tile_predictions:
            tile_predictions_array[row, column] = 1 - prediction

        average_sigmoid_value = np.mean(1 - np.array(predictions))
        slide_info['Average_Sigmoid_Value'] = average_sigmoid_value

        hist_proportion_norm = hist_proportion / hist_proportion.sum()
        weighted_average_sigmoid_value = np.sum(tile_predictions_array * hist_proportion_norm)
        slide_info['Weighted_Average_Sigmoid_Value'] = weighted_average_sigmoid_value

        # Overlay raw predictions heatmap over H&E tissue mask
        predictions_heatmap = plt.figure(figsize=(20, 10))
        hmax = sns.heatmap(tile_predictions_array, vmin=0, vmax=1, square=True, xticklabels=False, yticklabels=False,
                           cmap=my_cmap, mask=(hist_proportion == 0),
                           cbar_kws={'pad': 0.01})
        cbar_axes = hmax.figure.axes[-1]
        cbar_axes.tick_params(labelsize=20)
        hmax.imshow(np.array(he_norm_cropped),
                    cmap='gray',
                    aspect=hmax.get_aspect(),
                    extent=hmax.get_xlim() + hmax.get_ylim(),
                    interpolation='bicubic',
                    zorder=0)
        hmax.set_title(f'HEMnet Predictions for {SAMPLE_NAME}', fontsize=20)
        predictions_heatmap.tight_layout()
        verbose_save_fig(predictions_heatmap, OUTPUT_PATH.joinpath(f'{SAMPLE_NAME}_predictions_heatmap.jpeg'))

        cancer_tile_array = tile_predictions_array >= 0.5
        cancer_tissue_area_array = cancer_tile_array * hist_proportion
        cancer_area_proportion = np.sum(cancer_tissue_area_array) / np.sum(hist_proportion)
        slide_info['Cancer_Area_Proportion'] = cancer_area_proportion
        verbose_print(f'{cancer_area_proportion * 100 :.2f}% of tissue area was predicted as cancer')

        slide_info_df = slide_info_df.append(pd.Series(slide_info, name=SAMPLE_NAME))

        slide_info_df.to_csv(OUTPUT_PATH.joinpath('Slide_Predictions.csv'))

        # Close all open figures to prevent excessive memory usage
        plt.close('all')

    print('Inference Finished!')


