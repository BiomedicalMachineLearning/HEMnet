# ------------------------------------------------------------------------
# Generate a train dataset from whole slide images
#
# Place TP53 and H&E slides in a single directory
# TP53 slides must be named : SlideName_Pathology_TP53
#     e.g. patient123_N_TP53.svs for a normal slide or patient123_T_TP53.svs for a cancer slide
# H&E slides must be named : slide_Pathology_HandE
#     e.g. patient123_N_HandE.svs
#
# Example command:
# python HEMnet_train_dataset.py -b '/gpfs1/scratch/90days/s4436005' -s 'Slides/Training' -o 'img_reg/27_05_20_testing' -t 'Slides/1957_T_9668_3_HandE.svs' -v
# python HEMnet_train_dataset.py -b '/gpfs1/scratch/90days/s4436005' -s 'Slides/Training' -o 'img_reg/train_dataset_10x_27_05_20_strict_vahadane2' -t 'Slides/1957_T_9668_3_HandE.svs' -v
# ------------------------------------------------------------------------

import argparse
import importlib
import io
import matplotlib.pyplot as plt
import numpy as np
import os
from openslide import open_slide
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageChops, ImageDraw
import SimpleITK as sitk
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes
import staintools
from staintools import stain_normalizer, LuminosityStandardizer
from staintools.preprocessing.input_validation import is_uint8_image
import sys

#Allow Pillow to open very big images
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
    #Restrict argument to float between 0 and 1 (inclusive)
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError('{0} not a floating point literal'.format(x))
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError('{0} not in range [0.0, 1.0]'.format(x))
    return x

def dab_tile_array(img, tile_size):
    """Creates array with mean DAB intensity value for each tile

    Parameters
    ----------
    img : Pillow image (RGB)
    tile_size : int

    Returns
    -------
    dab_tile_array : ndarray
    """
    dab_values = []
    tgen = tile_gen(img, tile_size)
    shape = next(tgen)
    for tile in tgen:
        tile = tile.convert('RGB')
        tile_hed = rgb2hed(tile)
        tile_dab = -tile_hed[:,:,2]
        dab_values.append(tile_dab.mean())
    return np.reshape(dab_values, shape)

def uncertain_mask(img, tile_size, cancer_thresh, non_cancer_thresh):
    """Create mask of uncertain tiles

    Parameters
    ----------
    img : Pillow image (RGB)
    tile_size : int
    cancer_thresh : float
        DAB intensity threshold between 0 and 1.0 (inclusive)
        Below threshold is cancer
    non_cancer_thresh : float
        DAB intensity threshold between 0 and 1.0 (inclusive)
        Above threshold is non-cancer

    Returns
    -------
    uncertain_mask : ndarray
    """
    dab_array = dab_tile_array(img, tile_size)
    binary_mask = ((cancer_thresh < dab_array) & (dab_array < non_cancer_thresh))
    return (np.invert(binary_mask)).astype(np.uint8)


def save_train_tiles(path, tile_gen, cancer_mask, tissue_mask, uncertain_mask, prefix = ''):
    """Save tiles for train dataset

    Parameters
    ----------
    path : Pathlib Path
    tile_gen : tile_gen
    cancer_mask : ndarray
    tissue_mask : ndarray
    uncertain_mask : ndarray
    prefix : str (optional)

    Returns
    -------
    None
    """
    os.makedirs(path.joinpath('cancer'), exist_ok = True)
    os.makedirs(path.joinpath('non-cancer'), exist_ok = True)
    os.makedirs(path.joinpath('uncertain'), exist_ok = True)
    x_tiles, y_tiles = next(tile_gen)
    verbose_print('Whole Image Size is {0} x {1}'.format(x_tiles, y_tiles))
    i = 0
    cancer = 0
    uncertain = 0
    non_cancer = 0
    for tile in tile_gen:
        img = tile.convert('RGB')
        ###
        img_norm = normaliser.transform_tile(img)
        ###
        # Name tile as horizontal position _ vertical position starting at (0,0)
        tile_name = prefix + str(np.floor_divide(i,x_tiles)) + '_' +  str(i%x_tiles)
        if uncertain_mask.ravel()[i] == 0:
            img_norm.save(path.joinpath('uncertain', tile_name + '.jpeg'), 'JPEG')
            uncertain += 1
        elif cancer_mask.ravel()[i] == 0:
            img_norm.save(path.joinpath('cancer', tile_name + '.jpeg'), 'JPEG')
            cancer += 1
        elif tissue_mask.ravel()[i] == 0:
            img_norm.save(path.joinpath('non-cancer', tile_name + '.jpeg'), 'JPEG')
            non_cancer += 1
        i += 1
    verbose_print('Cancer tiles: {0}, Non Cancer tiles: {1}, Uncertain tiles: {2}'.format(cancer, non_cancer, uncertain))
    verbose_print('Exported tiles for {0}'.format(prefix))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type = Path, required = True,
                        help = 'Base Directory')
    parser.add_argument('-s', '--slides_dir', type = Path, default = Path(),
                        help = 'Directory containing H&E and TP53 slides - relative to base directory')
    parser.add_argument('-o', '--out_dir', type = Path, default = Path(),
                        help = 'Output Directory - relative to base directory')
    parser.add_argument('-t', '--template_path', type = Path,
                        help = 'Path to normalisation template slide - relative to base directory')
    parser.add_argument('-m', '--tile_mag', type = float, default = 10,
                        help = 'Magnification for generating tiles')
    parser.add_argument('-ts', '--tile_size', type = int, default = 224,
                        help = 'Output tile size in pixels')
    parser.add_argument('-n', '--normaliser', type = str, default = 'vahadane',
                        choices=['none','reinhard','macenko','vahadane'], help = 'H&E normalisation method')
    parser.add_argument('-std', '--standardise_luminosity', action='store_false',
                        help = 'Disable luminosity standardisation')
    parser.add_argument('-a', '--align_mag', type = float, default = 2,
                        help = 'Magnification for aligning H&E and TP53 slide' )
    parser.add_argument('-c', '--cancer_thresh', type = restricted_float, default = 0.39,
                        help = 'TP53 threshold for cancer classification')
    parser.add_argument('-nc', '--non_cancer_thresh', type = restricted_float, default = 0.40,
                        help = 'TP53 threshold for non-cancer classification')
    # parser.add_argument('-f', '--fix_orientation', action = 'store_true',
    #                     help = 'Automatically fix the slide orientation')
    parser.add_argument('-v', '--verbosity', action = 'store_true',
                        help = 'Increase output verbosity')

    args = parser.parse_args()

    ####################
    # Paths and Inputs #
    ####################
    #Paths
    BASE_PATH = args.base_dir
    SLIDES_PATH = BASE_PATH.joinpath(args.slides_dir)
    OUTPUT_PATH = BASE_PATH.joinpath(args.out_dir)
    TEMPLATE_SLIDE_PATH = BASE_PATH.joinpath(args.template_path)

    #User selectable parameters
    ALIGNMENT_MAG = args.align_mag
    TILE_MAG = args.tile_mag
    CANCER_THRESH = args.cancer_thresh
    NON_CANCER_THRESH = args.non_cancer_thresh
    # FIX_ORIENTATION = args.fix_orientation
    VERBOSE = args.verbosity
    NORMALISER_METHOD = args.normaliser
    STANDARDISE_LUMINOSITY = args.standardise_luminosity
    OUTPUT_TILE_SIZE = args.tile_size

    # Verbose functions
    if VERBOSE:
        verbose_print = lambda *args: print(*args)
        verbose_save_img = lambda img, path, img_type: img.save(path, img_type)
        verbose_save_fig = lambda fig, path, dpi=300: fig.savefig(path, dpi=dpi)
    else:
        verbose_print = lambda *args: None
        verbose_save_img = lambda *args: None
        verbose_save_fig = lambda *args: None

    # Find Slides
    slides = []
    for slide in SLIDES_PATH.glob('*.svs'):
        name = slide.name
        slides.append(name)
    slides.sort()

    TP53_slides = [slide for slide in slides if 'TP53' in slide]
    HE_slides = [slide for slide in slides if 'HandE' in slide]
    Paired_slides = list(zip(TP53_slides, HE_slides))
    print('Found {0} pairs of slides'.format(len(Paired_slides)))
    for i, pair in enumerate(Paired_slides):
        tp53, he = pair
        verbose_print('{0}. {1}|{2}'.format(i + 1, tp53, he))

    # Create a dataframe to store the image registration performance metric for each slide
    performance_df = pd.DataFrame((Paired_slides), columns=["TP53_Slide_Name", "H&E_Slide_Name"])
    performance_df["Template_Slide_Name"] = TEMPLATE_SLIDE_PATH.name

    template_slide = open_slide(str(TEMPLATE_SLIDE_PATH))
    template_img = read_slide_at_mag(template_slide, ALIGNMENT_MAG).convert('RGB')

    normaliser = IterativeNormaliser(NORMALISER_METHOD, STANDARDISE_LUMINOSITY)
    normaliser.fit_target(template_img)

    # Process each pair of slides
    for num in range(len(Paired_slides)):
    #for num in [8]:
        SLIDE_NUM = num
        PREFIX = Paired_slides[SLIDE_NUM][0][:-8]
        print('-' * (18 + len(PREFIX)))
        print('Processing Slide: {0}'.format(PREFIX))
        tp53_name, he_name = Paired_slides[SLIDE_NUM]
        tp53_slide = open_slide(str(SLIDES_PATH.joinpath(tp53_name)))
        he_slide = open_slide(str(SLIDES_PATH.joinpath(he_name)))

        # Load Slides
        he = read_slide_at_mag(he_slide, ALIGNMENT_MAG)
        tp53 = read_slide_at_mag(tp53_slide, ALIGNMENT_MAG)

        # Normalise H&E Slide
        normaliser.fit_source(he)
        he_norm = normaliser.transform_tile(he)

        verbose_save_img(he_norm.convert('RGB'),
                         OUTPUT_PATH.joinpath(PREFIX + str(ALIGNMENT_MAG) + 'x_normalised.jpeg'), 'JPEG')

        ######################
        # Image Registration #
        ######################

        INTERPOLATOR = sitk.sitkLanczosWindowedSinc

        # Convert to grayscale
        tp53_gray = tp53.convert('L')
        he_gray = he_norm.convert('L')
        # Convert to ITK format
        tp53_itk = get_itk_from_pil(tp53_gray)
        he_itk = get_itk_from_pil(he_gray)
        # Set fixed and moving images
        fixed_img = he_itk
        moving_img = tp53_itk

        # Check initial registration
        # Centre the two images, then compare their alignment
        initial_transform = sitk.CenteredTransformInitializer(fixed_img, moving_img, sitk.Euler2DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
        moving_rgb = sitk_transform_rgb(tp53, he_norm, initial_transform)

        # Visualise and save alignment
        align_plotter = PlotImageAlignment('vertical', 300)
        comparison_pre_v_stripes = align_plotter.plot_images(he, moving_rgb)
        verbose_save_img(comparison_pre_v_stripes.convert('RGB'),
                         OUTPUT_PATH.joinpath(PREFIX + 'comparison_pre_align_v_stripes.jpeg'), 'JPEG')

        align_plotter = PlotImageAlignment('horizontal', 300)
        comparison_pre_h_stripes = align_plotter.plot_images(he, moving_rgb)
        verbose_save_img(comparison_pre_h_stripes.convert('RGB'),
                         OUTPUT_PATH.joinpath(PREFIX + 'comparison_pre_align_h_stripes.jpeg'), 'JPEG')

        align_plotter = PlotImageAlignment('mosaic', 300)
        comparison_pre_mosaic = align_plotter.plot_images(he, moving_rgb)
        verbose_save_img(comparison_pre_mosaic.convert('RGB'),
                         OUTPUT_PATH.joinpath(PREFIX + 'comparison_pre_align_mosaic.jpeg'), 'JPEG')

        comparison_pre_colour_overlay = show_alignment(he_norm, moving_rgb, prefilter=True)
        verbose_save_img(comparison_pre_colour_overlay.convert('RGB'),
                         OUTPUT_PATH.joinpath(PREFIX + 'comparison_pre_align_colour_overlay.jpeg'), 'JPEG')

        # Compute the mutual information between the two images before registration
        moving_resampled_initial = sitk.Resample(moving_img, fixed_img, initial_transform,
                                                 INTERPOLATOR, 0.0, moving_img.GetPixelID())
        initial_mutual_info = calculate_mutual_info(np.array(he_gray),
                                                    np.array(get_pil_from_itk(moving_resampled_initial)))
        verbose_print('Initial mutual information metric: {0}'.format(initial_mutual_info))
        performance_df.loc[SLIDE_NUM, "Initial_Mutual_Info"] = initial_mutual_info

        # --- Affine Registration --- #

        initial_transform = sitk.CenteredTransformInitializer(fixed_img, moving_img, sitk.Euler2DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
        affine_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings.
        affine_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        affine_method.SetMetricSamplingStrategy(affine_method.RANDOM)
        affine_method.SetMetricSamplingPercentage(0.15)

        affine_method.SetInterpolator(INTERPOLATOR)

        # Optimizer settings.
        affine_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=100,
                                                    convergenceMinimumValue=1e-6, convergenceWindowSize=20)
        affine_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
        affine_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4])
        affine_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2])
        affine_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place, we would possibly like to run this cell multiple times.
        affine_method.SetInitialTransform(initial_transform, inPlace=False)

        # Connect all of the observers so that we can perform plotting during registration.
        affine_method.AddCommand(sitk.sitkStartEvent, start_plot)
        affine_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
        affine_method.AddCommand(sitk.sitkIterationEvent, lambda: update_plot(affine_method))

        affine_transform = affine_method.Execute(sitk.Cast(fixed_img, sitk.sitkFloat32),
                                                 sitk.Cast(moving_img, sitk.sitkFloat32))

        affine_fig = plot_metric('Plot of mutual information cost in affine registration')
        plt.show()
        verbose_save_fig(affine_fig, OUTPUT_PATH.joinpath(PREFIX + 'affine_metric_plot.jpeg'))
        end_plot()

        verbose_print(
            'Affine Optimizer\'s stopping condition, {0}'.format(
                affine_method.GetOptimizerStopConditionDescription()))

        # Compute the mutual information between the two images after affine registration
        moving_resampled_affine = sitk.Resample(moving_img, fixed_img, affine_transform,
                                                INTERPOLATOR, 0.0, moving_img.GetPixelID())
        affine_mutual_info = calculate_mutual_info(np.array(he_gray),
                                                   np.array(get_pil_from_itk(moving_resampled_affine)))
        verbose_print('Affine mutual information metric: {0}'.format(affine_mutual_info))
        performance_df.loc[SLIDE_NUM, "Affine_Mutual_Info"] = affine_mutual_info

        # --- B-spline registration --- #

        bspline_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings.
        bspline_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        bspline_method.SetMetricSamplingStrategy(bspline_method.RANDOM)
        bspline_method.SetMetricSamplingPercentage(0.15)

        bspline_method.SetInterpolator(INTERPOLATOR)

        # Optimizer settings.
        bspline_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=200,
                                                     convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        bspline_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
        bspline_method.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
        bspline_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
        bspline_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place, we would possibly like to run this cell multiple times.
        transformDomainMeshSize = [8] * moving_resampled_affine.GetDimension()
        initial_transform = sitk.BSplineTransformInitializer(fixed_img, transformDomainMeshSize)
        bspline_method.SetInitialTransform(initial_transform, inPlace=False)

        # Connect all of the observers so that we can perform plotting during registration.
        bspline_method.AddCommand(sitk.sitkStartEvent, start_plot)
        bspline_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
        bspline_method.AddCommand(sitk.sitkIterationEvent, lambda: update_plot(bspline_method))

        bspline_transform = bspline_method.Execute(sitk.Cast(fixed_img, sitk.sitkFloat32),
                                                   sitk.Cast(moving_resampled_affine, sitk.sitkFloat32))

        bspline_fig = plot_metric('Plot of mutual information cost in B-spline registration')
        plt.show()
        verbose_save_fig(bspline_fig, OUTPUT_PATH.joinpath(PREFIX + 'bspline_metric_plot.jpeg'))
        end_plot()

        verbose_print('B-spline Optimizer\'s stopping condition, {0}'.format(
            bspline_method.GetOptimizerStopConditionDescription()))

        # Compute the mutual information between the two images after B-spline registration
        moving_resampled_final = sitk.Resample(moving_resampled_affine, fixed_img, bspline_transform,
                                               INTERPOLATOR, 0.0, moving_img.GetPixelID())
        bspline_mutual_info = calculate_mutual_info(np.array(he_gray),
                                                    np.array(get_pil_from_itk(moving_resampled_final)))
        verbose_print('B-spline mutual information metric: {0}'.format(bspline_mutual_info))
        performance_df.loc[SLIDE_NUM, "Final_Mutual_Info"] = bspline_mutual_info

        # Transform the original TP53 into the aligned TP53 image
        moving_rgb_affine = sitk_transform_rgb(tp53, he_norm, affine_transform, INTERPOLATOR)
        tp53_aligned = sitk_transform_rgb(moving_rgb_affine, he_norm, bspline_transform, INTERPOLATOR)

        # Visualise and save alignment
        align_plotter = PlotImageAlignment('vertical', 300)
        comparison_post_v_stripes = align_plotter.plot_images(he, tp53_aligned)
        verbose_save_img(comparison_post_v_stripes.convert('RGB'),
                         OUTPUT_PATH.joinpath(PREFIX + 'comparison_post_align_v_stripes.jpeg'), 'JPEG')

        align_plotter = PlotImageAlignment('horizontal', 300)
        comparison_post_h_stripes = align_plotter.plot_images(he, tp53_aligned)
        verbose_save_img(comparison_post_h_stripes.convert('RGB'),
                         OUTPUT_PATH.joinpath(PREFIX + 'comparison_post_align_h_stripes.jpeg'), 'JPEG')

        align_plotter = PlotImageAlignment('mosaic', 300)
        comparison_post_mosaic = align_plotter.plot_images(he, tp53_aligned)
        verbose_save_img(comparison_post_mosaic.convert('RGB'),
                         OUTPUT_PATH.joinpath(PREFIX + 'comparison_post_align_mosaic.jpeg'), 'JPEG')

        # Remove backgrounds from TP53 and H&E images
        tp53_filtered = filter_green(tp53_aligned)
        he_filtered = filter_green(he_norm)
        tp53_filtered = filter_grays(tp53_filtered, tolerance=2)
        he_filtered = filter_grays(he_filtered, tolerance=15)

        # Visually compare alignment between the registered TP53 and original H&E image
        comparison_post_colour_overlay = show_alignment(he_filtered, tp53_filtered)
        verbose_save_img(comparison_post_colour_overlay.convert('RGB'),
                         OUTPUT_PATH.joinpath(PREFIX + 'comparison_post_align_colour_overlay.jpeg'), 'JPEG')

        verbose_save_img(tp53_aligned.convert('RGB'),
                         OUTPUT_PATH.joinpath(PREFIX + str(ALIGNMENT_MAG) + 'x_TP53_aligned.jpeg'), 'JPEG')
        verbose_save_img(tp53_filtered.convert('RGB'),
                         OUTPUT_PATH.joinpath(PREFIX + str(ALIGNMENT_MAG) + 'x_TP53_aligned_white.jpeg'), 'JPEG')

        ####################################
        # Generate cancer and tissue masks #
        ####################################

        # Scale tile size for alignment mag
        tile_size = OUTPUT_TILE_SIZE * ALIGNMENT_MAG / TILE_MAG

        # Generate cancer mask and tissue mask from filtered tp53 image
        c_mask = cancer_mask(tp53_filtered, tile_size, 250).astype(np.bool)
        t_mask_tp53 = tissue_mask(tp53_filtered, tile_size)
        t_mask_he = tissue_mask(he_filtered, tile_size)

        # Generate tissue mask with tissue common to both the TP53 and H&E image
        t_mask = np.logical_not(np.logical_not(t_mask_tp53) & np.logical_not(t_mask_he))

        # Generate uncertain mask
        u_mask = uncertain_mask(tp53_filtered, tile_size, CANCER_THRESH, NON_CANCER_THRESH)
        u_mask_filtered = np.logical_not(np.logical_not(u_mask) & np.logical_not(t_mask))

        # Filter tissue mask such that any uncertain tiles are removed
        t_mask_filtered = np.zeros(t_mask.shape)
        for x in range(t_mask.shape[0]):
            for y in range(t_mask.shape[1]):
                if t_mask[x, y] == 0 and u_mask[x, y] == 1:
                    t_mask_filtered[x, y] = False
                else:
                    t_mask_filtered[x, y] = True

        non_c_mask = np.invert(c_mask)
        non_c_mask = np.logical_not(np.logical_and(np.logical_not(non_c_mask), np.logical_not(t_mask_filtered)))

        # If Slide is normal
        if '_N_' in PREFIX:
            # if True:
            # Merge cancer mask with uncertain mask
            # This marks all tiles that are uncertain or cancer as uncertain
            u_mask_filtered = np.logical_not(np.logical_or(np.logical_not(u_mask_filtered), np.logical_not(c_mask)))
            # Blank out cancer mask so no cancer tiles exist
            c_mask_filtered = np.ones(c_mask.shape, dtype=bool)
            # Non cancer tiles are tiles that are in the tissue and not cancer
            non_c_mask_filtered = np.logical_not(
                np.logical_and(np.logical_not(non_c_mask), np.logical_not(t_mask_filtered)))
            verbose_print('Normal Slide Identified')

        # If Slide is cancerous
        if 'T' in PREFIX:
            # if False:
            # Merge non-cancer mask with uncertain mask
            u_mask_filtered = np.logical_not(np.logical_or(np.logical_not(non_c_mask), np.logical_not(u_mask_filtered)))
            # Blank out non cancer mask
            non_c_mask_filtered = np.ones(non_c_mask.shape, dtype=bool)
            # Cancer tile are tiles that are in the tissue and not cancer
            # Make sure all cancer tiles exist in the tissue mask
            c_mask_filtered = np.logical_not(np.logical_not(c_mask) & np.logical_not(t_mask_filtered))
            verbose_print('Cancer Slide Identified')

            # Overlay masks onto TP53 and H&E Image
            overlay_tp53 = plot_masks(tp53_filtered, c_mask_filtered, t_mask_filtered, tile_size, u_mask_filtered)
            verbose_save_img(overlay_tp53.convert('RGB'), OUTPUT_PATH.joinpath(PREFIX + 'TP53_overlay.jpeg'), 'JPEG')

            overlay_he = plot_masks(he_filtered, c_mask_filtered, t_mask_filtered, tile_size, u_mask_filtered)
            verbose_save_img(overlay_he.convert('RGB'), OUTPUT_PATH.joinpath(PREFIX + 'HE_overlay.jpeg'), 'JPEG')

        ##############
        # Save Tiles #
        ##############

        # Make Directory to save tiles
        TILES_PATH = OUTPUT_PATH.joinpath('tiles_' + str(TILE_MAG) + 'x')
        os.makedirs(TILES_PATH, exist_ok=True)

        # Save tiles
        tgen = tile_gen_at_mag(he_slide, TILE_MAG, OUTPUT_TILE_SIZE)
        save_train_tiles(TILES_PATH, tgen, c_mask_filtered, t_mask_filtered, u_mask_filtered, prefix=PREFIX)

        non_cancer_tiles = np.invert(non_c_mask_filtered).sum()

        uncertain_tiles = np.invert(u_mask_filtered).sum()

        cancer_tiles = np.invert(c_mask_filtered).sum()

        performance_df.loc[SLIDE_NUM, "Cancer_Tiles"] = cancer_tiles
        performance_df.loc[SLIDE_NUM, "Uncertain_Tiles"] = uncertain_tiles
        performance_df.loc[SLIDE_NUM, "Non_Cancer_Tiles"] = non_cancer_tiles

    performance_df.to_csv(OUTPUT_PATH.joinpath('performance_metrics.csv'))



