import importlib
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import openslide
from openslide import open_slide
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageChops, ImageDraw
import SimpleITK as sitk
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes
import sys
import staintools
from staintools import stain_normalizer, LuminosityStandardizer

Image.MAX_IMAGE_PIXELS = None

# Import package 

print(sys.executable) 

BASE_DIR = Path().resolve()
HEMNET_DIR = BASE_DIR.parent.joinpath('HEMnet')

#os.chdir(HEMNET_DIR)

sys.path.append(str(HEMNET_DIR))

print(os.getcwd())

from slide import *
from utils import *

# User Input parameters
#SLIDES_PATH = Path('/gpfs1/scratch/90days/s4436005/Slides/gold_standards')
SLIDES_PATH = Path('/gpfs1/scratch/90days/s4436005')
#OUTPUT_PATH = Path('/gpfs1/scratch/90days/s4436005/img_reg/gold_standards/1957_T')
OUTPUT_PATH = Path('/gpfs1/scratch/90days/s4436005/img_reg/train_dataset_10x_13_12_19_test')
TEMPLATE_SLIDE_PATH = Path('/gpfs1/scratch/90days/s4436005/2171_T_11524A_2_HandE.svs')

ALIGNMENT_MAG = 2
TILE_MAG = 10
VERBOSE = True
CANCER_THRESH = 0.39
NON_CANCER_THRESH = 0.40

#Verbose functions
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
verbose_print(slides)

TP53_slides = [slide for slide in slides if 'TP53' in slide]
HE_slides = [slide for slide in slides if 'HandE' in slide]
Paired_slides = list(zip(TP53_slides, HE_slides))
verbose_print(Paired_slides)

def thumbnail(img, size = (1000,1000)):
    img_thumbnail = img.copy()
    img_thumbnail.thumbnail(size)
    return img_thumbnail

#Load and fit template slide
template_slide = open_slide(str(TEMPLATE_SLIDE_PATH))
template = read_slide_at_mag(template_slide, ALIGNMENT_MAG)
normaliser = staintools.StainNormalizer(method='vahadane')
template_std = LuminosityStandardizer.standardize(np.array(template))
normaliser.fit(template_std)
#thumbnail(Image.fromarray(template_std))

for num in range(len(Paired_slides)):
    SLIDE_NUM = num
    PREFIX = Paired_slides[SLIDE_NUM][0][:-8]
    verbose_print('Processing Slide: {0}'.format(PREFIX))
    tp53_name, he_name = Paired_slides[SLIDE_NUM]
    tp53_slide = open_slide(str(SLIDES_PATH.joinpath(tp53_name)))
    he_slide = open_slide(str(SLIDES_PATH.joinpath(he_name)))
    #Load Slides
    he = read_slide_at_mag(he_slide, ALIGNMENT_MAG)
    tp53 = read_slide_at_mag(tp53_slide, ALIGNMENT_MAG)
    #Normalise H&E Slide
    he_std = LuminosityStandardizer.standardize(np.array(he))
    he_transformed = normaliser.transform(he_std)
    he_norm_small = Image.fromarray(he_transformed)
    ######################
    # Image Registration #
    ######################

    INTERPOLATOR = sitk.sitkLanczosWindowedSinc

    #Convert to grayscale
    tp53_gray = tp53.convert('L')
    he_gray = he_norm_small.convert('L')
    #Convert to ITK format
    tp53_itk = get_itk_from_pil(tp53_gray)
    he_itk = get_itk_from_pil(he_gray)
    #Set fixed and moving images
    fixed_img = he_itk
    moving_img = tp53_itk

    # Check initial registration
    # Centre the two images, then compare their alignment
    initial_transform = sitk.CenteredTransformInitializer(fixed_img, moving_img, sitk.Euler2DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    moving_rgb = sitk_transform_rgb(tp53, he_norm_small, initial_transform)
    comparison_pre = show_alignment(he_norm_small, moving_rgb, prefilter = True)
    verbose_save_img(comparison_pre.convert('RGB'), OUTPUT_PATH.joinpath(PREFIX + 'comparison_pre_registration.jpeg'), 'JPEG' )
    ################################
    # Mutual Information Functions #
    ################################

    def mutual_information(hgram):
        """Mutual information for joint histogram
        """
        # Convert bins counts to probability values
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis = 1) # marginal for x over y
        py = np.sum(pxy, axis = 0) # marginal for y over x
        px_py = px[:, None] * py[None, :] #Broadcat to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays 
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    def mutual_info_histogram(fixed_img, moving_img, bins = 20, log = False):
        hist_2d, x_edges, y_edges = np.histogram2d(fixed_img.ravel(), moving_img.ravel(), bins = bins)
        if log:
            hist_2d_log = np.zeros(hist_2d.shape)
            non_zeros = hist_2d != 0
            hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
            return hist_2d_log
        return hist_2d

    def plot_mutual_info_histogram(histogram):
        plt.imshow(histogram.T, origin = 'lower')
        plt.xlabel('Fixed Image')
        plt.ylabel('Moving Image')

    def calculate_mutual_info(fixed_img, moving_img):
        hist = mutual_info_histogram(fixed_img, moving_img)
        return mutual_information(hist)
    
    # Compute the mutual information between the original (unregistered) images
    moving_resampled = sitk.Resample(moving_img, fixed_img, initial_transform, 
                                     INTERPOLATOR, 0.0, moving_img.GetPixelID())
    original_mutual_info = calculate_mutual_info(np.array(he_gray), np.array(get_pil_from_itk(moving_resampled)))
    verbose_print('Pre-registration mutual information metric: {0}'.format(original_mutual_info))
    #Affine Registration
    initial_transform = sitk.CenteredTransformInitializer(fixed_img, moving_img, sitk.Euler2DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    affine_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    affine_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    affine_method.SetMetricSamplingStrategy(affine_method.RANDOM)
    affine_method.SetMetricSamplingPercentage(0.15)

    affine_method.SetInterpolator(INTERPOLATOR)

    # Optimizer settings.
    affine_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=20)
    affine_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    affine_method.SetShrinkFactorsPerLevel(shrinkFactors = [8,4])
    affine_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4,2])
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

    verbose_print('Affine metric value: {0}'.format(affine_method.GetMetricValue()))
    verbose_print('Affine Optimizer\'s stopping condition, {0}'.format(affine_method.GetOptimizerStopConditionDescription()))
    #Compute the mutual information between the two images after affine registration
    moving_resampled_affine = sitk.Resample(moving_img, fixed_img, affine_transform, 
                                     INTERPOLATOR, 0.0, moving_img.GetPixelID())
    affine_mutual_info = calculate_mutual_info(np.array(he_gray), np.array(get_pil_from_itk(moving_resampled_affine)))
    verbose_print('Affine mutual information metric: {0}'.format(affine_mutual_info))
    #B-spline registration
    bspline_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    bspline_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    bspline_method.SetMetricSamplingStrategy(bspline_method.RANDOM)
    bspline_method.SetMetricSamplingPercentage(0.15)

    bspline_method.SetInterpolator(INTERPOLATOR)

    # Optimizer settings.
    bspline_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    bspline_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    bspline_method.SetShrinkFactorsPerLevel(shrinkFactors = [2,1])
    bspline_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1,0])
    bspline_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    transformDomainMeshSize = [8]*moving_resampled_affine.GetDimension()
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

    verbose_print('B-spline metric value: {0}'.format(bspline_method.GetMetricValue()))
    verbose_print('B-spline Optimizer\'s stopping condition, {0}'.format(bspline_method.GetOptimizerStopConditionDescription()))
    #Compute the mutual information between the two images after B-spline registration
    moving_resampled_final = sitk.Resample(moving_resampled_affine, fixed_img, bspline_transform, 
                                     INTERPOLATOR, 0.0, moving_img.GetPixelID())
    bspline_mutual_info = calculate_mutual_info(np.array(he_gray), np.array(get_pil_from_itk(moving_resampled_final)))
    verbose_print('B-spline mutual information metric: {0}'.format(bspline_mutual_info))
    #Transform the original TP53 into the aligned TP53 image
    moving_rgb_affine = sitk_transform_rgb(tp53, he_norm_small, affine_transform, INTERPOLATOR)
    tp53_aligned = sitk_transform_rgb(moving_rgb_affine, he_norm_small, bspline_transform, INTERPOLATOR)
    thumbnail(show_alignment(he_norm_small, tp53_aligned, prefilter = True))
    #Remove backgrounds from TP53 and H&E images
    tp53_filtered = filter_green(tp53_aligned)
    he_filtered = filter_green(he_norm_small)
    tp53_filtered = filter_grays(tp53_filtered, tolerance = 3)
    he_filtered = filter_grays(he_filtered, tolerance = 15)

    #Visually compare alignment between the registered TP53 and original H&E image
    comparison_post = show_alignment(he_filtered, tp53_filtered)
    verbose_save_img(comparison_post.convert('RGB'), OUTPUT_PATH.joinpath(PREFIX + 'comparison_post_registration.jpeg'), 'JPEG' )
    
    ####################################
    # Generate cancer and tissue masks #
    ####################################

    def cancer_mask_v1(img, tile_size, cancer_thresh):
        #Determine Dab threshold with a smaller 1000x1000 image
        downsample = max(img.size)/1000
        img_small_size = tuple([np.int(np.round(dim/downsample)) for dim in img.size])
        img_small = img.resize(img_small_size, resample = Image.BICUBIC)
        hed_small = rgb2hed(img_small)
        dab_thresh = threshold_otsu_masked(hed_small)
        mask = []
        tgen = tile_gen(img, tile_size)
        shape = next(tgen)
        for tile in tgen:
            tile = tile.convert('RGB')
            tile_hed = rgb2hed(tile)
            tile_dab = -tile_hed[:,:,2]
            binary_dab = tile_dab > dab_thresh
            if np.array(binary2gray(binary_dab)).mean() < cancer_thresh:
                mask.append(0)
            else:
                mask.append(1)
        return np.reshape(mask, shape) 

    def cancer_mask(img, tile_size, cancer_thresh = 250):
        #Determine Dab threshold with a smaller 1000x1000 image
        downsample = max(img.size)/1000
        img_small_size = tuple([np.int(np.round(dim/downsample)) for dim in img.size])
        img_small = img.resize(img_small_size, resample = Image.BICUBIC)
        hed_small = rgb2hed(img_small)
        dab_thresh = threshold_otsu_masked(hed_small)
        #Extract Dab channel (stain)
        hed = rgb2hed(img)
        dab_channel = -hed[:,:,2]
        dab_binary = dab_channel > dab_thresh
        #Remove background staining 
        dab_binary_filtered = remove_small_holes(dab_binary, area_threshold = 64)
        tgen = tile_gen(binary2gray(dab_binary_filtered), tile_size)
        c_mask = threshold_mask(tgen, cancer_thresh)
        return c_mask 

    def show_mask(mask):
        mask = mask.astype(np.uint8)
        plt.imshow(np.reshape(mask, mask.shape[::-1]))
        plt.colorbar()


    tile_size = 299*ALIGNMENT_MAG/TILE_MAG

    #Generate cancer mask and tissue mask from filtered tp53 image
    c_mask = cancer_mask(tp53_filtered, tile_size, 250)
    t_mask_tp53 = tissue_mask(tp53_filtered, tile_size)
    t_mask_he = tissue_mask(he_filtered, tile_size)

    def dab_tile_array(img, tile_size):
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
        dab_array = dab_tile_array(img, tile_size)
        binary_mask = ((cancer_thresh < dab_array) & (dab_array < non_cancer_thresh))
        return (np.invert(binary_mask)).astype(np.uint8)

    #Generate tissue mask with tissue common to both the TP53 and H&E image
    t_mask = np.logical_not(np.logical_not(t_mask_tp53) & np.logical_not(t_mask_he))

    u_mask = uncertain_mask(tp53_filtered, tile_size, CANCER_THRESH, NON_CANCER_THRESH)
    u_mask_filtered = np.logical_not(np.logical_not(u_mask) & np.logical_not(t_mask))

    #Filter tissue mask such that any uncertain tiles are removed
    t_mask_filtered = np.zeros(t_mask.shape)
    for x in range(t_mask.shape[0]):
        for y in range(t_mask.shape[1]):
            if t_mask[x, y] == 0 and u_mask[x, y] == 1:
                t_mask_filtered[x, y] = False
            else:
                t_mask_filtered[x, y] = True

    #Make sure all cancer tiles exist in the tissue mask
    c_mask_filtered = np.logical_not(np.logical_not(c_mask) & np.logical_not(t_mask_filtered))


    def plot_masks(img, c_mask, t_mask,  tile_size, u_mask = None):
        img_overlay = img.copy()
        d = ImageDraw.Draw(img_overlay)
        tile_coords = tile_coordinates(img, tile_size)
        tile_coords['c_mask'] = c_mask.ravel()
        tile_coords['t_mask'] = t_mask.ravel()
        if u_mask is not None:
            tile_coords['u_mask'] = u_mask.ravel()
        width = int(np.round(tile_size*0.03))
        for row in tile_coords.itertuples(index = False):
            x_top_left, y_top_left = np.round(row[2:4])
            x_bottom_right, y_bottom_right = np.ceil(row[4:6])
            if row[6] == 0:
                outline = 'red'
            elif row[7] == 0:
                outline = 'lime'
            else:
                outline = 'grey'
            if u_mask is not None and row[8] == 0:
                outline = 'orange'
            d.rectangle([(x_top_left, y_top_left), (x_bottom_right, y_bottom_right)], outline = outline, width = width)
        return img_overlay


    overlay_tp53 = plot_masks(tp53_filtered, c_mask_filtered, t_mask_filtered, tile_size, u_mask_filtered)
    verbose_save_img(overlay_tp53.convert('RGB'), OUTPUT_PATH.joinpath(PREFIX + 'TP53_overlay.jpeg'), 'JPEG' )

    overlay_he = plot_masks(he_filtered, c_mask_filtered, t_mask_filtered, tile_size, u_mask_filtered)
    verbose_save_img(overlay_he.convert('RGB'), OUTPUT_PATH.joinpath(PREFIX + 'HE_overlay.jpeg'), 'JPEG' )

    ##############
    # Save Tiles #
    ##############

    #Make Directory to save tiles
    TILES_PATH = OUTPUT_PATH.joinpath('tiles_' + str(TILE_MAG) + 'x') 
    os.makedirs(TILES_PATH, exist_ok = True)

    def save_test_tiles(path, tile_gen, cancer_mask, tissue_mask, uncertain_mask, prefix = ''):
        os.makedirs(TILES_PATH.joinpath('cancer'), exist_ok = True)
        os.makedirs(TILES_PATH.joinpath('non-cancer'), exist_ok = True)
        os.makedirs(TILES_PATH.joinpath('uncertain'), exist_ok = True)
        x_tiles, y_tiles = next(tile_gen)
        verbose_print('Whole Image Size is {0} x {1}'.format(x_tiles, y_tiles))
        i = 0 
        for tile in tile_gen:
            img = tile.convert('RGB')
            #Name tile as horizontal position _ vertical position starting at (0,0)
            tile_name = prefix + str(np.floor_divide(i,x_tiles)) + '_' +  str(i%x_tiles) 
            if uncertain_mask.ravel()[i] == 0:
                img.save(path.joinpath('uncertain', tile_name + '.jpeg'), 'JPEG')
            elif cancer_mask.ravel()[i] == 0:
                img.save(path.joinpath('cancer', tile_name + '.jpeg'), 'JPEG')
            elif tissue_mask.ravel()[i] == 0:
                img.save(path.joinpath('non-cancer', tile_name + '.jpeg'), 'JPEG')
            i += 1
        verbose_print('Exported tiles for {0}'.format(prefix))

    he_tile_mag = read_slide_at_mag(he_slide, TILE_MAG)
    he_std = LuminosityStandardizer.standardize(np.array(he_tile_mag))
    he_transformed = normaliser.transform(he_std)
    he_norm = Image.fromarray(he_transformed)

    tgen = tile_gen(he_norm, 299)
    save_test_tiles(TILES_PATH, tgen, c_mask_filtered, t_mask_filtered, u_mask_filtered, prefix = PREFIX)

    t_mask_filtered.astype(np.bool)

    tissue_tiles = np.invert(t_mask_filtered.astype(np.bool)).sum()

    uncertain_tiles = np.invert(u_mask_filtered).sum()

    cancer_tiles = np.invert(c_mask_filtered).sum()

    uncertain_percentage = uncertain_tiles/tissue_tiles

    cancer_percentage = cancer_tiles/tissue_tiles

    tile_metrics = pd.DataFrame(np.array([[tissue_tiles, uncertain_tiles, cancer_tiles, uncertain_percentage, cancer_percentage]]), 
                                index = [PREFIX],
                                 columns = ['tissue_tiles', 'uncertain_tiles', 'cancer_tiles', 'uncertain_percentage', 'cancer_percentage'])

    tile_metrics

    tile_metrics.to_csv(OUTPUT_PATH.joinpath(PREFIX + 'metrics.csv'))


