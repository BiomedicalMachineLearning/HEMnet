from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps, ImageChops
import numpy as np
import SimpleITK as sitk
from skimage.exposure import histogram
from skimage.filters import threshold_otsu

######################
# Image Registration #
######################

def get_itk_from_pil(pil_img):
    """Converts Pillow image into ITK image
    """
    return sitk.GetImageFromArray(np.array(pil_img))

def get_pil_from_itk(itk_img):
    """Converts ITK image into Pillow Image
    """
    return Image.fromarray(sitk.GetArrayFromImage(itk_img).astype(np.uint8))

def show_alignment(fixed_img, moving_img, prefilter = None):
    """Visualises alignment of fixed image with moving image
    
    Fixed image is displayed as blue
    Moving image is displayed as pink 
    """
    if prefilter == 'TP53':
        tp53_filtered = filter_green(moving_img)
        tp53_filtered = filter_grays(tp53_filtered, tolerance = 3)
        moving_img = filter_otsu_global(tp53_filtered, 'PIL')
        he_filtered = filter_green(fixed_img)
        he_filtered = filter_grays(he_filtered, tolerance = 15)
        fixed_img = filter_otsu_global(he_filtered, 'PIL')
    background = (255,255,255)
    img_red = ImageOps.colorize(moving_img.convert('L'), (255, 0, 0), background)
    img_blue = ImageOps.colorize(fixed_img.convert('L'), (0, 0, 255), background)
    img_red.putalpha(120)
    img_blue.putalpha(70)
    return Image.alpha_composite(img_red, img_blue)

def sitk_transform_rgb(moving_rgb_img, fixed_rgb_img, transform, interpolator = sitk.sitkLanczosWindowedSinc):
    """Applies a Simple ITK transform (e.g. Affine, B-spline) to an RGB image
    
    The transform is applied to each channel
    
    Parameters
    ----------
    moving_rgb_img : Pillow Image 
        This image will be transformed to produce the output image
    fixed_rgb_img : Pillow Image
        This reference image provides the output information (spacing, size, and direction) of the output image
    transform : SimpleITK transform
        Generated from image registration
    interpolator : SimpleITK interpolator
    
    Returns
    -------
    rgb_transformed : Pillow Image
        Transformed moving image 
    """
    transformed_channels = []
    r_moving, g_moving, b_moving, = moving_rgb_img.convert('RGB').split()
    r_fixed, g_fixed, b_fixed = fixed_rgb_img.convert('RGB').split()
    for moving_img, fixed_img in [(r_moving, r_fixed), (g_moving, g_fixed), (b_moving, b_fixed)]:
        moving_img_itk = get_itk_from_pil(moving_img)
        fixed_img_itk = get_itk_from_pil(fixed_img)
        transformed_img = sitk.Resample(moving_img_itk, fixed_img_itk, transform, 
                            interpolator, 0.0, moving_img_itk.GetPixelID())
        transformed_channels.append(get_pil_from_itk(transformed_img))
    rgb_transformed = Image.merge('RGB', transformed_channels)
    return rgb_transformed    

def start_plot():
    """Setup data for plotting
    
    Invoked when StartEvent happens at the beginning of registration.
    """
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

def end_plot():
    """Cleanup the data and figures 
    """
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

def update_plot(registration_method):
    """Plot metric value after each registration iteration
    
    Invoked when IterationEvent happens.
    """
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.show()
    
def update_multires_iterations():
    """Update the index in the metric values list that corresponds to a change in registration resolution
    
    Invoked when the sitkMultiResolutionIterationEvent happens.
    """
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))
    
def plot_metric(title = 'Plot of registration metric vs iterations'):
    """Plots the mutual information over registration iterations
    
    Parameters
    ----------
    title : str
    
    Returns
    -------
    fig : matplotlib figure
    """
    global metric_values, multires_iterations
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Iteration Number', fontsize=12)
    ax.set_ylabel('Mutual Information Cost', fontsize=12)
    ax.plot(metric_values, 'r')
    ax.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*', label = 'change in resolution')
    ax.legend()
    return fig

#################
# Image Filters #
#################

def filter_green(img, g_thresh = 240):
    """Replaces green pixels greater than threshold with white pixels
    
    Used to remove background from tissue images
    """
    img = img.convert('RGB')
    r, g, b = img.split()
    green_mask = (np.array(g) > 240)*255
    green_mask_img = Image.fromarray(green_mask.astype(np.uint8), 'L')
    white_image = Image.new('RGB', img.size, (255,255,255))
    img_filtered = img.copy()
    img_filtered.paste(white_image, mask = green_mask_img)
    return img_filtered

def filter_grays(img, tolerance = 3):
    """Replaces gray pixels greater than threshold with white pixels
    
    Used to remove background from tissue images
    """
    img = img.convert('RGB')
    r, g, b = img.split()
    rg_diff = np.array(ImageChops.difference(r,g)) <= tolerance
    rb_diff = np.array(ImageChops.difference(r,b)) <= tolerance
    gb_diff = np.array(ImageChops.difference(g,b)) <= tolerance
    grays = (rg_diff & rb_diff & gb_diff)*255
    grays_mask = Image.fromarray(grays.astype(np.uint8), 'L')
    white_image = Image.new('RGB', img.size, (255,255,255))
    img_filtered = img.copy()
    img_filtered.paste(white_image, mask = grays_mask)
    return img_filtered

def filter_otsu_global(img, mode = 'PIL'):
    """Performs global otsu thresholding on Pillow Image
    """
    img_gray = img.convert('L')
    threshold = threshold_otsu(np.array(img_gray))
    img_binary = np.array(img_gray) > threshold
    if mode == '1':
        return img_binary
    else:
        return binary_array_to_pil(img_binary)

####################
# Image Operations #
####################

def binary_array_to_pil(array):
    """Converts a binary array to a Pillow Image
    """
    img_shape = array.shape
    img = Image.new('1', img_shape)
    int_list = array.astype(int).tolist()
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i,j] = int_list[i][j]
    return ImageOps.mirror(img).rotate(90, expand = True)

def binary2gray(img):
    """Converts binary arrays to grayscale Pillow image
    
    Parameters
    ----------
    img : binary ndarray
    
    Returns
    -------
    out : grayscale Pillow image
    """
    img_rescaled = (img*255).astype('uint8')
    return Image.fromarray(img_rescaled).convert('L')

def tile_gen(img, tile_size):
    '''Generates tiles for Pillow images
    '''
    width, height = img.size
    x_tiles = int(np.floor(width/tile_size))
    y_tiles = int(np.floor(height/tile_size))
    yield (x_tiles, y_tiles)
    for y in range(y_tiles):
        for x in range(x_tiles):
            x_coord = x*tile_size
            y_coord = y*tile_size
            yield img.crop((x_coord, y_coord, np.int(np.round(x_coord+tile_size)), np.int(np.round(y_coord+tile_size))))
            
def max_tiles(img_dim, tile_dim, overlap = 0):
    """Maximum tiles that can fit across an image dimension
    
    Parameters
    ----------
    img_dim : int, float
    tile_dim : int, float
    overlap : float
        overlap as a proportion - zero is no overlap, one is complete overlap
    
    Returns
    -------
    out : int
    """
    max_tiles = ((img_dim/tile_dim) - 1)/(1- overlap) + 1
    return int(np.floor(max_tiles))

def tile_coordinates(img, tile_size, overlap = 0):
    """Computes a dataframe of tile coordinates for an image
    
    Parameters
    ----------
    img : Pillow image 
    tile_size : int, float
    overlap : float
        overlap as a proportion - zero is no overlap, one is complete overlap
    
    Returns
    -------
    out : DataFrame
    """
    width, height = img.size
    x_tiles = max_tiles(width, tile_size, overlap)
    y_tiles = max_tiles(height, tile_size, overlap)
    coords = []
    for y in range(y_tiles):
        for x in range(x_tiles):
            x_top_left = x*tile_size*(1-overlap)
            y_top_left = y*tile_size*(1-overlap)
            x_bottom_right = x_top_left + tile_size
            y_bottom_right = y_top_left + tile_size
            coords.append([x, y, x_top_left, y_top_left, x_bottom_right, y_bottom_right])
    return pd.DataFrame(coords, columns = ['X','Y','x_top_left', 'y_top_left', 'x_bottom_right', 'y_bottom_right'])
            
##################
# Mask Functions #
##################

def threshold_otsu_masked(hed_img):
    """Otsu thresholds the DAB component of an image
    
       Masks the background so only tissue regions are used for threshold calculation. 
       Without masking, the background would likely be thresholded from the tissue due to background staining. 
       
    Parameters
    ----------
    hed_img : ndarray
        The image in Hematoxylin, Eosin, DAB (HED) format, in a 3-D array of shape ``(.., .., 3)``
        
    Returns
    -------
    threshold : float
        Pixel threshold value that best segments the DAB stain, as determined by the ostu method
    """
    dab = -hed_img[:, :, 2]
    hem = -hed_img[:,:, 0]
    hem_thresh = threshold_otsu(hem)
    hem_binary_otsu = hem > hem_thresh
    #Use binary mask to mask out background of DAB images
    dab_masked = np.where(hem_binary_otsu == False, dab, 0)
    dab_values = np.array([i for i in dab_masked.ravel() if i != 0])
    #Otsu Thresholding
    hist, bin_centers = histogram(dab_values, 256)
    hist = hist.astype(float)
    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    #Consider all cells are normal if variance is too small
    if variance12[idx] < 500000:
        threshold = 0
    else:
        threshold = bin_centers[:-1][idx]
    return threshold

def threshold_mask(tile_gen, threshold):
    """Creates mask from tiles using a threshold
    
    Averages the pixel intensities of each tile and applies a threshold. 
    
    Parameters
    ----------
    tile_gen : tile generator
    threshold : float
    
    Returns
    -------
    out : ndarray
        binary mask where zero is below threshold
    """
    mask = []
    shape = next(tile_gen)
    for tile in tile_gen:
        if np.array(tile).mean() < threshold:
            mask.append(0)
        else:
            mask.append(1)
    return np.reshape(mask, shape)