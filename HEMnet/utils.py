from PIL import Image, ImageOps, ImageChops
import numpy as np
import SimpleITK as sitk
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

def transform_rgb(rgb_img, transform_param_map, resize = None):
    r, g, b, = rgb_img.convert('RGB').split()
    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(transform_param_map)
    transformed_channels = []
    for img in [r, g, b]:
        img_itk = get_itk_from_pil(img)
        transformix.SetMovingImage(img_itk)
        transformix.Execute()
        transformed_img = get_pil_from_itk(transformix.GetResultImage())
        transformed_channels.append(transformed_img)
    rgb_transformed = Image.merge('RGB', transformed_channels)
    if resize is None:
        return rgb_transformed
    else: 
        downsample = max(rgb_transformed.size)/resize
        final_size = tuple([np.int(np.round(dim/downsample)) for dim in rgb_transformed.size])
        return rgb_transformed.resize(final_size, resample = Image.BICUBIC)

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

