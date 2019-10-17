# ------------------------------------------------------------------------
# Functions for working with OpenSlide objects for Whole Slide Images
# ------------------------------------------------------------------------

from openslide import open_slide
from PIL import Image, ImageOps, ImageChops

def highest_mag(slide):
    """Returns the highest magnification for the slide
    """
    return int(slide.properties['aperio.AppMag'])

def level_mags(slide):
    """Returns the magnification for each level in a slide
    """
    return [highest_mag(slide)/downsample for downsample in slide.level_downsamples]

def get_level_size(slide, level):
    """Returns the dimensions of a level
    """
    return slide.level_dimensions[level]

def get_level_mag(slide, level):
    """Returns the magnification at a particular level
    """
    return level_mags(slide)[level]

def get_level_for_mag(slide, mag):
    """Get the level corresponding to a certain magnification, if available
    """
    level_mags_rounded = list(np.round(level_mags(slide), decimals = 2))
    if mag in level_mags_rounded:
        return level_mags_rounded.index(mag)
    else: 
        return None
    
def get_mag_for_size(slide, size):
    """Get the magnification corresponding to an image size at the highest magnification
    """
    max_size = slide.dimensions
    max_mag = highest_mag(slide)
    downsample = np.average([max_dim/size_dim for max_dim, size_dim in zip(max_size, size)])
    return max_mag/downsample

def get_size_for_mag(slide, mag):
    """Get the image size the highest magnification image would have to be resized to get an equivalent magnification
    """
    max_size = slide.dimensions
    max_mag = highest_mag(slide)
    downsample = max_mag/mag
    return [np.int(np.round(dim/downsample)) for dim in max_size]

def read_slide_at_mag(slide, mag):
    """Outputs at Pillow Image for a particular magnification 
    """
    exact_level = get_level_for_mag(slide, mag)
    if exact_level is not None:
        return slide.read_region((0,0), exact_level, get_level_size(slide, exact_level))
    else:
        max_size = slide.dimensions
        region_size = tuple(get_size_for_mag(slide, mag))
        downsample = np.average([max_dim/region_dim for max_dim, region_dim in zip(max_size, region_size)])
        best_level = slide.get_best_level_for_downsample(downsample)
        best_level_size = get_level_size(slide, best_level)
        best_level_img = slide.read_region((0,0), best_level, best_level_size)
        return best_level_img.resize(region_size, resample = Image.BICUBIC)  
    
def tile_gen_at_mag(wsi, mag, tile_size):
    """Generates tiles from whole slide images at a particular magnification 
    """
    #Get size of WSI at Level 0 (Max Magnification)
    x0, y0 = wsi.level_dimensions[0]
    #Get size of WSI at the mag we want
    x_mag, y_mag = get_size_for_mag(wsi, mag)
    x_tiles = int(np.floor(x_mag/tile_size))
    y_tiles = int(np.floor(y_mag/tile_size))
    #Scale tile size accordingly
    scale = highest_mag(wsi)/mag
    yield (x_tiles, y_tiles)
    tiles = []
    for y in range(y_tiles):
        for x in range(x_tiles):
            x_coord = round(x*scale*tile_size)
            y_coord = round(y*scale*tile_size)
            tile = wsi.read_region((x_coord, y_coord), 0, (tile_size, tile_size))
            yield tile.resize((tile_size, tile_size), resample = Image.BICUBIC)