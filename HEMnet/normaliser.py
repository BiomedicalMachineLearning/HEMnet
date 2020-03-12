# ------------------------------------------------------------------------
# Tools for stain normalisation
# ------------------------------------------------------------------------

import numpy as np
import cv2 as cv
from staintools.preprocessing.input_validation import is_uint8_image
from staintools import ReinhardColorNormalizer, LuminosityStandardizer, StainNormalizer 
from staintools.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from staintools.miscellaneous.optical_density_conversion import convert_OD_to_RGB
from staintools.miscellaneous.get_concentrations import get_concentrations

class LuminosityStandardizerIterative(LuminosityStandardizer):
    """
    Transforms image to a standard brightness
    Modifies the luminosity channel such that a fixed percentile is saturated
    
    Standardiser can fit to source slide image and apply the same luminosity standardisation settings to all tiles generated
    from the source slide image
    """
    def __init__(self):
        super().__init__()
        self.p = None
        
    def fit(self, I, percentile = 95):
        assert is_uint8_image(I), "Image should be RGB uint8."
        I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        L_float = I_LAB[:, :, 0].astype(float)
        self.p = np.percentile(L_float, percentile)
        
    def standardize_tile(self, I):
        I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        L_float = I_LAB[:, :, 0].astype(float)
        I_LAB[:, :, 0] = np.clip(255 * L_float / self.p, 0, 255).astype(np.uint8)
        I = cv.cvtColor(I_LAB, cv.COLOR_LAB2RGB)
        return I

class ReinhardColorNormalizerIterative(ReinhardColorNormalizer):
    """
    Normalise each tile from a slide to a target slide using the method of:
    E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley,
    'Color transfer between images'

    Normaliser can fit to source slide image and apply the same normalisation settings to all tiles generated from the
    source slide image

    Attributes
    ----------
    target_means : tuple float
        means pixel value for each channel in target image
    target_stds : tuple float
        standard deviation of pixel values for each channel in target image
    source_means : tuple float
        mean pixel value for each channel in source image
    source_stds : tuple float
        standard deviation of pixel values for each channel in source image

    Methods
    -------
    fit_target(target)
        Fit normaliser to target image
    fit_source(source)
        Fit normaliser to source image
    transform(I)
        Transform an image to normalise it to the target image
    transform_tile(I)
        Transform a tile using precomputed parameters that normalise the source slide image to the target slide image
    lab_split(I)
        Convert from RGB unint8 to LAB and split into channels
    merge_back(I1, I2, I3)
        Take separate LAB channels and merge back to give RGB uint8
    get_mean_std(I)
        Get mean and standard deviation of each channel
    """
    def __init__(self):
        super().__init__()
        self.source_means = None
        self.source_stds = None

    def fit_target(self, target):
        """Fit to a target image

        Parameters
        ----------
        target : Image RGB uint8

        Returns
        -------
        None
        """
        means, stds = self.get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def fit_source(self, source):
        """Fit to a source image

        Parameters
        ----------
        source : Image RGB uint8

        Returns
        -------
        None
        """
        means, stds = self.get_mean_std(source)
        self.source_means = means
        self.source_stds = stds

    def transform_tile(self, I):
        """Transform a tile using precomputed parameters that normalise the source slide image to the target slide image
        Parameters
        ----------
        I : Image RGB uint8

        Returns
        -------
        transformed_tile : Image RGB uint8
        """
        I1, I2, I3 = self.lab_split(I)
        norm1 = ((I1 - self.source_means[0]) * (self.target_stds[0] / self.source_stds[0])) + self.target_means[0]
        norm2 = ((I2 - self.source_means[1]) * (self.target_stds[1] / self.source_stds[1])) + self.target_means[1]
        norm3 = ((I3 - self.source_means[2]) * (self.target_stds[2] / self.source_stds[2])) + self.target_means[2]
        return self.merge_back(norm1, norm2, norm3)

class StainNormalizerIterative(StainNormalizer):
    """Normalise each tile from a slide to a target slide using the Macenko or Vahadane method
    """
    def __init__(self):
        super().__init__()
        self.maxC_source = None
        
    def fit_source(self, I):
        self.stain_matrix_source = self.extractor.get_stain_matrix(I)
        source_concentrations = get_concentrations(I, self.stain_matrix_source)
        self.maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        
    def transform_tile(self, I):
        source_concentrations = get_concentrations(I, self.stain_matrix_source)
        source_concentrations *= (self.maxC_target / self.maxC_source)
        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        return tmp.reshape(I.shape).astype(np.uint8)

