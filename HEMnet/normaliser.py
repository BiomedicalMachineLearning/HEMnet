# ------------------------------------------------------------------------
# Tools for stain normalisation
# ------------------------------------------------------------------------

from staintools import ReinhardColorNormalizer

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
