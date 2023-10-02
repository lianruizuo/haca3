import sys
import argparse
import torch
import nibabel as nib
import numpy as np
import os
from torchvision.transforms import ToTensor, CenterCrop, Compose, ToPILImage
from modules import HACA3


def normalize_intensity(image):
    """
    Normalize image intensity (from 0% to 99%) to 0-1.

    ===INPUTS===
    * image: numpy.ndarray (224, 224, 224)
        Input image

    ===OUTPUTS===
    * image: np.ndarray (224, 224, 224)
        Normalized image.
    * thresh: np.float
        99 percentile of the input intensity.
    """
    thresh = np.percentile(image.flatten(), 99)
    image = np.clip(image, a_min=0.0, a_max=thresh)
    image = image / thresh
    return image, thresh
