"""Routine for preprocessing video frames.

Method of preprocessing is:
    1. resize image
    2. extract foreground
    3. denoise image
"""
from __future__ import division

import cv2

# Resize factor (downsize) for analysis:
RESIZE_FACTOR = 0.25

# Number of frames that constitute the background history:
BACKGROUND_HISTORY = 900

# Number of gaussians in BG mixture model:
NUM_GAUSSIANS = 5

# Minimum percent of frame considered background:
BACKGROUND_RATIO = 0.7

# Morphological kernel size (square):
MORPH_KERN_SIZE = 3

# Init the background modeling and foreground extraction mask.
mask = cv2.bgsegm.createBackgroundSubtractorMOG(
    history=BACKGROUND_HISTORY,
    nmixtures=NUM_GAUSSIANS,
    backgroundRatio=BACKGROUND_RATIO,
    noiseSigma=0,
)

# Init the morphological transformations for denoising kernel.
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (MORPH_KERN_SIZE, MORPH_KERN_SIZE)
)


def _resize(frame):
    """Resizing function utilizing OpenCV.

    Args:
      frame: A frame from a cv2.video_reader object to process

    Returns:
      resized_frame: the frame, resized
    """
    resized_frame = cv2.resize(
        frame,
        None,
        fx=RESIZE_FACTOR,
        fy=RESIZE_FACTOR,
        interpolation=cv2.INTER_AREA,
    )

    return resized_frame


def preprocess(frame):
    """Preprocess video frames.

    Resize, perform background modeling, and denoise.

    Args:
        frame: A frame from a cv2.video_reader object to process

    Returns:
        output: the preprocessed frame
    """
    # 1. Resize the input.
    output = _resize(frame)

    # 2. Model the background and extract the foreground with a mask.
    output = mask.apply(output)

    # 3. Apply the morphological operators to suppress noise.
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)

    return output
