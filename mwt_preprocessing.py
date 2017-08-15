##
##  Near-shore Wave Tracking
##  mwt_preprocessing.py
##
##  Created by Justin Fung on 8/1/17.
##  Copyright 2017 justin fung. All rights reserved.
##
## ========================================================

"""Routine for preprocessing video frames.
   Helper functions above, routines below.

   Method of preprocessing is:
   -1. resize image
   -2. extract foreground 
   -3. denoise image
   """

from __future__ import division

import cv2

RESIZE_FACTOR = 0.25        # resize factor for video analysis
BACKGROUND_HISTORY = 300    # number of frames that constitute background history
NUM_GAUSSIANS = 5           # number of gaussians in BG mixture model
BACKGROUND_RATIO = 0.7      # minimum percent of frame considered background
MORPH_KERN_SIZE = 5         # morphological kernel size (square)


# 1. RESIZE RAW VIDEO FRAMES
# DOCS: http://docs.opencv.org/3.1.0/da/d6e/tutorial_py_geometric_transformations.html
def _resize(frame):
    """Resizing function utilizing OpenCV.

    Args:
    frame: A frame from a cv2.video_reader object to process.

    Returns:
    A resized frame.
    """
    return cv2.resize(frame,                                # input frame
                      None,                                 # output frame
                      fx = RESIZE_FACTOR,                   # x-axis scale factor
                      fy = RESIZE_FACTOR,                   # y-axis scale factor
                      interpolation = cv2.INTER_AREA)       # rescale interpolation method

# 2. BACKGROUND MODELING AND FOREGROUND EXTRACTION
# DOCS: http://docs.opencv.org/trunk/db/d5c/tutorial_py_bg_subtraction.html
mask = cv2.bgsegm.createBackgroundSubtractorMOG(history = BACKGROUND_HISTORY,           # length of history
                                                nmixtures = NUM_GAUSSIANS,              # num. of gaussian mixtures
                                                backgroundRatio = BACKGROUND_RATIO,     # minimum percent of frame considered background
                                                noiseSigma = 0)                         # noise strength

## 3. MORPHOLOGICAL TRANSFORMATIONS FOR DENOISING
## DOCS: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,                          # shape (rectangle)
                                   (MORPH_KERN_SIZE,MORPH_KERN_SIZE))       # kernel size (square)

## ========================================================

def preprocess(input):
    """Preprocesses video frames through resizing, background modeling,
    and denoising.

    Args:
    input: A frame from a cv2.video_reader object to process.

    Returns:
    A preprocessed frame.
    """

    # 1. resize
    output = _resize(input)

    # 2. mask input
    output = mask.apply(output)

    # 3. morphological operators to suppress noise
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)

    return output




