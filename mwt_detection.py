##
##  Near-shore Wave Tracking
##  mwt_detection.py
##
##  Created by Justin Fung on 8/1/17.
##  Copyright 2017 justin fung. All rights reserved.
##
## ========================================================

"""Routine for detecting potential waves.
   Helper functions above, routines below.

   Method for detecting is:
   -1. detect contours
   -2. filter contours
   -3. create list of "Section" objects from filtered contours
   """

from __future__ import division

import cv2
import math
import numpy as np
from collections import deque

from mwt_objects import Section

FLAGS_FILTER_BY_AREA = True         # boolean flag to filter blobs by area
FLAGS_FILTER_BY_INERTIA = True      # boolean flag to filter blobs by inertia (shape)

MINIMUM_AREA = 100                  # minimum area threshold for contour
MAXIMUM_AREA = 1e7                  # maximum area threshold for contour
MINIMUM_INERTIA_RATIO = 0.0         # minimum inertia threshold for contour
MAXIMUM_INERTIA_RATIO = 0.1         # maximum inertia threshold for contour


def find_contours(frame):
    """Contour finding function utilizing OpenCV.

    Args:
    frame: A frame from a cv2.video_reader object to process.

    Returns:
    contours: An array of contours, each represented by an array of points.
    """
    # Find All Contours: 
    # DOCS: http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#cv.FindContours
    # DOCS: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    _, contours, hierarchy = cv2.findContours(image = frame,                    # source
                                              mode = cv2.RETR_EXTERNAL,         # Contour retrieval mode; retrieves only the extreme outer contours
                                              method = cv2.CHAIN_APPROX_NONE,   # Contour approximation method; stores absolutely all the contour points
                                              hierarchy = None,                 # Optional output vector, containing information about the image topology
                                              offset= None)                     # Optional offset by which every contour point is shifted

    return contours


def filter_contour(contour,
                   area = FLAGS_FILTER_BY_AREA, 
                   inertia = FLAGS_FILTER_BY_INERTIA,
                   minArea = MINIMUM_AREA,
                   maxArea = MAXIMUM_AREA,
                   minInertiaRatio = MINIMUM_INERTIA_RATIO,
                   maxInertiaRatio = MAXIMUM_INERTIA_RATIO):
    """Contour filtering function utilizing OpenCV.

    Args:
    contour: A contour from an array of contours.
    area: boolean flag to filter contour by area.
    inertia: boolean flag to filter contour by inertia.
    minArea: minimum area threshold for contour.
    maxArea: maximum area threshold for contour.
    minInertiaRatio: minimum inertia threshold for contour.
    maxInertiaRatio: maximum inertia threshold for contour.

    Returns:
    ret: A boolean TRUE if contour meets conditions, else FALSE.
    """
    # initialize return value
    ret = True

    # obtains contour moments
    moments = cv2.moments(contour)

    # Filter Contours By Area
    if area == True and ret == True:
        area = cv2.contourArea(contour)
        if area < minArea or area >= maxArea:
            ret = False

    # Filter contours by inertia
    if inertia == True and ret == True:
        denominator = math.sqrt((2 * moments['m11']) ** 2 + (moments['m20'] - moments['m02']) ** 2)
        epsilon = 0.01
        ratio = 0.0

        if denominator > epsilon:
            cosmin = (moments['m20'] - moments['m02']) / denominator;
            sinmin = 2 * moments['m11'] / denominator;
            cosmax = -cosmin;
            sinmax = -sinmin;
                
            imin = 0.5 * (moments['m20'] + moments['m02']) - 0.5 * (moments['m20'] - moments['m02']) * cosmin - moments['m11'] * sinmin;
            imax = 0.5 * (moments['m20'] + moments['m02']) - 0.5 * (moments['m20'] - moments['m02']) * cosmax - moments['m11'] * sinmax;
            ratio = imin / imax;
        else:
            ratio = 1;  

        if ratio < minInertiaRatio or ratio >= maxInertiaRatio:
            ret = False
            #center.confidence = ratio * ratio;

    return ret

## ========================================================

def detect_sections(frame, frame_number):
    """Finds sections that meet certain criteria.

    Args:
    frame: a frame from a cv2.video_reader object.
    frame_number: number of the frame in the video sequence.

    Returns:
    sections: a list of Section objects.
    """    
    # convert to single channel for blob detection if necessary 
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # initiate empty list of sections
    sections = []

    # 1. find contours
    contours = find_contours(frame)
    #visual = frame

    # 2. filter contours
    for contour in contours:
        
        # filter
        if filter_contour(contour) == False:
            continue
        
        # if contour passes thresholds, convert to Section
        section = Section(points = contour,
                          birth = frame_number)

        # 3. add section to sections
        sections.append(section)

    return sections
