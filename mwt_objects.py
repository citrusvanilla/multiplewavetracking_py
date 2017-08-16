##
##  Near-shore Wave Tracking
##  mwt_objects.py
##
##  Created by Justin Fung on 8/1/17.
##  Copyright 2017 justin fung. All rights reserved.
##
## ========================================================

"""Objects for implementing wave tracking."""

from __future__ import division

import cv2
import math
import numpy as np
from collections import deque


SEARCH_REGION_BUFFER = 15           # pixel height to buffer a sections's search region for other sections
TRACKING_HISTORY = 300              # macro is in section.displacement attribute

ANALYSIS_FRAME_WIDTH = 320          # width of frame in analysis
ANALYSIS_FRAME_HEIGHT = 180         # height of frame in analysis

DISPLACEMENT_THRESHOLD = 10         # minimum displacement to be considered an actual wave
MASS_THRESHOLD = 1000           # minimum mass to be considered an actual wave

GLOBAL_WAVE_AXIS = 5.0              # axis of major waves in the scene, counter clockwise from horizon

NAME_SEED = 0                       # integer var seed for naming detected waves


class Section():
    """filtered contours become "sections" with the following attributes.
       dynamic attributes are updated in each frame through tracking routine.
    """
    def __init__(self, points, birth):
        self.name = _generate_name()
        self.points = points 
        self.birth = birth
        self.axis_angle = GLOBAL_WAVE_AXIS 

        self.centroid = _get_centroid(self.points)
        self.original_axis = _get_standard_form_line(self.centroid, 
                                                     self.axis_angle)        
        
        self.searchROI_coors = _get_searchROI_coors(self.centroid, 
                                                    self.axis_angle, 
                                                    SEARCH_REGION_BUFFER, 
                                                    ANALYSIS_FRAME_WIDTH)           # tuple of (x,y)
        self.boundingbox_coors = np.int0(cv2.boxPoints(cv2.minAreaRect(points)))
        self.displacement = 0
        self.displacement_vec = deque([0], maxlen = TRACKING_HISTORY)
        self.mass = len(self.points)
        self.is_wave = False
        self.death = None
        

## ========================================================

def _get_standard_form_line(point, angle_from_horizon):
    """
    A function returning a 3-element array corresponding to coefficients 
    of the standard form for a line of Ax+By=C.  Requires one point in [x,y],
    and an angle from the horizion in degrees.

    Args:
      point: a two-element array in [x,y] representing a point 
      angle_from_horizon: a float representing counterclockwise angle
                          from horizon of the line

    Returns:
      coefficients: a three-element array as [A,B,C]
    """
    coefficients = [None, None, None]

    coefficients[0] = np.tan(np.deg2rad(-angle_from_horizon))
    coefficients[1] = -1
    coefficients[2] = point[1] - np.tan(np.deg2rad(-angle_from_horizon)) \
                      * point[0]

    return coefficients


def _get_centroid(points):
    """
    function for getting the centroid of an object that is represented
    by positive pixels in a bilevel image.

    Args:
      points: array of points
    Returns:
      centroid: 2 element array as [x,y] if points is not empty
    """

    centroid = None

    if points is not None:
        centroid = [int(sum([p[0][0] for p in points]) / len(points)), 
                    int(sum([p[0][1] for p in points]) / len(points))]

    return centroid


def _get_searchROI_coors(centroid, axis_angle, searchROI_buffer, frame_width):
    delta_y_left = np.round(centroid[0] * np.tan(np.deg2rad(axis_angle)))
    delta_y_right = np.round((frame_width - centroid[0]) * np.tan(np.deg2rad(axis_angle)))

    upper_left_y = int(centroid[1] + delta_y_left - searchROI_buffer)
    upper_left_x = 0
    upper_right_y = int(centroid[1] - delta_y_right - searchROI_buffer)
    upper_right_x = frame_width

    lower_left_y = int(centroid[1] + delta_y_left + searchROI_buffer)
    lower_left_x = 0
    lower_right_y = int(centroid[1] - delta_y_right + searchROI_buffer)
    lower_right_x = frame_width

    return [[upper_left_x,upper_left_y],
            [upper_right_x,upper_right_y],
            [lower_right_x,lower_right_y],
            [lower_left_x,lower_left_y]]

def _generate_name():
    """name generator for identifying waves by incremental numeric sequence.

    Args:
      None

    Returns:
      NAME_SEED: next integer in a sequence seeded by "NAME_SEED" global var.
    """
    global NAME_SEED
    NAME_SEED += 1

    return NAME_SEED


def update_searchROI_coors(wave):
    """function that adjusts the search ROI for tracking a wave in future 
       frames.

    Args:
        wave: a Section object

    Returns:
        VOID: updates the section.searchROI_coors attribute
    """
    wave.searchROI_coors = _get_searchROI_coors(wave.centroid, 
                                                wave.axis_angle, 
                                                SEARCH_REGION_BUFFER, 
                                                ANALYSIS_FRAME_WIDTH)


def update_points(wave, frame):
    """captures all positive pixels the search ROI based on 
       measurement of the wave's position in the previous frame by
       using a mask.

    Args:
      wave: a Section object
      frame: frame in which to obtain new binary representation of the wave

    Returns:
      VOID: returns all points to section.points attribute.
    """
    
    points = None
    
    # create mask using searchROI coordinates
    # DOCS: https://stackoverflow.com/questions/17437846/
    #       opencv-zero-pixels-outside-of-region-of-interest-with-rotated-rectangle
    # DOCS: https://stackoverflow.com/questions/10469235/
    #       opencv-apply-mask-to-a-color-image
    rect = wave.searchROI_coors
    poly = np.array([rect], dtype=np.int32)                                 #poly object
    img = np.zeros((ANALYSIS_FRAME_HEIGHT, ANALYSIS_FRAME_WIDTH), np.uint8)  #blank mask
    img = cv2.fillPoly(img, poly, 255)                                          #mask with ROI
    res = cv2.bitwise_and(frame, frame, mask = img) 
    points = cv2.findNonZero(res)
    
    # update points
    wave.points = points


def update_death(wave, frame_number):
    """checks to see if wave has died, which occurs when no pixels are 
       found in the wave's search ROI.  "None" indicates wave is alive,
       while an integer represents the frame number of death.

    Args:
    wave: a Section object
    frame_number: number of frame in a video sequence.

    Returns:
    VOID: sets wave death to section.death attribute.
    """
    if wave.points is None:
        wave.death = frame_number


def update_centroid(wave):
    """calculates the mass of all positive pixels in a wave's search ROI,
       using first-order moments.

    Args:
    wave: a Section object

    Returns:
    VOID: updates section.centroid
    """

    wave.centroid = _get_centroid(wave.points)


def update_boundingbox_coors(wave):
    """finds minimum area rectangle that bounds the points inside the wave's
       search ROI.  returns four coordinates of the bounding box.

    Args:
    wave: a Section object

    Returns:
    VOID: updates section.boundingbox_coors attribute.
    """

    boundingbox_coors = None

    if wave.points is not None:
        # obtain moments of the object from its points array
        x = [p[0][0] for p in wave.points]
        y = [p[0][1] for p in wave.points]
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x)
        std_y = np.std(y)

        # we only capture points without outliers for display purposes
        points_without_outliers = np.array([p[0] for p in wave.points \
                                            if np.abs(p[0][0] - mean_x) < 3 * std_x \
                                            and np.abs(p[0][1] - mean_y) < 3 * std_y])

        rect = cv2.minAreaRect(points_without_outliers)
        box = cv2.boxPoints(rect)
        boundingbox_coors = np.int0(box)

    wave.boundingbox_coors = boundingbox_coors


def update_displacement_vec(wave):
    """calculates the displacement of the wave orthogonal to its major axis.
       stores displacement to a deque .

    Args:
    wave: a Section object

    Returns:
    VOID: append positional displacement to the queue.
    """
    if wave.centroid is not None:
        a = wave.original_axis[0]
        b = wave.original_axis[1]
        c = wave.original_axis[2]
        x0 = wave.centroid[0]
        y0 = wave.centroid[1]

        disp = np.abs(a*x0 + b*y0 + c) / math.sqrt(a**2 + b**2)

        wave.displacement_vec.append(int(disp))


def update_displacement(wave):
    """evaluates absolute displacement based on comparison with original position.

    Args:
    wave: a Section object

    Returns:
    VOID: updates section.displacement attribute.
    """
    wave.displacement = wave.displacement_vec[-1]


def update_mass(wave):
    """calculates mass of the wave by weighting each pixel in a search ROI equally
       and performing a simple count.

    Args:
    wave: a Section object

    Returns:
    VOID: updates section.mass attribute.
    """
    #if len(wave.points) > wave.max_mass:
    #    wave.mass = len(wave.points)
    #else:
    #    return
    mass = 0
    
    if wave.points is not None:
        mass = len(wave.points)

    wave.mass = mass


def update_is_wave(wave):
    """updates the boolean section.is_wave to True if wave mass and wave displacement 
       exceed user-defined thresholds.

    Args:
    wave: a Section object

    Returns:
    VOID: changes section.is_wave boolean to True is conditions are met.
    """
    is_wave = False

    if wave.displacement >= DISPLACEMENT_THRESHOLD \
       and wave.mass >= MASS_THRESHOLD:
        #print "WAVE DETECTED."
        is_wave = True
    
    wave.is_wave = is_wave
