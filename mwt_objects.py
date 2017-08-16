##
##  Near-shore Wave Tracking
##  mwt_objects.py
##
##  Created by Justin Fung on 8/1/17.
##  Copyright 2017 justin fung. All rights reserved.
##
## ====================================================================

"""Objects for implementing wave tracking."""

from __future__ import division

import math
from collections import deque

import cv2
import numpy as np

# Pixel height to buffer a sections's search region for other sections:
SEARCH_REGION_BUFFER = 15           
# Length of Deque to keep track of displacement of the wave.
TRACKING_HISTORY = 300

# Width of frame in analysis steps (not original width):
ANALYSIS_FRAME_WIDTH = 320
# Height of frame in analysis steps (not original height):
ANALYSIS_FRAME_HEIGHT = 180

# The minimum orthogonal displacement to be considered an actual wave:
DISPLACEMENT_THRESHOLD = 10         
# The minimum mass to be considered an actual wave:
MASS_THRESHOLD = 1000

# The axis of major waves in the scene, counter clockwise from horizon:
GLOBAL_WAVE_AXIS = 5.0

# Integer global variable seed for naming detected waves by number:
NAME_SEED = 0


class Section():
    """Filtered contours become "sections" with the following
    attributes. Dynamic attributes are updated in each frame through
    tracking routine.
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
                                                    ANALYSIS_FRAME_WIDTH)
        self.boundingbox_coors = np.int0(cv2.boxPoints(
                                            cv2.minAreaRect(points)))
        self.displacement = 0
        self.max_displacement = self.displacement
        self.displacement_vec = deque([self.displacement], 
                                      maxlen = TRACKING_HISTORY)
        self.mass = len(self.points)
        self.max_mass = self.mass
        self.is_wave = False
        self.death = None
        

## ====================================================================


def _get_standard_form_line(point, angle_from_horizon):
    """A function returning a 3-element array corresponding to
    coefficients of the standard form for a line of Ax+By=C.  
    Requires one point in [x,y], and a counterclockwise angle from the
    horizion in degrees.

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
    coefficients[2] = (point[1]
                       - np.tan(np.deg2rad(-angle_from_horizon))
                       * point[0])

    return coefficients


def _get_centroid(points):
    """Function for getting the centroid of an object that is
    represented by positive pixels in a bilevel image.

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
    """ Helper function for returning the four coordinates of a
    polygonal search region- a region in which we would want to merge
    several independent wave objects into one wave object because they
    are indeed one wave.  Creates a buffer based on searchROI_buffer
    and the polygon (wave) axis angle.

    Args:
      centroid: a two-element array representing center of mass of
                a wave
      axis_angle: counterclosewise angle from horizon of a wave's 
                  axis 
      searchROI_buffer: a buffer, in pixels, in which to generate 
                        a search region buffer
      frame_width: the width of the frame, to establish left and 
                   right bounds of a polygon

    Returns:
      polygon_coors: a four element array representing the top left,
                     top right, bottom right, and bottom left
                     coordinates of a search region polygon

    """
    polygon_coors = [[None,None],[None,None],[None,None],[None,None]]

    delta_y_left = np.round(centroid[0] * np.tan(np.deg2rad(axis_angle)))
    delta_y_right = np.round((frame_width - centroid[0])
                             * np.tan(np.deg2rad(axis_angle)))

    upper_left_y = int(centroid[1] + delta_y_left - searchROI_buffer)
    upper_left_x = 0
    upper_right_y = int(centroid[1] - delta_y_right - searchROI_buffer)
    upper_right_x = frame_width

    lower_left_y = int(centroid[1] + delta_y_left + searchROI_buffer)
    lower_left_x = 0
    lower_right_y = int(centroid[1] - delta_y_right + searchROI_buffer)
    lower_right_x = frame_width

    polygon_coors = [[upper_left_x,upper_left_y],
                     [upper_right_x,upper_right_y],
                     [lower_right_x,lower_right_y],
                     [lower_left_x,lower_left_y]]

    return polygon_coors


def _generate_name():
    """Name generator for identifying waves by simple incremental
    numeric sequence.

    Args:
      None

    Returns:
      NAME_SEED: next integer in a sequence seeded by the "NAME_SEED"
                 global variable
    """
    global NAME_SEED
    NAME_SEED += 1

    return NAME_SEED


def update_searchROI_coors(wave):
    """Function that adjusts the search ROI for tracking a wave in
    future Frames.

    Args:
      wave: a wave object

    Returns:
      NONE: updates the wave.searchROI_coors attribute
    """
    wave.searchROI_coors = _get_searchROI_coors(wave.centroid, 
                                                wave.axis_angle, 
                                                SEARCH_REGION_BUFFER, 
                                                ANALYSIS_FRAME_WIDTH)


def update_points(wave, frame):
    """Captures all positive pixels the search ROI based on measurement
    of the wave's position in the previous frame by using a mask.
    
    Docs:
      https://stackoverflow.com/questions/17437846/
      https://stackoverflow.com/questions/10469235/

    Args:
      wave: a Section object
      frame: frame in which to obtain new binary representation of the
             wave

    Returns:
      points: returns all points as an array to the wave.points
              attribute
    """
    points = None
    
    # make a polygon object of the wave's search region
    rect = wave.searchROI_coors
    poly = np.array([rect], dtype=np.int32)

    # make a zero valued image on which to overlay the ROI polygon
    img = np.zeros((ANALYSIS_FRAME_HEIGHT, ANALYSIS_FRAME_WIDTH), np.uint8)
    
    # fill the polygon ROI in the zero-value image with ones
    img = cv2.fillPoly(img, poly, 255)
    
    # bitwise AND with the actual image to obtain a "masked" image
    res = cv2.bitwise_and(frame, frame, mask=img) 
    
    # all points in the ROI are now expressed with ones
    points = cv2.findNonZero(res)
    
    # update points
    wave.points = points


def update_death(wave, frame_number):
    """Checks to see if wave has died, which occurs when no pixels are
    found in the wave's search ROI.  "None" indicates wave is alive,
    while an integer represents the frame number of death.

    Args:
      wave: a wave object
      frame_number: number of frame in a video sequence

    Returns:
      NONE: sets wave death to wave.death attribute
    """
    if wave.points is None:
        wave.death = frame_number


def update_centroid(wave):
    """Calculates the center of mass of all positive pixels that
    represent the wave, using first-order moments.  See _get_centroid.

    Args:
      wave: a wave object

    Returns:
      NONE: updates wave.centroid
    """
    wave.centroid = _get_centroid(wave.points)


def update_boundingbox_coors(wave):
    """Finds minimum area rectangle that bounds the points of a wave.
    Returns four coordinates of the bounding box.  This is primarily
    for visualization purposes.

    Args:
      wave: a wave object

    Returns:
      NONE: updates wave.boundingbox_coors attribute
    """

    boundingbox_coors = None

    if wave.points is not None:
        # Obtain the moments of the object from its points array.
        x = [p[0][0] for p in wave.points]
        y = [p[0][1] for p in wave.points]
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x)
        std_y = np.std(y)

        # We only capture points without outliers for display purposes.
        points_without_outliers = np.array(
            [p[0] for p in wave.points if np.abs(p[0][0]-mean_x) < 3*std_x
                                      and np.abs(p[0][1]-mean_y) < 3*std_y])

        rect = cv2.minAreaRect(points_without_outliers)
        box = cv2.boxPoints(rect)
        boundingbox_coors = np.int0(box)

    wave.boundingbox_coors = boundingbox_coors


def update_displacement(wave):
    """Evaluates orthogonal displacement compared to original axis.
    Updates max_displacement if necessary.

    Args:
      wave: a wave object

    Returns:
      NONE: updates wave.displacement and wave.max_displacement
            attributes
    """
    # Update instantaneous displacement of the wave.
    if wave.centroid is not None:
        # Retrieve standard form coefficients of original axis.
        a = wave.original_axis[0]
        b = wave.original_axis[1]
        c = wave.original_axis[2]

        # Retrieve current location of the wave.
        x0 = wave.centroid[0]
        y0 = wave.centroid[1]

        # Calculate orthogonal distance from current postion to
        # original axis.
        ortho_disp = np.abs(a*x0 + b*y0 + c) / math.sqrt(a**2 + b**2)
        
        wave.displacement = int(ortho_disp)

        # Update max displacement of the wave if necessary.
        if wave.displacement > wave.max_displacement:
            wave.max_displacement = wave.displacement


def update_displacement_vec(wave):
    """Appends displacement to a deque.

    Args:
      wave: a Section object

    Returns:
      NONE: append positional displacement to the queue.
    """
    wave.displacement_vec.append(wave.displacement)


def update_mass(wave):
    """Calculates mass of the wave by weighting each pixel in a search
    ROI equally and performing a simple count.  Updates max_mass
    attribute if necessary.

    Args:
      wave: a Section object

    Returns:
      NONE: updates wave.mass and wave.max_mass attributes
    """
    mass = 0
    
    if wave.points is not None:
        # Update instantaneous mass of the wave.
        mass = len(wave.points)

    wave.mass = mass

    # Update max_mass for the wave if necessary.
    if wave.mass > wave.max_mass:
        wave.max_mass = wave.mass


def update_is_wave(wave):
    """Updates the boolean wave.is_wave to True if wave mass and wave
    displacement exceed user-defined thresholds.  Once a wave is a
    wave, wave is not checked again.

    Args:
      wave: a wave object

    Returns:
      NONE: changes wave.is_wave boolean to True if conditions are met
    """
    if wave.is_wave is False:
        if wave.max_displacement >= DISPLACEMENT_THRESHOLD \
            and wave.max_mass >= MASS_THRESHOLD:
            wave.is_wave = True
