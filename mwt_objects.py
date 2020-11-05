##
##  Near-shore Wave Tracking
##  mwt_objects.py
##
##  Created by Justin Fung on 9/1/17.
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
TRACKING_HISTORY = 21

# Width of frame in analysis steps (not original width):
ANALYSIS_FRAME_WIDTH = 320
# Height of frame in analysis steps (not original height):
ANALYSIS_FRAME_HEIGHT = 180

# The minimum orthogonal displacement to be considered an actual wave:
DISPLACEMENT_THRESHOLD = 10
# The minimum mass to be considered an actual wave:
MASS_THRESHOLD = 200

# The axis of major waves in the scene, counter clockwise from horizon:
GLOBAL_WAVE_AXIS = 5.0

# Integer global variable seed for naming detected waves by number:
NAME_SEED = 0


class Section(object):
    """Filtered contours become "sections" with the following
    attributes. Dynamic attributes are updated in each frame through
    tracking routine.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, points, birth):
        self.name = _generate_name()
        self.points = points
        self.birth = birth
        self.axis_angle = GLOBAL_WAVE_AXIS
        self.centroid = _get_centroid(self.points)
        self.centroid_vec = deque([self.centroid],
                                  maxlen=TRACKING_HISTORY)
        self.original_axis = _get_standard_form_line(self.centroid,
                                                     self.axis_angle)
        self.searchroi_coors = _get_searchroi_coors(self.centroid,
                                                    self.axis_angle,
                                                    SEARCH_REGION_BUFFER,
                                                    ANALYSIS_FRAME_WIDTH)
        self.boundingbox_coors = np.int0(cv2.boxPoints(
                                            cv2.minAreaRect(points)))
        self.displacement = 0
        self.max_displacement = self.displacement
        self.displacement_vec = deque([self.displacement],
                                      maxlen=TRACKING_HISTORY)
        self.mass = len(self.points)
        self.max_mass = self.mass
        self.recognized = False
        self.death = None


    def update_searchroi_coors(self):
        """Method that adjusts the search roi for tracking a wave in
        future Frames.

        Args:
          NONE

        Returns:
          NONE: updates self.searhroi_coors
        """
        self.searchroi_coors = _get_searchroi_coors(self.centroid,
                                                    self.axis_angle,
                                                    SEARCH_REGION_BUFFER,
                                                    ANALYSIS_FRAME_WIDTH)


    def update_death(self, frame_number):
        """Checks to see if wave has died, which occurs when no pixels
        are found in the wave's search roi.  "None" indicates wave is
        alive, while an integer represents the frame number of death.

        Args:
          frame_number: number of frame in a video sequence

        Returns:
          NONE: sets wave death to wave.death attribute
        """
        if self.points is None:
            self.death = frame_number


    def update_points(self, frame):
        """Captures all positive pixels the search roi based on
        measurement of the wave's position in the previous frame by
        using a mask.

        Docs:
          https://stackoverflow.com/questions/17437846/
          https://stackoverflow.com/questions/10469235/

        Args:
          frame: frame in which to obtain new binary representation of
                 the wave

        Returns:
          NONE: returns all positive points as an array to the
                self.points attribute
        """
        # make a polygon object of the wave's search region
        rect = self.searchroi_coors
        poly = np.array([rect], dtype=np.int32)

        # make a zero valued image on which to overlay the roi polygon
        img = np.zeros((ANALYSIS_FRAME_HEIGHT, ANALYSIS_FRAME_WIDTH),
                       np.uint8)

        # fill the polygon roi in the zero-value image with ones
        img = cv2.fillPoly(img, poly, 255)

        # bitwise AND with the actual image to obtain a "masked" image
        res = cv2.bitwise_and(frame, frame, mask=img)

        # all points in the roi are now expressed with ones
        points = cv2.findNonZero(res)

        # update points
        self.points = points


    def update_centroid(self):
        """Calculates the center of mass of all positive pixels that
        represent the wave, using first-order moments.
        See _get_centroid.

        Args:
          NONE

        Returns:
          NONE: updates wave.centroid
        """
        self.centroid = _get_centroid(self.points)

        # Update centroid vector.
        self.centroid_vec.append(self.centroid)


    def update_boundingbox_coors(self):
        """Finds minimum area rectangle that bounds the points of the
        wave. Returns four coordinates of the bounding box.  This is
        primarily for visualization purposes.

        Args:
          NONE

        Returns:
          NONE: updates self.boundingbox_coors attribute
        """
        boundingbox_coors = None

        if self.points is not None:
            # Obtain the moments of the object from its points array.
            X = [p[0][0] for p in self.points]
            Y = [p[0][1] for p in self.points]
            mean_x = np.mean(X)
            mean_y = np.mean(Y)
            std_x = np.std(X)
            std_y = np.std(Y)

            # We only capture points without outliers for display
            # purposes.
            points_without_outliers = np.array(
                                       [p[0] for p in self.points
                                        if np.abs(p[0][0]-mean_x) < 3*std_x
                                        and np.abs(p[0][1]-mean_y) < 3*std_y])

            rect = cv2.minAreaRect(points_without_outliers)
            box = cv2.boxPoints(rect)
            boundingbox_coors = np.int0(box)

        self.boundingbox_coors = boundingbox_coors


    def update_displacement(self):
        """Evaluates orthogonal displacement compared to original axis.
        Updates self.max_displacement if necessary.  Appends new
        displacement to deque.

        Args:
          NONE

        Returns:
          NONE: updates self.displacement and self.max_displacement
                attributes
        """
        if self.centroid is not None:
            self.displacement = _get_orthogonal_displacement(
                                                        self.centroid,
                                                        self.original_axis)

        # Update max displacement of the wave if necessary.
        if self.displacement > self.max_displacement:
            self.max_displacement = self.displacement

        # Update displacement vector.
        self.displacement_vec.append(self.displacement)


    def update_mass(self):
        """Calculates mass of the wave by weighting each pixel in a
        search roi equally and performing a simple count.  Updates
        self.max_mass attribute if necessary.

        Args:
          wave: a Section object

        Returns:
          NONE: updates self.mass and self.max_mass attributes
        """
        self.mass = _get_mass(self.points)

        # Update max_mass for the wave if necessary.
        if self.mass > self.max_mass:
            self.max_mass = self.mass


    def update_recognized(self):
        """Updates the boolean self.recognized to True if wave mass and
        wave displacement exceed user-defined thresholds.  Once a wave
        is recognized, the wave is not checked again.

        Args:
          wave: a wave object

        Returns:
          NONE: changes self.recognized boolean to True if conditions
                are met.
        """
        if self.recognized is False:
            if self.max_displacement >= DISPLACEMENT_THRESHOLD \
               and self.max_mass >= MASS_THRESHOLD:
                self.recognized = True


## ====================================================================


def _get_mass(points):
    """Simple function to calculate mass of an array of points with
    equal weighting of the points.

    Args:
      points: an array of non-zero points

    Returns:
      mass:  "mass" of the points
    """
    mass = 0

    if points is not None:
        mass = len(points)

    return mass


def _get_orthogonal_displacement(point, standard_form_line):
    """Helper function to calculate the orthogonal distance of a point
    to a line.

    Args:
      point: 2-element array representing a point as [x,y]
      standard_form_line: 3-element array representing a line in
                          standard form coordinates as [A,B,C]
    Returns:
      ortho_disp: distance of point to line in pixels
    """
    ortho_disp = 0

    # Retrieve standard form coefficients of original axis.
    a = standard_form_line[0]
    b = standard_form_line[1]
    c = standard_form_line[2]

    # Retrieve current location of the wave.
    x0 = point[0]
    y0 = point[1]

    # Calculate orthogonal distance from current postion to
    # original axis.
    ortho_disp = np.abs(a*x0 + b*y0 + c) / math.sqrt(a**2 + b**2)

    return int(ortho_disp)


def _get_standard_form_line(point, angle):
    """Helper function returning a 3-element array corresponding to
    coefficients of the standard form for a line of Ax+By=C.
    Requires one point in [x,y], and a counterclockwise angle from the
    horizion in degrees.

    Args:
      point: a two-element array in [x,y] representing a point
      angle: a float representing counterclockwise angle from horizon
             of a line

    Returns:
      coefficients: a three-element array as [A,B,C]
    """
    coefficients = [None, None, None]

    coefficients[0] = np.tan(np.deg2rad(-angle))
    coefficients[1] = -1
    coefficients[2] = (point[1] - np.tan(np.deg2rad(-angle))*point[0])

    return coefficients


def _get_centroid(points):
    """Helper function for getting the x,y coordinates of the center of
    mass of an object that is represented by positive pixels in a
    bilevel image.

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


def _get_searchroi_coors(centroid, angle, searchroi_buffer, frame_width):
    """Helper function for returning the four coordinates of a
    polygonal search region- a region in which we would want to merge
    several independent wave objects into one wave object because they
    are indeed one wave.  Creates a buffer based on searchroi_buffer
    and the polygon (wave) axis angle.

    Args:
      centroid: a two-element array representing center of mass of
                a wave
      angle: counterclosewise angle from horizon of a wave's axis
      searchroi_buffer: a buffer, in pixels, in which to generate
                        a search region buffer
      frame_width: the width of the frame, to establish left and
                   right bounds of a polygon

    Returns:
      polygon_coors: a four element array representing the top left,
                     top right, bottom right, and bottom left
                     coordinates of a search region polygon

    """
    polygon_coors = [[None, None],
                     [None, None],
                     [None, None],
                     [None, None]]

    delta_y_left = np.round(centroid[0] * np.tan(np.deg2rad(angle)))
    delta_y_right = np.round((frame_width - centroid[0])
                             * np.tan(np.deg2rad(angle)))

    upper_left_y = int(centroid[1] + delta_y_left - searchroi_buffer)
    upper_left_x = 0
    upper_right_y = int(centroid[1] - delta_y_right - searchroi_buffer)
    upper_right_x = frame_width

    lower_left_y = int(centroid[1] + delta_y_left + searchroi_buffer)
    lower_left_x = 0
    lower_right_y = int(centroid[1] - delta_y_right + searchroi_buffer)
    lower_right_x = frame_width

    polygon_coors = [[upper_left_x, upper_left_y],
                     [upper_right_x, upper_right_y],
                     [lower_right_x, lower_right_y],
                     [lower_left_x, lower_left_y]]

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
