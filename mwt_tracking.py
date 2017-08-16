##
##  Near-shore Wave Tracking
##  mwt_tracking.py
##
##  Created by Justin Fung on 8/1/17.
##  Copyright 2017 justin fung. All rights reserved.
##
## ====================================================================

"""Functions for using tracking to recognize actual waves.

   Method for recognition is:
   -1. merge potential waves
   -2. track potential waves
   -3. declare actual waves conditional on wave dynamics
   """

from __future__ import division

import cv2
import math
import numpy as np

import mwt_objects


## ====================================================================

def will_be_merged(section, list_of_waves):
    """Boolean evaluating whether or not a section is in an
       existing wave's search region.

    Args:
      section: a Section (wave) object
      list_of_waves: a list of waves in which a wave might fall

    Returns:
      going_to_be_merged: evaluates to True if the section is 
                          in an existing wave's search region.
    """
    # all sections are initially potential new waves & will not be merged
    going_to_be_merged = False

    # find the section's major axis' project on the y axis
    delta_y_left = np.round(section.centroid[0] \
                            * np.tan(np.deg2rad(section.axis_angle)))
    left_y = int(section.centroid[1] + delta_y_left)
    
    # for each existing wave, 
    # see if the section's axis falls in another wave's search region
    for wave in list_of_waves:

        # check if the section projection on y-axis 
        # falls between the search region projections
        if left_y >= wave.searchROI_coors[0][1] \
           and left_y <= wave.searchROI_coors[3][1]:
            going_to_be_merged = True
            break

    return going_to_be_merged


def track(list_of_waves, frame, frame_number):
    """tracking routine performed by updating Wave attributes.

    Args:
      list_of_waves: a list of waves to track
      frame: a frame from a cv2.video_reader object
      frame_number: number of the frame in a sequence

    Returns:
      NONE: updates Section attributes
    """
    for wave in list_of_waves:

        # update search ROI for tracking waves and merging waves
        mwt_objects.update_searchROI_coors(wave)

        # capture all non-zero points in the new ROI
        mwt_objects.update_points(wave, frame)
        
        # check if wave has died
        mwt_objects.update_death(wave, frame_number)

        # update centroid
        mwt_objects.update_centroid(wave)

        # update bounding box for display
        mwt_objects.update_boundingbox_coors(wave)

        # update displacement vector
        mwt_objects.update_displacement_vec(wave)

        # update absolute displacement
        mwt_objects.update_displacement(wave)

        # update wave mass
        mwt_objects.update_mass(wave)

        # check mass and dynamics conditionals
        mwt_objects.update_is_wave(wave)

