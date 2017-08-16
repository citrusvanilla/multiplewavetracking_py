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

import math

import cv2
import numpy as np

import mwt_objects


## ====================================================================


def will_be_merged(section, list_of_waves):
    """Boolean evaluating whether or not a section is in an existing
    wave's search region.

    Args:
      section: a wave object
      list_of_waves: a list of waves having search regions in which a
                     wave might fall

    Returns:
      going_to_be_merged: evaluates to True if the section is in an
                          existing wave's search region.
    """
    # All sections are initially new waves & will not be merged.
    going_to_be_merged = False

    # Find the section's major axis' projection on the y axis.
    delta_y_left = np.round(section.centroid[0]
                            * np.tan(np.deg2rad(section.axis_angle)))
    left_y = int(section.centroid[1] + delta_y_left)
    
    # For each existing wave, see if the section's axis falls in
    # another wave's search region.
    for wave in list_of_waves:
        if left_y >= wave.searchROI_coors[0][1] \
           and left_y <= wave.searchROI_coors[3][1]:
            going_to_be_merged = True
            break

    return going_to_be_merged


def track(list_of_waves, frame, frame_number, last_frame):
    """Tracking routine performed by updating the dynamic Wave
    attributes.

    Args:
      list_of_waves: a list of waves to track
      frame: a frame from a cv2.video_reader object
      frame_number: number of the frame in a sequence
      last_frame: final frame number, provided to kill all waves if
                  necessary

    Returns:
      NONE: updates wave attributes
    """
    for wave in list_of_waves:

        # Update search ROI for tracking waves and merging waves.
        mwt_objects.update_searchROI_coors(wave)

        # Capture all non-zero points in the new ROI.
        mwt_objects.update_points(wave, frame)
        
        # Check if wave has died.
        mwt_objects.update_death(wave, frame_number)

        # Kill all waves if it is the last frame in the video.
        if frame_number == last_frame:
            wave.death = frame_number

        # Update centroids.
        mwt_objects.update_centroid(wave)

        # Update bounding boxes for display.
        mwt_objects.update_boundingbox_coors(wave)

        # Update displacement vectors.
        mwt_objects.update_displacement_vec(wave)

        # Update absolute displacements.
        mwt_objects.update_displacement(wave)

        # Update wave masses.
        mwt_objects.update_mass(wave)

        # Check masses and dynamics conditionals.
        mwt_objects.update_is_wave(wave)

