##
##  Near-shore Wave Tracking
##  mwt_tracking.py
##
##  Created by Justin Fung on 9/1/17.
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
        if left_y >= wave.searchroi_coors[0][1] \
           and left_y <= wave.searchroi_coors[3][1]:
            going_to_be_merged = True
            break

    return going_to_be_merged


def track(list_of_waves, frame, frame_number, last_frame, original_frame):
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

        # Update search roi for tracking waves and merging waves.
        wave.update_searchroi_coors()

        # Capture all non-zero points in the new roi.
        wave.update_points(frame)

        # Check if wave has died.
        wave.update_death(frame_number)

        # Kill all waves if it is the last frame in the video.
        if frame_number == last_frame:
            wave.death = frame_number

        # Update centroids.
        wave.update_centroid()

        # Update bounding boxes for display.
        wave.update_boundingbox_coors()

        # Save frame data for rating
        wave.update_frame_data(original_frame)

        # Update displacement vectors.
        wave.update_displacement()

        # Update wave masses.
        wave.update_mass()

        # Check masses and dynamics conditionals.
        wave.update_recognized()
