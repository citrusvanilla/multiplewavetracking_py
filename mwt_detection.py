"""Routine for detecting potential waves.

Method for detecting is:
    1. detect contours
    2. filter contours
    3. create list of wave objects from filtered contours
"""

from __future__ import division

import math

import cv2

from mwt_objects import Section

# Boolean flag to filter blobs by area:
FLAGS_FILTER_BY_AREA = True
# Boolean flag to filter blobs by inertia (shape):
FLAGS_FILTER_BY_INERTIA = True

# Minimum area threshold for contour:
MINIMUM_AREA = 100
# Maximum area threshold for contour:
MAXIMUM_AREA = 1e7
# Minimum inertia threshold for contour:
MINIMUM_INERTIA_RATIO = 0.0
# Maximum inertia threshold for contour:
MAXIMUM_INERTIA_RATIO = 0.1


def find_contours(frame):
    """Contour finding function utilizing OpenCV.

    Args:
      frame: A frame from a cv2.video_reader object to process.

    Returns:
      contours: An array of contours, each represented by an array of
                points.
    """
    contours, hierarchy = cv2.findContours(
        image=frame,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
        hierarchy=None,
        offset=None,
    )

    return contours


def keep_contour(
    contour,
    area=FLAGS_FILTER_BY_AREA,
    inertia=FLAGS_FILTER_BY_INERTIA,
    min_area=MINIMUM_AREA,
    max_area=MAXIMUM_AREA,
    min_inertia_ratio=MINIMUM_INERTIA_RATIO,
    max_inertia_ratio=MAXIMUM_INERTIA_RATIO,
):
    """Return whether or not to keep a potential wave shape.

    Contour filtering function utilizing OpenCV.  In our case,
    we are looking for oblong shapes that exceed a user-defined area.

    Args:
        contour: A contour from an array of contours
        area: boolean flag to filter contour by area
        inertia: boolean flag to filter contour by inertia
        min_area: minimum area threshold for contour
        max_area: maximum area threshold for contour
        min_inertia_ratio: minimum inertia threshold for contour
        max_inertia_ratio: maximum inertia threshold for contour

    Returns:
        ret: A boolean TRUE if contour meets conditions, else FALSE
    """
    # Initialize the return value.
    ret = True

    # Obtain contour moments.
    moments = cv2.moments(contour)

    # Filter Contours By Area.
    if area is True and ret is True:
        area = cv2.contourArea(contour)
        if area < min_area or area >= max_area:
            ret = False

    # Filter contours by inertia.
    if inertia is True and ret is True:
        denominator = math.sqrt(
            (2 * moments["m11"]) ** 2 + (moments["m20"] - moments["m02"]) ** 2
        )
        epsilon = 0.01
        ratio = 0.0

        if denominator > epsilon:
            cosmin = (moments["m20"] - moments["m02"]) / denominator
            sinmin = 2 * moments["m11"] / denominator
            cosmax = -cosmin
            sinmax = -sinmin

            imin = (
                0.5 * (moments["m20"] + moments["m02"])
                - 0.5 * (moments["m20"] - moments["m02"]) * cosmin
                - moments["m11"] * sinmin
            )
            imax = (
                0.5 * (moments["m20"] + moments["m02"])
                - 0.5 * (moments["m20"] - moments["m02"]) * cosmax
                - moments["m11"] * sinmax
            )
            ratio = imin / imax
        else:
            ratio = 1

        if ratio < min_inertia_ratio or ratio >= max_inertia_ratio:
            ret = False
            # center.confidence = ratio * ratio;

    return ret


def detect_sections(frame, frame_number):
    """Find sections that meet the user-defined criteria.

    Args:
        frame: a frame from a cv2.video_reader object
        frame_number: number of the frame in the video sequence

    Returns:
        sections: a list of Section objects
    """
    # Convert to single channel for blob detection if necessary.
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initiate and empty list of sections.
    sections = []

    # 1. Find the contours.
    contours = find_contours(frame)

    # 2. Filter the contours.
    for contour in contours:

        if keep_contour(contour) is False:
            continue

        # If contour passes thresholds, convert it to a Section.
        section = Section(points=contour, birth=frame_number)

        # 3. Add the section to sections list.
        sections.append(section)

    return sections
