##
##  Near-shore Wave Tracking
##  mwt_io.py
##
##  Created by Justin Fung on 8/1/17.
##  Copyright 2017 justin fung. All rights reserved.
##
## ========================================================

"""A module for handling input and output of near-shore wave tracking
analysis.
"""
from __future__ import division

import os
import csv

# We are going to dump our output in a folder in the same directory.
OUTPUT_DIR = "output"
WAVE_LOG_FILE = "wave_log.csv"
RECOGNIZED_WAVE_REPORT_FILE = "recognized_waves.txt"
TRACKED_WAVE_FILE = "tracked_waves.mp4"


## ========================================================


def write_log_to_csv(log):
    """Takes a list of list of wave attributes representing each
    tracked wave's statistics in every frame of the video for which the
    wave was present.  Outputs each item in the list as a line to a
    CSV.

    Args:
      log: a list of lists of wave statistics

    Returns:
      NONE: writes the log to a csv file.
    """
    # Print status update.
    if not log:
        print "No waves or sections detected.  No log written."
    else:
        print "Writing wave log to", WAVE_LOG_FILE

    # Declare headers here.
    log_headers = ["frame_num", "wave_id", "inst_mass", "max_mass",
                   "inst_displacement", "max_displacement",
                   "frame_birth", "frame_death", "is_wave", "centroid"]

    # Make an output directory if necessary.
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Using a context manager, write each item to a new row in the csv.
    output_path = os.path.join(OUTPUT_DIR, WAVE_LOG_FILE)
    with open(output_path, "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(log_headers)
        for row in log:
            writer.writerow(row)


def write_report(waves, performance):
    """Takes a list of recognized wave objects and writes attributes
    out to a plain text file.  Supplements this information with
    program performance and user stats.

    Args:
      waves: a list of wave objects
      performance_metric: a double representing speed of program

    Returns:
      NONE: writes the report to a txt file.
    """
    # Provide User feedback here.
    if not waves:
        print "No waves found.  No report written."
    else:
        print "Writing analysis report to", RECOGNIZED_WAVE_REPORT_FILE

    # Make an output directory if necessary.
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Write recognized waves to text file.
    report_path = os.path.join(OUTPUT_DIR, RECOGNIZED_WAVE_REPORT_FILE)

    # Use the context manager here:
    with open(report_path, "w") as text_file:
        text_file.write("Program performance: {} frames per second."
                        .format(performance))
        for i, wave in enumerate(waves):
            text_file.write("Wave #{}: ID: {}, Birth: {}, Death: {}, "
                            .format(i+1, wave.name, wave.birth, wave.death))
            text_file.write("Max Displacement: {}, Max Mass: {}\n"
                            .format(wave.max_displacement, wave.max_mass))
