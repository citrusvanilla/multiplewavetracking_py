##
##  Near-shore Wave Tracking
##  mwt.py
##
##  Created by Justin Fung on 8/1/17.
##  Copyright 2017 justin fung. All rights reserved.
##
## ========================================================

"""A module for recognition and tracking of multiple nearshore waves
from input videos.

Performance:
mwt.py achieves realtime inference in the presence of multiple tracked
objects for input videos of 1280x720 that are downscaled by a factor of
four at runtime on consumer hardware.

System                       | Step Time (sec/frame)  | Performance
-----------------------------------------------------------------------
1 CPU 2.6 GHz Intel Core i5  | 0.015 - 0.030          | 30Hz - 60Hz

Usage:
Please see the README for how to compile the program and run the model.
"""
from __future__ import division

import sys
import os
import getopt
import time
import csv

import numpy as np
import cv2

import mwt_detection
import mwt_preprocessing
import mwt_tracking

# We are going to dump our output in a folder in the same directory.
if not os.path.exists("output"):
    os.mkdir("output")

WAVE_LOG_FILE = "output/wave_log.csv"
RECOGNIZED_WAVE_REPORT_FILE = "output/recognized_waves.txt"
TRACKED_WAVE_FILE = "output/tracked_waves.mp4"


## ========================================================


def status_update(frame_number,tot_frames):
    """A simple inline status update for stdout.
    Prints frame number for every 100 frames completed.

    Args:
      frame_number: number of frames completed
      tot_frames: total number of frames to analyze

    Returns:
      VOID: writes status to stdout
    """
    if frame_number == 1:
        sys.stdout.write("Starting analysis of %d frames...\n" %tot_frames)
        sys.stdout.flush()
    
    if frame_number % 100 == 0:
        sys.stdout.write("%d" %frame_number)
        sys.stdout.flush()
    elif frame_number % 10 == 0:
        sys.stdout.write(".")
        sys.stdout.flush()


def draw(waves, frame, resize_factor):
    """Simple function to draw bounding boxes on a frame for output.

    Args:
      waves: list of waves
      frame: frame on which to draw waves
      resize_factor: factor to resize boundingbox coors to match output
                     frame.

    Returns:
      frame: input frame with waves drawn on top 
    """
    # Draw detection boxes on original frame and write out.
    for wave in waves:
        
        if wave.death is None:
            # Get boundingbox coors from wave objects.
            rect = wave.boundingbox_coors

            # Resize (upsize) for output.
            rect[:] = [resize_factor*rect[i] for i in range(4)]
            
            # If wave is not yet a wave, draw yellow, else green.
            if wave.is_wave is False:
                frame = cv2.drawContours(frame,[rect],0,(0,255,255),2)
            else:
                frame = cv2.drawContours(frame,[rect],0,(0,255,0),2)

    return frame


def analyze(video, Recognized_Waves, log):
    """Main routine for analyzing nearshore wave videos. Overlays
    detected waves onto orginal frames and writes to a new video.
    Modifies passed log with detected wave attrbutes, frame by frame.

    Args:
      video: mp4 video 
      Recognized_Waves: empty list to contain all recognized waves
      log: empty list to append wave attributes, for CSV-ready
           formatting

    Returns:
      VOID: none
    """
    # Initiate an empty list of potential waves.
    Tracked_Waves = []

    # Initialize frame counters.
    frame_num = 1
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Grab some video stats for videowriter object.
    original_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Initiate video writer object by defining the codec and initiating
    # the VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TRACKED_WAVE_FILE, 
                          fourcc,
                          fps, 
                          (int(original_width),int(original_height)), 
                          isColor=True)

    # Begin timer for performance measurement.
    time_start = time.time()

    # The main loop is here:
    while True:

        # Write status update to stdio.
        status_update(frame_num, num_frames)

        # Read frames until end of clip.
        successful_read, original_frame = video.read()
        if not successful_read:
            if frame_num < num_frames:
                print "Did not reach end of video successfully."
            else:
                print "End of video reached."
            break

        # Preprocess frames.
        analysis_frame = mwt_preprocessing.preprocess(original_frame)

        # Detect all sections.
        Sections = mwt_detection.detect_sections(analysis_frame,
                                                 frame_num)

        # Track all waves in Tracked_Waves.
        mwt_tracking.track(Tracked_Waves,
                           analysis_frame,
                           frame_num,
                           num_frames)

        # Write tracked wave stats to wave_log.
        for wave in Tracked_Waves:
            log.append((frame_num, wave.name, wave.mass, wave.max_mass,
                        wave.displacement, wave.max_displacement,
                        wave.birth, wave.death, wave.is_wave, wave.centroid))

        # Remove dead waves from Tracked_Waves.
        for wave in Tracked_Waves:
            # Remove dead waves.
            if wave.death is not None:
                # If wave became actual wave, add to recognized_waves.
                if wave.is_wave is True:
                    Recognized_Waves.append(wave)
                Tracked_Waves.remove(wave)

        # Remove duplicate waves, keeping earliest wave.
        Tracked_Waves.sort(key=lambda x: x.birth, reverse=True)
        for wave in Tracked_Waves:
            other_waves = [wav for wav in Tracked_Waves if not wav == wave]
            if mwt_tracking.will_be_merged(wave, other_waves):
                Tracked_Waves.remove(wave)
        Tracked_Waves.sort(key=lambda x: x.birth, reverse=False)

        # Check sections for any new potential waves and add to
        # Tracked_Waves.
        for section in Sections:
            if mwt_tracking.will_be_merged(section, Tracked_Waves):
                continue
            else:
                Tracked_Waves.append(section)
        
        # Draw detection boxes on original frame for visualization.
        original_frame = draw(Tracked_Waves, 
                              original_frame, 
                              1/mwt_preprocessing.RESIZE_FACTOR)
    
        # Write frame to output video.
        out.write(original_frame)

        # Increment the frame count.
        frame_num += 1

    # Stop timer and print.
    time_elapsed = (time.time() - time_start)
    print "Program performance: %0.1f frames per second." \
                   % (num_frames / time_elapsed)

    # Clean-up resources.
    out.release()
    

def main(argv):
    # The command line should have one argument- 
    # the name of the videofile.
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "i:")
    except getopt.GetoptError:
        print "usage: mwt.py -i <inputfile>"
        sys.exit(2)
    for opt, arg in opts:
        if opt == ("-i"):
            inputfile = arg

    # Read video.
    print "Checking video from", inputfile
    inputvideo = cv2.VideoCapture(inputfile)

    # Exit if video cannot be opened.
    if not inputvideo.isOpened():
        sys.exit("Could not open video.")

    # Initialize a wave_log list, recognized_wave list, and analyze.
    wave_log = []
    recognized_waves = []
    print "Writing to tracked_waves_output.mp4"
    analyze(inputvideo, recognized_waves, wave_log)

    # Write log to CSV.
    wave_log_headers = ["frame_num", "wave_id", "inst_mass", "max_mass",
                        "inst_displacement", "max_displacement",
                        "frame_birth", "frame_death", "is_wave", "centroid"]
    with open(WAVE_LOG_FILE, "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(wave_log_headers)
        for row in wave_log:
            writer.writerow(row)
    
    # Write recognized waves to text file.
    if len(recognized_waves) > 0:
        print "{} wave(s) recognized.  See {} and {} for details.".format(
                len(recognized_waves), 
                RECOGNIZED_WAVE_REPORT_FILE,
                WAVE_LOG_FILE)
        
        for i in range(len(recognized_waves)):
            print ("Wave #{}: ID: {}, Birth: {}, Death: {}," \
                   + " Max Displacement: {}, Max Mass: {}").format(
                        i+1,
                        recognized_waves[i].name,
                        recognized_waves[i].birth,
                        recognized_waves[i].death,
                        recognized_waves[i].max_displacement,
                        recognized_waves[i].max_mass)

        with open(RECOGNIZED_WAVE_REPORT_FILE, "w") as text_file:
            for i in range(len(recognized_waves)):
                text_file.write("Wave #{}: ID: {}, Birth: {}, Death: {}, "
                                    .format(i+1,
                                            recognized_waves[i].name,
                                            recognized_waves[i].birth,
                                            recognized_waves[i].death))
                text_file.write("Max Displacement: {}, Max Mass: {}\n"
                                    .format(
                                    recognized_waves[i].max_displacement,
                                    recognized_waves[i].max_mass))
    else:
        print "No waves recognized. Check video or adjust threshold settings."

    # Clean-up resources.
    inputvideo.release()


if __name__ == "__main__":
    main(sys.argv[1:])