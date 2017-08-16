##
##  Near-shore Wave Tracking
##  mwt.py
##
##  Created by Justin Fung on 8/1/17.
##  Copyright 2017 justin fung. All rights reserved.
##
## ========================================================

"""
A module for recognition and tracking of multiple nearshore waves from
input videos.

Performance:
mwt.py achieves realtime inference in the presence of multiple tracked
objects for input videos of 1280x720 that are downscaled by a factor of
four at runtime on consumer hardware.

System                       | Step Time (sec/frame)  | Performance
-----------------------------------------------------------------------
1 CPU 2.6 GHz Intel Core i5  | 0.02 - 0.03            | 30Hz - 45Hz

Usage:
Please see the README for how to compile the program and run the model.
"""
from __future__ import division

import numpy as np
import cv2
import sys
import os
import getopt
import time
import csv

import mwt_detection
import mwt_preprocessing
import mwt_tracking

scene_dir = "/Users/justinfung/Desktop/udacity/mwt/private/scenes/test"
os.chdir(scene_dir)

WAVE_LOG_FILE = "wave_log.csv"
DETECTED_WAVE_REPORT_FILE = "detected_waves.txt"
TRACKED_WAVE_FILE = "tracked_waves.mp4"


## ========================================================

def status_update(frame_number,tot_frames):
    """
    A simple inline status update for stdout.
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

    if frame_number == tot_frames:
        sys.stdout.write("\nAnalysis of %d frames completed.\n" %tot_frames)
        sys.stdout.flush()


def draw(waves, frame, resize_factor):
    """
    simple function to draw bounding boxes on a frame for output.

    Args:
      waves: list of waves
      frame: frame on which to draw waves
      resize_factor: factor to resize boundingbox coors to match output frame.

    Returns:
      frame: input frame with waves drawn on top 
    """
    # draw detection boxes on original frame and write out
    for wave in waves:
        
        if wave.death is None:
            # get boundingbox coors from wave objects
            rect = wave.boundingbox_coors

            # resize for output
            rect[:] = [resize_factor*rect[i] for i in range(4)]
            
            # if wave is not yet a wave, draw yellow, else green
            if wave.is_wave == False:
                frame = cv2.drawContours(frame,[rect],0,(0,255,255),2) # yellow
            else:
                frame = cv2.drawContours(frame,[rect],0,(0,255,0),2) # green

    return frame


def analyze(video, log):
    """
    Main routine for analyzing nearshore wave videos.
    Overlays detected waves onto orginal frames and writes to a new video.
    Modifies passed log with detected wave attrbutes, frame by frame.

    Args:
      video: mp4 vid
      log: empty list to append wave attributes, for CSV-ready formatting.

    Returns:
      VOID:
    """
    # initiate empty list of potential waves
    Tracked_Waves = []
    Dead_Waves = []

    # grab some video stats and initialize a global counter
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_num = 1
    
    # initiate video writer object
    # Define the codec and create VideoWriter object
    # from: http://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TRACKED_WAVE_FILE, fourcc, fps, 
                          (int(original_width),int(original_height)), 
                          isColor = True)

    # begin timer for performance measurement
    time_start = time.time()

    # main loop
    while True:

        # write status update to stdio
        status_update(frame_num, num_frames)

        # Read frames until end of clip
        successful_read, original_frame = video.read()
        
        if not successful_read:
            break

        # preprocess frames
        analysis_frame = mwt_preprocessing.preprocess(original_frame)

        # detect all sections
        Sections = mwt_detection.detect_sections(analysis_frame, frame_num)

        # track all waves in Tracked Waves
        mwt_tracking.track(Tracked_Waves, analysis_frame, frame_num)

        # do some tidying up of duplicate and dead waves
        for wave in Tracked_Waves:
            
            # Remove dead waves.
            if wave.death is not None:

                # if wave became actual wave, add to Dead_Waves.
                if wave.is_wave is True:
                    Dead_Waves.append(wave)
                
                Tracked_Waves.remove(wave)
                break

            # Remove Waves that have become merged.
            other_waves = [wav for wav in Tracked_Waves if not wav == wave]
            if mwt_tracking.will_be_merged(wave, other_waves):
                Tracked_Waves.remove(wave)

            # write tracked wave stats to tracked wave log
            log.append((frame_num, wave.name, wave.mass, wave.displacement, 
                        wave.birth, wave.death, wave.is_wave, wave.centroid))


        # check sections for any new potential waves and add to Tracked Waves
        for section in Sections:
            if mwt_tracking.will_be_merged(section, Tracked_Waves):
                continue
            else:
                Tracked_Waves.append(section)
        
        # draw detect boxes on original frame for visualization and write out
        original_frame = draw(Tracked_Waves, 
                              original_frame, 
                              1/mwt_preprocessing.RESIZE_FACTOR)
    
        # write frame to video
        out.write(original_frame)

        # increment frame count
        frame_num += 1

    # stop timer
    time_elapsed = (time.time() - time_start)
    print "Program performance: %0.1f frames per second." % (num_frames / time_elapsed)

    # clean-up
    out.release()

    # debug
    for wave in Dead_Waves:
        print wave.name
    

def main(argv):
    # command line should have one argument for the name of the videofile
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "i:")
    except getopt.GetoptError:
        print 'usage: mwt.py -i <inputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == ("-i"):
            inputfile = arg

    # Read video
    print "Checking video from", inputfile
    inputvideo = cv2.VideoCapture(inputfile)

    # Exit if video cannot be opened.
    if not inputvideo.isOpened():
        sys.exit("Could not open video.")

    # Exit if frames cannot be read.
    successful_read, _ = inputvideo.read()
    if not successful_read:
        sys.exit('Cannot read video file.')

    # initialize a wavelog list, analyze and write out
    wave_log = []
    print "Writing to tracked_waves_output.mp4"
    log = analyze(inputvideo, wave_log)

    # write log to CSV
    wave_log_headers = ["frame_num", "wave_id", "wave_mass", "displacement", 
                        "frame_birth", "frame_death", "is_wave", "centroid"]
    with open("output_test.csv", "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(wave_log_headers)
        for row in wave_log:
            writer.writerow(row)
    
    # clean up
    inputvideo.release()


if __name__ == "__main__":
    main(sys.argv[1:])