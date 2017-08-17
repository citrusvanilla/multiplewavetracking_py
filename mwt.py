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

import cv2

import mwt_detection
import mwt_preprocessing
import mwt_tracking
import mwt_io


## ========================================================


def status_update(frame_number, tot_frames):
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
                frame = cv2.drawContours(frame, [rect], 0, (0, 255, 255), 2)
            else:
                frame = cv2.drawContours(frame, [rect], 0, (0, 255, 0), 2)

    return frame


def analyze(video):
    """Main routine for analyzing nearshore wave videos. Overlays
    detected waves onto orginal frames and writes to a new video.
    Returns a log with detected wave attrbutes, frame by frame.

    Args:
      video: mp4 video

    Returns:
      recognized_waves:
      wave_log:
      time_elapsed:
    """
    # Initiate an empty list of tracked waves, ultimately recognized
    # waves, and a log of all tracked waves in each frame.
    tracked_waves = []
    recognized_waves = []
    wave_log = []

    # Initialize frame counters.
    frame_num = 1
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Grab some video stats for videowriter object.
    original_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Initiate video writer object by defining the codec and initiating
    # the VideoWriter object.
    # Make an output directory if necessary.
    if not os.path.exists(mwt_io.OUTPUT_DIR):
        os.mkdir(mwt_io.OUTPUT_DIR)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(mwt_io.OUTPUT_DIR, mwt_io.TRACKED_WAVE_FILE)
    out = cv2.VideoWriter(output_path,
                          fourcc,
                          fps,
                          (int(original_width), int(original_height)),
                          isColor=True)

    # Initiate a timer for program performance:
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
        sections = mwt_detection.detect_sections(analysis_frame,
                                                 frame_num)

        # Track all waves in tracked_waves.
        mwt_tracking.track(tracked_waves,
                           analysis_frame,
                           frame_num,
                           num_frames)

        # Write tracked wave stats to wave_log.
        for wave in tracked_waves:
            wave_log.append((frame_num, wave.name, wave.mass, wave.max_mass,
                             wave.displacement, wave.max_displacement,
                             wave.birth, wave.death, wave.is_wave, wave.centroid))

        # Remove dead waves from tracked_waves.
        for wave in tracked_waves:
            # Remove dead waves.
            if wave.death is not None:
                # If wave became actual wave, add to recognized_waves.
                if wave.is_wave is True:
                    recognized_waves.append(wave)
                # Regardless, remove the wave.
                tracked_waves.remove(wave)

        # Remove duplicate waves, keeping earliest wave.
        tracked_waves.sort(key=lambda x: x.birth, reverse=True)
        for wave in tracked_waves:
            other_waves = [wav for wav in tracked_waves if not wav == wave]
            if mwt_tracking.will_be_merged(wave, other_waves):
                tracked_waves.remove(wave)
        tracked_waves.sort(key=lambda x: x.birth, reverse=False)

        # Check sections for any new potential waves and add to
        # tracked_waves.
        for section in sections:
            if mwt_tracking.will_be_merged(section, tracked_waves):
                continue
            else:
                tracked_waves.append(section)

        # Draw detection boxes on original frame for visualization.
        original_frame = draw(tracked_waves,
                              original_frame,
                              1/mwt_preprocessing.RESIZE_FACTOR)

        # Write frame to output video.
        out.write(original_frame)

        # Increment the frame count.
        frame_num += 1

    # Stop timer here and calc performance.
    time_elapsed = (time.time() - time_start)
    performance = (num_frames / time_elapsed)

    # Provide update to user here.
    if recognized_waves is not None:
        print "{} wave(s) recognized.".format(len(recognized_waves))
        print "Program performance: %0.1f frames per second." %performance
        for i, wave in enumerate(recognized_waves):
            print ("Wave #{}: ID: {}, Birth: {}, Death: {}," \
                   + " Max Displacement: {}, Max Mass: {}").format(
                        i+1, wave.name, wave.birth, wave.death,
                        wave.max_displacement, wave.max_mass)
    else:
        print "No waves recognized."

    # Clean-up resources.
    out.release()

    return recognized_waves, wave_log, performance


def main(argv):
    """main"""
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
    # Keep time for program speed evaluation.
    recognized_waves, wave_log, program_speed = analyze(inputvideo)

    # Write the wave log to csv.
    mwt_io.write_log_to_csv(wave_log)

    # Write the analysis report to txt.
    mwt_io.write_report(recognized_waves, program_speed)

    # Clean-up resources.
    inputvideo.release()


if __name__ == "__main__":
    main(sys.argv[1:])
