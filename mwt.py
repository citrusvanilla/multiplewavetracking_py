"""Near-shore Wave Tracking module.

A module for recognition and tracking of multiple nearshore waves
from input videos.

Performance:
mwt.py achieves realtime inference in the presence of multiple tracked
objects for input videos of 1280x720 that are downscaled by a factor of
four at runtime on consumer hardware.

System                       | Step Time (sec/frame)  | Performance
--------------------------------------------------------------------
1 CPU 2.6 GHz Intel Core i5  | 0.015 - 0.030          | 30Hz - 60Hz

Usage:
    Please see the README for how to compile the program and run the model.

Created by Justin Fung on 9/1/17.

Copyright 2017 justin fung. All rights reserved.
"""
from __future__ import division

import argparse
import sys
import time

import cv2

import mwt_detection
import mwt_preprocessing
import mwt_tracking
import mwt_io


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    "video_path", metavar="path", type=str, help="the path to the input video"
)


def status_update(frame_number, tot_frames):
    """Update status to stdout.

    A simple inline status update for stdout.

    Prints frame number for every 100 frames completed.

    Args:
        frame_number: number of frames completed
        tot_frames: total number of frames to analyze
    """
    if frame_number == 1:
        sys.stdout.write("Starting analysis of %d frames...\n" % tot_frames)
        sys.stdout.flush()

    if frame_number % 100 == 0:
        sys.stdout.write("%d" % frame_number)
        sys.stdout.flush()
    elif frame_number % 10 == 0:
        sys.stdout.write(".")
        sys.stdout.flush()

    if frame_number == tot_frames:
        print("End of video reached successfully.")

    return


def analyze(video, write_output=True):
    """Analyze the video.

    Main routine for analyzing nearshore wave videos. Overlays detected waves
    onto orginal frames and writes to a new video.  Returns a log with
    detected wave attrbutes, frame by frame.

    Args:
        video: mp4 video
        write_output: boolean indicating if a video with tracking overlay
                      is to be written out.

    Returns:
        recognized_waves: list of recognized wave objects
        wave_log: list of list of wave attributes for csv
        time_elapsed: performance of the program in frames/second
    """
    # Initiate an empty list of tracked waves, ultimately recognized
    # waves, and a log of all tracked waves in each frame.
    tracked_waves = []
    recognized_waves = []
    wave_log = []

    # Initialize frame counters.
    frame_num = 1
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # If an output video is to be made:
    if write_output is True:
        out = mwt_io.create_video_writer(video)

    # Initiate a timer for program performance:
    time_start = time.time()

    # The main loop is here:
    while True:

        # Write status update to stdio.
        status_update(frame_num, num_frames)

        # Read frames until end of clip.
        successful_read, original_frame = video.read()
        if not successful_read:
            break

        # Preprocess frames.
        analysis_frame = mwt_preprocessing.preprocess(original_frame)

        # Detect all sections.
        sections = mwt_detection.detect_sections(analysis_frame, frame_num)

        # Track all waves in tracked_waves.
        mwt_tracking.track(
            tracked_waves, analysis_frame, frame_num, num_frames
        )

        # Write tracked wave stats to wave_log.
        for wave in tracked_waves:
            wave_log.append(
                (
                    frame_num,
                    wave.name,
                    wave.mass,
                    wave.max_mass,
                    wave.displacement,
                    wave.max_displacement,
                    wave.birth,
                    wave.death,
                    wave.recognized,
                    wave.centroid,
                )
            )

        # Remove dead waves from tracked_waves.
        dead_recognized_waves = [
            wave
            for wave in tracked_waves
            if wave.death is not None and wave.recognized is True
        ]
        recognized_waves.extend(dead_recognized_waves)

        tracked_waves = [wave for wave in tracked_waves if wave.death is None]

        # Remove duplicate waves, keeping earliest wave.
        tracked_waves.sort(key=lambda x: x.birth, reverse=True)
        for wave in tracked_waves:
            other_waves = [wav for wav in tracked_waves if not wav == wave]
            if mwt_tracking.will_be_merged(wave, other_waves):
                wave.death = frame_num
        tracked_waves = [wave for wave in tracked_waves if wave.death is None]
        tracked_waves.sort(key=lambda x: x.birth, reverse=False)

        # Check sections for any new potential waves and add to
        # tracked_waves.
        for section in sections:
            if not mwt_tracking.will_be_merged(section, tracked_waves):
                tracked_waves.append(section)

        # analysis_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_GRAY2RGB)

        if write_output is True:
            # Draw detection boxes on original frame for visualization.
            original_frame = mwt_io.draw(
                tracked_waves,
                original_frame,
                # 1)
                1 / mwt_preprocessing.RESIZE_FACTOR,
            )

            # Write frame to output video.
            # out.write(analysis_frame)
            out.write(original_frame)

        # Increment the frame count.
        frame_num += 1

    # Stop timer here and calc performance.
    time_elapsed = time.time() - time_start
    performance = num_frames / time_elapsed

    # Provide update to user here.
    if recognized_waves is not None:
        print(f"{len(recognized_waves)} wave(s) recognized.")
        print("Program performance: %0.1f frames per second." % performance)
        for i, wave in enumerate(recognized_waves):
            print(
                f"[Wave #{i + 1}] "
                f"ID: {wave.name}, "
                f"Birth: {wave.birth}, "
                f"Death: {wave.death}, "
                f"Max Displacement: {wave.max_displacement}, "
                f"Max Mass: {wave.max_mass}"
            )
    else:
        print("No waves recognized.")

    # Clean-up resources.
    if write_output is True:
        out.release()

    return recognized_waves, wave_log, performance


def main():
    """Define main."""
    # CLI.
    args = arg_parser.parse_args()
    inputfile = args.video_path

    # Read video.
    print("Checking video from", inputfile)
    inputvideo = cv2.VideoCapture(inputfile)

    # Exit if video cannot be opened.
    if not inputvideo.isOpened():
        sys.exit("Could not open video. Exiting.")

    # Get a wave log, list of recognized waves, and program performance
    # from analyze, as well as create a visualization video.
    recognized_waves, wave_log, program_speed = analyze(
        inputvideo, write_output=True
    )

    # Write the wave log to csv.
    mwt_io.write_log(wave_log, output_format="json")

    # Write the analysis report to txt.
    mwt_io.write_report(recognized_waves, program_speed)

    # Clean-up resources.
    inputvideo.release()


if __name__ == "__main__":
    main()
