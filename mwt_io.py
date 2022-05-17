"""Handle input and output of near-shore wave tracking analysis."""
from __future__ import division

import os
import csv
import json

import numpy as np
import cv2

# We are going to dump our output in a folder in the same directory.
OUTPUT_DIR = "output"

# Names of output files to be written to output/ go here:
WAVE_LOG_CSVFILE = "wave_log.csv"
WAVE_LOG_JSONFILE = "wave_log.json"
RECOGNIZED_WAVE_REPORT_FILE = "recognized_waves.txt"
TRACKED_WAVE_FILE = "tracked_waves.mp4"


def create_video_writer(input_video):
    """Create a video.

    Creates a OpenCV Video Writer object using the mp4c codec and
    input video stats (frame width, height, fps) for tracking
    visualization.

    Args:
        input_video: video read into program using opencv methods

    Returns:
        out: cv2 videowriter object to which individual frames can be
             written
    """
    # Grab some video stats for videowriter object.
    original_width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = input_video.get(cv2.CAP_PROP_FPS)

    # Make an output directory if necessary.
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Initiate video writer object by defining the codec and initiating
    # the VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(OUTPUT_DIR, TRACKED_WAVE_FILE)
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (int(original_width), int(original_height)),
        isColor=True,
    )

    return out


def write_log(log, output_format="csv"):
    """Write the log.

    Takes a list of list of wave attributes representing each tracked wave's
    statistics in every frame of the video for which the wave was present.
    Outputs each item in the list as a line to a CSV.

    Args:
        log: a list of lists of wave statistics
        format: either "csv" for CSV or "json" for JSON
    """
    # Print status update.
    if not log:
        print("No waves or sections detected.  No log written.")
    else:
        if output_format == "csv":
            print("Writing wave log to", WAVE_LOG_CSVFILE)
        elif output_format == "json":
            print("Writing wave log to", WAVE_LOG_JSONFILE)

    # Declare headers here.
    log_headers = [
        "frame_num",
        "wave_id",
        "inst_mass",
        "max_mass",
        "inst_displacement",
        "max_displacement",
        "frame_birth",
        "frame_death",
        "recognized",
        "centroid",
    ]

    # Make an output directory if necessary.
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Using a context manager, write each item to a new row in output.
    if output_format == "csv":
        output_path = os.path.join(OUTPUT_DIR, WAVE_LOG_CSVFILE)
    elif output_format == "json":
        output_path = os.path.join(OUTPUT_DIR, WAVE_LOG_JSONFILE)

    with open(output_path, "w") as outfile:
        if output_format == "csv":
            writer = csv.writer(outfile, delimiter=",")
            writer.writerow(log_headers)
            for row in log:
                writer.writerow(row)
        elif output_format == "json":
            outfile.write("[")
            for count, row in enumerate(log):
                json_line = {}
                for i, j in enumerate(log_headers):
                    json_line[j] = row[i]
                json.dump(json_line, outfile)
                if count == len(log) - 1:
                    outfile.write("\n")
                else:
                    outfile.write(",\n")
            outfile.write("]")

    return


def write_report(waves, performance):
    """Write a simple report.

    Takes a list of recognized wave objects and writes attributes out to a
    plain text file.  Supplements this information with program performance
    and user stats.

    Args:
        waves: a list of wave objects
        performance_metric: a double representing speed of program
    """
    # Provide User feedback here.
    if not waves:
        print("No waves found.  No report written.")
    else:
        print("Writing analysis report to", RECOGNIZED_WAVE_REPORT_FILE)

    # Make an output directory if necessary.
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Write recognized waves to text file.
    report_path = os.path.join(OUTPUT_DIR, RECOGNIZED_WAVE_REPORT_FILE)

    # Use the context manager here:
    with open(report_path, "w") as text_file:
        text_file.write(
            f"Program performance: {performance} frames per second.\n"
        )
        for i, wave in enumerate(waves):
            text_file.write(
                f"[Wave #{i+1}] "
                f"ID: {wave.name}, "
                f"Birth: {wave.birth}, "
                f"Death: {wave.death}"
            )
            text_file.write(
                f"Max Displacement: {wave.max_displacement}, "
                f"Max Mass: {wave.max_mass}\n"
            )

    return


def draw(waves, frame, resize_factor):
    """Draw detection on a frame.

    Simple function to draw on a frame for output.  Draws bounding boxes in
    accordance with wave.boundingbox_coors attribute, and draws some wave stats
    to accompany each potential wave, including whether or not the object is
    actually a wave (i.e. wave.recognized == True).

    Args:
        waves: list of waves
        frame: frame on which to draw waves
        resize_factor: factor to resize boundingbox coors to match output
                       frame.

    Returns:
        frame: input frame with waves drawn on top
    """
    # Iterate through a list of waves.

    for wave in waves:

        # For drawing circles on detected features
        # center = (wave.centroid[0],wave.centroid[1])
        # radius = 15
        # cv2.circle(frame,center,radius,(0,255,0),2)

        if wave.death is None:
            # If wave is a wave, draw green, else yellow.
            # Set wave text accordingly.
            if wave.recognized is True:
                drawing_color = (0, 255, 0)
                text = (
                    f"Wave Detected!\n"
                    f"mass: { wave.mass}\n"
                    f"displacement: {wave.displacement}"
                )

            else:
                drawing_color = (0, 255, 255)
                text = (
                    f"Potential Wave\n"
                    f"mass: {wave.mass}\n"
                    f"displacement: {wave.displacement}"
                )

            if len(wave.centroid_vec) > 20:
                # Draw Bounding Boxes:
                # Get boundingbox coors from wave objects and resize.
                """
                rect = wave.boundingbox_coors
                rect[:] = [resize_factor*rect[i] for i in range(4)]
                frame = cv2.drawContours(frame, [rect], 0, drawing_color, 2)"""

                # Use moving averages of wave centroid for stat locations
                moving_x = np.mean(
                    [
                        wave.centroid_vec[-k][0]
                        for k in range(1, min(20, 1 + len(wave.centroid_vec)))
                    ]
                )
                moving_y = np.mean(
                    [
                        wave.centroid_vec[-k][1]
                        for k in range(1, min(20, 1 + len(wave.centroid_vec)))
                    ]
                )

                # Draw wave stats on each wave.
                for i, j in enumerate(text.split("\n")):
                    frame = cv2.putText(
                        frame,
                        text=j,
                        org=(
                            int(resize_factor * moving_x),
                            int(resize_factor * moving_y) + (50 + i * 45),
                        ),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5,
                        color=drawing_color,
                        thickness=3,
                        lineType=cv2.LINE_AA,
                    )

    return frame
