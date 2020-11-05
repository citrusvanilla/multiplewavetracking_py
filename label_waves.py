from __future__ import division

import sys
import getopt
import time
import pandas as pd
import numpy as np
import copy

import cv2

import mwt_detection
import mwt_preprocessing
import mwt_tracking
import mwt_io
import mwt_objects

def label(video, log):
    # Initiate an empty list of tracked waves, ultimately recognized
    # waves, and a log of all tracked waves in each frame.

    # Initialize frame counters.
    frame_num = 1
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initiate a timer for program performance:
    time_start = time.time()

    # The main loop is here:
    while True:

        # Read frames until end of clip.
        print(frame_num)
        successful_read, original_frame = video.read()
        if not successful_read:
            break

        # Preprocess frames.
        analysis_frame = mwt_preprocessing.preprocess(original_frame)
        if frame_num == 63:
            contours, _ = cv2.findContours(
                                image=analysis_frame,
                                mode=cv2.RETR_EXTERNAL,
                                method=cv2.CHAIN_APPROX_NONE,
                                hierarchy=None,
                                offset=None)
            for contour in contours:
                centroid = mwt_objects._get_centroid(contour)
                # print(centroid)
                # input()
                # if centroid == [219, 169]:
                sm_boundRect = cv2.boundingRect(contour)
                print(sm_boundRect)
                color = (0,0,255)
                resize_factor = 1/mwt_preprocessing.RESIZE_FACTOR
                boundRect = []
                for i in range(len(sm_boundRect)):
                    boundRect.append(sm_boundRect[i] * 4)
                print(boundRect)
                display_frame = copy.deepcopy(original_frame)
                cv2.rectangle(display_frame, (int(boundRect[0]), int(boundRect[1])), \
                (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), color, 2)
                cv2.imshow('Wave', display_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        frame_num += 1

def main():

    """main"""
    # The command line should have one argument-
    # the name of the videofile.
    if (len(sys.argv) != 3):
        print("usage: label_waves.py <videofile> <wavelog.json>")
        sys.exit(2)
    videofile = sys.argv[1]
    wavelog = sys.argv[2]
    # try:
    #     opts, args = getopt.getopt(argv, "i:")
    # except getopt.GetoptError:
    #     print ("usage: label_waves.py -i <inputfile>")
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt == ("-i"):
    #         inputfile = arg

    # Read video.
    print ("Checking video from", videofile)
    inputvideo = cv2.VideoCapture(videofile)

    # Exit if video cannot be opened.
    if not inputvideo.isOpened():
        sys.exit("Could not open video.")

    log = pd.read_json(wavelog)
    # Get a wave log, list of recognized waves, and program performance
    # from analyze, as well as create a visualization video.
    label(inputvideo, log)

    # Clean-up resources.
    inputvideo.release()

if __name__=='__main__':
    main()