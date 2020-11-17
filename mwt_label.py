import numpy as np
import os
import sys
import cv2
import time

LABEL_DIR = "wave_data"

def label(wave_list, fps, dead=False):
    if not os.path.exists(LABEL_DIR):
        os.mkdir(LABEL_DIR)

    for wave in wave_list:
        # Display waves for rating
        if len(wave.frame_data) >= fps or dead:
            cv2.imshow("Enter a number between 0 and 9", wave.frame_data[0])
            ret = -1
            # '0' is ord 48, '9' is ord 57 
            while (ret < 48 or ret > 57):
                ret = cv2.waitKey(0)

            # User enters 0-9. Rating is scaled to be 1-10
            rating = int(chr(ret)) + 1
            rating_dir = "rating_" + str(rating)
            rating_path = os.path.join(LABEL_DIR, rating_dir)
            
            if not os.path.exists(rating_path):
                os.mkdir(rating_path)
            
            # Write either a second's worth of data, or the rest of 
            # data to the rating of the first frame
            for i in range(min(fps, len(wave.frame_data))):
                img_name = str(time.time()).replace(".","_") + ".jpg"
                img_path = os.path.join(rating_path, img_name)
                cv2.imwrite(img_path, wave.frame_data[i])
            
            # Update frame data, don't need to track frames already written
            wave.frame_data = wave.frame_data[i+1:]





