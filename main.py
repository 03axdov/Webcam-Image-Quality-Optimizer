import os
import numpy as np
import tensorflow as tf
import cv2
import time

from data_processing import process_image

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    cap = cv2.VideoCapture(0)

    tic = time.time()
    ret, frame = cap.read() 
    cap.release()

    if ret:
        hr_frame = process_image(frame).numpy()[0]
        hr_frame /= 255.0
        print("")
        print(f"HR_FRAME: {hr_frame}")
        print("")
        hr_frame = cv2.resize(hr_frame, (960, 540)) 
        frame = cv2.resize(frame, (960, 540))
        toc = time.time() 
        print("")
        print(f"[ PROGRAM COMPLETED IN {int(toc-tic)} SECONDS ]")
        print("")

        cv2.imshow('Previous', frame)
        cv2.waitKey(0)
        cv2.imshow('High Resolution', hr_frame)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()