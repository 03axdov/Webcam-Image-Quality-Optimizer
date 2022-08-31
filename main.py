import os
import numpy as np
import tensorflow as tf
import cv2

from data_processing import process_image

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        if ret:
            hr_frame = process_image(frame)
            cv2.imshow('Frame', hr_frame.numpy())

        if cv2.waitKey(1) == ord('q'):
            break   


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()