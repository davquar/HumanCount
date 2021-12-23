import numpy as np
import cv2


class CountoursDetector:
    def __init__(self, threshold, maxval, type):
        self.threshold = threshold
        self.maxval = maxval
        self.type = type

    def work(self, frame, mode=cv2.RETR_EXTERNAL, remove_shadows=False):
        # frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not remove_shadows:
            _, thresh = cv2.threshold(frame, self.threshold, self.maxval, self.type)
        else:
            _, thresh = cv2.threshold(frame, 20, self.maxval, self.type)

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, _ = cv2.findContours(
            image=thresh, mode=mode, method=cv2.CHAIN_APPROX_NONE
        )

        canvas = np.zeros((len(frame), len(frame[0]), 3), np.uint8)

        # draw contours on the original image
        cv2.drawContours(
            image=canvas,
            contours=contours,
            contourIdx=-1,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        return canvas, contours
