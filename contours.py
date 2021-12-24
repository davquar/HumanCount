import numpy as np
import cv2


class CountoursDetector:
    """
    Class that packs techniques and parameters to perform contours detection
    """

    def __init__(self, threshold, maxval, threshold_type):
        self.threshold = threshold
        self.maxval = maxval
        self.threshold_type = threshold_type

    def work(self, frame, mode=cv2.RETR_EXTERNAL, remove_shadows=False):
        """
        Method that starts contours detection with the configured parameters
        """
        # frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not remove_shadows:
            _, thresholded = cv2.threshold(
                frame, self.threshold, self.maxval, self.threshold_type
            )
        else:
            _, thresholded = cv2.threshold(frame, 20, self.maxval, self.threshold_type)

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, _ = cv2.findContours(
            image=thresholded, mode=mode, method=cv2.CHAIN_APPROX_NONE
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
