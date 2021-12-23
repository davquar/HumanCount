import sys
import numpy as np
import cv2


class BackgroundSubtractor:
    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, history=None, varThreshold=50
        )

    def work(self, frame):
        return self.subtractor.apply(frame)
