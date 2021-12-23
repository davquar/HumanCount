import cv2


class BackgroundSubtractor:
    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, history=0, varThreshold=None
        )

    def work(self, frame):
        return self.subtractor.apply(frame)
