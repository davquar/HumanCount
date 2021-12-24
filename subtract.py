import cv2


class BackgroundSubtractor:
    """
    Class that packs techniques and parameters to perform background subtraction
    """

    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, history=0, varThreshold=None
        )

    def work(self, frame):
        """
        Method that starts MOG2 background subtraction with the configured parameters
        """
        return self.subtractor.apply(frame)
