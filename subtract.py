import cv2


class BackgroundSubtractor:
    """
    Class that packs techniques and parameters to perform background subtraction
    """

    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, history=100, varThreshold=50
        )
        self.subtractor.setNMixtures(5)

    def work(self, frame):
        """
        Method that starts MOG2 background subtraction with the configured parameters
        """
        self.subtractor.apply(frame)
        return self.subtractor.getBackgroundImage()
