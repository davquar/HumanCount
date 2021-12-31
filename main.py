import argparse
import numpy as np
import cv2

from subtract import BackgroundSubtractor
from contours import CountoursDetector
import utils


class App:
    """
    Coordinator for the application
    """

    def __init__(self):
        cv2.startWindowThread()
        self.cap = cv2.VideoCapture(args.input)

        # Initialize HOG and SVM for people detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Initialize background subtractor and contours detector
        self.subtractor = BackgroundSubtractor()
        self.contours_detector = CountoursDetector(50, 255, cv2.THRESH_BINARY)

        # Read the still background (given in input)
        self.background = cv2.imread(args.background)
        self.background = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        self.background = cv2.resize(self.background, (350, 300))

        # Extract background contours
        self.canvas_background, self.contours_background = self.contours_detector.work(
            self.background
        )

        # Initialize current frame holders (colored and grayscale)
        self.frame = None
        self.gray = None

    def do_hog_svm(self):
        """
        Performs the HOG-SVM people detection on the current frame
        """
        boxes, _ = self.hog.detectMultiScale(
            self.frame, winStride=(4, 4), scale=1.05, padding=(4, 4)
        )
        return boxes

    def do_object_detection(self):
        """
        Performs the generic object detection on the current frame,
        by combining segmentation and contours detection
        """
        segmented = cv2.absdiff(self.gray, self.background)
        canvas_segmented, contours_segmented = self.contours_detector.work(
            segmented, mode=cv2.RETR_EXTERNAL, remove_shadows=True
        )

        # diff = cv2.subtract(canvas_segmented, self.canvas_background)
        diff_contours = np.array(contours_segmented, copy=True, dtype=object)

        # remove common contours
        for contour in self.contours_background:
            if np.any(np.isin(contour, diff_contours)):
                try:
                    np.delete(diff_contours, contour)
                except IndexError as error:
                    print(error)

        # utils.draw_bounding_boxes(self.frame, diff_contours, (0, 255, 0), 200)
        return utils.normalize_small_boxes(diff_contours, 200, None)

    def start(self):
        """
        Holds the main loop for HOG+SVM and object detection on each frame
        """
        while True:
            read, self.frame = self.cap.read()
            if not read:
                break

            self.frame = cv2.resize(self.frame, (350, 300))
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            hog_boxes = self.do_hog_svm()
            small_boxes = self.do_object_detection()

            distance_boxes = utils.get_distance_to_camera(self.frame, small_boxes, 5.5, 15, 85)
            # filtered_boxes = utils.filter_bounding_boxes(hog_boxes, small_boxes, 200)

            # utils.draw_hog_bounding_boxes(self.frame, hog_boxes, (255, 0, 0))
            # utils.draw_bounding_boxes(self.frame, filtered_boxes, (0, 255, 0))
            utils.draw_bounding_boxes(self.frame, small_boxes, (0, 255, 0))

            avg = round((len(hog_boxes) + len(small_boxes)) / 2)
            utils.write_people_count(self.frame, avg)

            # draw people distance to camera
            # utils.draw_distance_to_camera(self.frame, distance_boxes)

            # draw distances between people
            utils.draw_distance_between_people(self.frame, distance_boxes, 1.70)

            cv2.imshow("frame", self.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("-i", "--input", help="Input video", required=True)
    argparse.add_argument(
        "-b", "--background", help="Background of the current video", required=True
    )
    argparse.add_argument(
        "-s", "--show", help="Show the result in a window", action="store_true"
    )
    args = argparse.parse_args()

    app = App()
    app.start()
