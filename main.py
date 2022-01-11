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
        self.cap = cv2.VideoCapture(conf["video"])

        # Initialize HOG and SVM for people detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Initialize background subtractor and contours detector
        self.subtractor = BackgroundSubtractor()
        self.contours_detector = CountoursDetector(50, 255, cv2.THRESH_BINARY)

        # Read the still background (given in input)
        self.background = cv2.imread(conf["background"])
        self.background = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        self.background = cv2.resize(self.background, (350, 300))

        # Extract background contours
        self.canvas_background, self.contours_background = self.contours_detector.work(
            self.background
        )

        # Initialize current frame holders (colored, grayscale, heatmap)
        self.frame = None
        self.gray = None
        self.heatmap = np.zeros_like(self.background, dtype=np.uint8)

    def do_hog_svm(self):
        """
        Performs the HOG-SVM people detection on the current frame
        """
        boxes, _ = self.hog.detectMultiScale(
            self.frame, winStride=(4, 4), scale=1.05, padding=(4, 4)
        )
        return boxes

    def do_object_detection(self, use_mog2=False):
        """
        Performs the generic object detection on the current frame,
        by combining segmentation and contours detection
        """
        if use_mog2:
            segmented = self.subtractor.work(self.gray)
        else:
            segmented = cv2.absdiff(self.gray, self.background)
        cv2.imshow("segmented", segmented)

        canvas_segmented, contours_segmented = self.contours_detector.work(
            segmented, mode=cv2.RETR_EXTERNAL, remove_shadows=True
        )
        cv2.imshow("contours", canvas_segmented)

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

    def draw_heatmap(self, hog_boxes):
        """
        Draws a heatmap based on the boxes that are given in input
        """
        for box in hog_boxes:
            x, y, w, h = box
            sub_img = self.heatmap[y : y + h, x : x + w]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            res = cv2.addWeighted(sub_img, 1, white_rect, 0.05, 0)
            self.heatmap[y : y + h, x : x + w] = res

    def darken_heatmap(self):
        """
        Puts a shade of black on top of the current heatmap
        """
        black_overlay = np.zeros_like(self.heatmap, dtype=np.uint8)
        self.heatmap = cv2.addWeighted(self.heatmap, 0.9, black_overlay, 0.5, 0)

    def start(self):
        """
        Holds the main loop for HOG+SVM and object detection on each frame
        """
        frame_counter = 0
        while True:
            read, self.frame = self.cap.read()
            if not read:
                break

            self.frame = cv2.resize(self.frame, (350, 300))
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            hog_boxes = self.do_hog_svm()
            small_boxes = self.do_object_detection(use_mog2=args.use_mog2)

            filtered_boxes = utils.filter_bounding_boxes(hog_boxes, small_boxes, 200)
            distance_boxes = utils.get_distance_to_camera(
                self.frame,
                filtered_boxes,
                conf["camera_conf"]["height"],
                conf["camera_conf"]["lower_angle"],
                conf["camera_conf"]["upper_angle"],
            )

            utils.draw_hog_bounding_boxes(self.frame, hog_boxes, (255, 0, 0))
            utils.draw_bounding_boxes(self.frame, filtered_boxes, (0, 255, 0))
            utils.draw_bounding_boxes(self.frame, small_boxes, (0, 255, 0))

            avg = round((len(hog_boxes) + len(small_boxes)) / 2)
            utils.write_people_count(self.frame, avg)

            # draw people distance to camera
            # utils.draw_distance_to_camera(self.frame, distance_boxes)

            # draw distances between people
            distances = utils.draw_distance_between_people(
                self.frame, distance_boxes, 1.70
            )

            # draw average distance between people
            average_distance = 0
            if len(distances) > 0:
                average_distance = round(sum(distances) / (len(distances)), 1)

            utils.write_average_people_distance(self.frame, average_distance)

            if frame_counter % 10 == 0:
                self.darken_heatmap()
                self.draw_heatmap(small_boxes)

            window_content = np.hstack(
                (self.frame, cv2.cvtColor(self.heatmap, cv2.COLOR_GRAY2BGR))
            )
            cv2.imshow("Frame and heatmap", window_content)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_counter += 1

        self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("-i", "--input", help="Input JSON", required=True)
    argparse.add_argument(
        "-m",
        "--use-mog2",
        help="Use MOG2 to perform background subtraction",
        action="store_true",
    )
    argparse.add_argument(
        "-s", "--show", help="Show the result in a window", action="store_true"
    )
    args = argparse.parse_args()

    conf = utils.read_input_json(args.input)

    app = App()
    app.start()
