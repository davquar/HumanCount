import cv2
import numpy as np


def draw_bounding_boxes(frame, contours, colors, min_size=None, max_size=None):
    """
    Utility function that draws bounding boxes on the given frame,
    according to the given parameters
    """
    for contour in contours:
        if min_size is not None and cv2.contourArea(contour) < min_size:
            continue
        if max_size is not None and cv2.contourArea(contour) > max_size:
            continue
        coords_x, coords_y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(
            frame,
            (coords_x, coords_y),
            (coords_x + width, coords_y + height),
            colors,
            2,
        )


def draw_hog_bounding_boxes(frame, boxes, colors):
    """
    Utility function that draws the given HOG-returned bounding boxes
    on the given frame, according to the given parameters
    """
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    for (x_a, y_a, x_b, y_b) in boxes:
        cv2.rectangle(frame, (x_a, y_a), (x_b, y_b), colors, 2)
