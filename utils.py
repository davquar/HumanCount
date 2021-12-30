import cv2
import numpy as np


def draw_bounding_boxes(frame, boxes, colors):
    """
    Utility function that draws bounding boxes on the given frame,
    according to the given parameters
    """
    for box in boxes:
        coords_x, coords_y, width, height = box
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


def normalize_small_boxes(contours, min_size, max_size):
    """
    Utility function that from a list of contours,
    returns a list of bounding boxes that satisfy the given size requirements
    """
    ret = []
    for contour in contours:
        if min_size is not None and cv2.contourArea(contour) < min_size:
            continue
        if max_size is not None and cv2.contourArea(contour) > max_size:
            continue
        ret.append(cv2.boundingRect(contour))  # coords_x, coords_y, width, height
    return ret


def filter_bounding_boxes(
    hog_boxes, small_boxes_contours, min_size=None, max_size=None
):
    """
    Utility function that maintains the bounding boxes that are inside the HOG ones
    """
    small_boxes = normalize_small_boxes(small_boxes_contours, min_size, max_size)

    ret = []
    for hog_box in hog_boxes:
        for small_box in small_boxes:
            if (
                small_box[0] >= hog_box[0]
                and small_box[1] >= hog_box[1]
                and (small_box[0] + small_box[2]) <= (hog_box[0] + hog_box[2])
                and (small_box[1] + small_box[3]) <= (hog_box[1] + hog_box[3])
            ):
                ret.append(small_box)
    return ret
