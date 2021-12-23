import cv2
import numpy as np


def draw_bounding_boxes(frame, contours, colors, min_size=None, max_size=None):
    for contour in contours:
        if min_size is not None and cv2.contourArea(contour) < min_size:
            continue
        if max_size is not None and cv2.contourArea(contour) > max_size:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2)


def draw_hog_bounding_boxes(frame, boxes, colors):
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), colors, 2)
