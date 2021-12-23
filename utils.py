import cv2


def draw_bounding_boxes(frame, contours, colors, min_size, max_size):
    for contour in contours:
        if min_size is not None and cv2.contourArea(contour) < min_size:
            continue
        if max_size is not None and cv2.contourArea(contour) > max_size:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2)
