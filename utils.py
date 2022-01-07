import math
import json
import os
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


def filter_bounding_boxes(hog_boxes, small_boxes, min_size=None, max_size=None):
    """
    Utility function that maintains the bounding boxes that are inside the HOG ones
    """

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


def write_people_count(frame, count):
    """
    Utility function to write the current people count on the frame
    """
    cv2.putText(
        frame,
        f"People: {count}",
        (5, 270),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        0.8,
        (255, 255, 255),
    )


def write_average_people_distance(frame, value):
    """
    Utility function to write the average distance between people on the frame
    """
    cv2.putText(
        frame,
        f"Avg. dist.: {value}m",
        (5, 290),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        0.8,
        (255, 255, 255),
    )


def degree_to_radians(deg):
    return deg * math.pi / 180


def point_distance(p1, p2):
    c1 = abs(p1[0] - p2[0])
    c2 = abs(p1[1] - p2[1])
    return math.sqrt(c1 * c1 + c2 * c2)


def point_direction(p1, p2):
    delta_x = p2[0] - p1[0]  # p2.x - p1.x
    delta_y = p1[1] - p2[1]  # p1.y - p2.y
    return math.atan2(delta_y, delta_x)


def get_distance_to_camera(frame, boxes, cam_height, cam_min_angle, cam_max_angle):
    distance_boxes = []
    for box in boxes:
        diff_angle = cam_max_angle - cam_min_angle
        cur_y = frame.shape[1] - (box[1] + box[3])
        cur_angle = cam_min_angle + diff_angle / frame.shape[1] * cur_y
        dist = cam_height * math.tan(degree_to_radians(cur_angle))
        distance_boxes.append((box, dist))
    return distance_boxes


def draw_distance_to_camera(frame, distance_boxes):
    for dist_box in distance_boxes:
        box, dist = dist_box

        cv2.putText(
            frame,
            "{:.2f}m".format(dist),
            (box[0], box[1]),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.6,
            (255, 255, 255),
        )


def draw_distance_between_people(frame, distance_boxes, pers_height):
    distances = []
    visited = []

    for dist_box1 in distance_boxes:
        closer_box = None
        closer_dist = None
        for dist_box2 in distance_boxes:
            not_visited = (dist_box1, dist_box2) not in visited and (
                dist_box2,
                dist_box1,
            ) not in visited
            if dist_box1 != dist_box2:
                pers1_x = dist_box1[0][0] + dist_box1[0][2] * 0.5
                pers2_x = dist_box2[0][0] + dist_box2[0][2] * 0.5

                pers1_ratio = pers_height / dist_box1[0][3]
                pers2_ratio = pers_height / dist_box2[0][3]

                dist_w_px = abs(pers1_x - pers2_x)
                dist1_w_m = dist_w_px * pers1_ratio
                dist2_w_m = dist_w_px * pers2_ratio

                c1 = abs(dist_box1[1] - dist_box2[1])
                c2 = (dist1_w_m + dist2_w_m) * 0.5

                dist_m = math.sqrt(c1 * c1 + c2 * c2)

                if not_visited:
                    distances.append(dist_m)

                if closer_box is None or dist_m < closer_dist:
                    closer_box = dist_box2
                    closer_dist = dist_m

                visited.append((dist_box1, dist_box2))

        if closer_box is not None:
            pers1_coord = (
                round(dist_box1[0][0] + dist_box1[0][2] * 0.5),
                round(dist_box1[0][1] + dist_box1[0][3]),
            )
            pers2_coord = (
                round(closer_box[0][0] + closer_box[0][2] * 0.5),
                round(closer_box[0][1] + closer_box[0][3]),
            )
            cv2.line(frame, pers1_coord, pers2_coord, (255, 255, 255), 1)

            cur_dist = point_distance(pers1_coord, pers2_coord)
            cur_dir = point_direction(pers1_coord, pers2_coord)
            cur_x = round(pers1_coord[0] + cur_dist * 0.5 * math.cos(cur_dir))
            cur_y = round(pers1_coord[1] + cur_dist * 0.5 * -math.sin(cur_dir))

            text_dist = "{:.1f}m".format(closer_dist)
            text_size = cv2.getTextSize(
                text_dist, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, None
            )[0]
            text_off_x = round(-text_size[0] * 0.5)
            text_off_y = round(-text_size[1] * 0.5)
            cv2.putText(
                frame,
                text_dist,
                (cur_x + text_off_x, cur_y + text_off_y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.6,
                (0, 0, 255),
            )
    return distances


def read_input_json(path: str) -> dict:
    conf = {}
    with open(path, "r") as f:
        conf = json.load(f)

    path_to_prepend = os.path.dirname(path) + "/"
    conf["video"] = path_to_prepend + conf["video"]
    conf["background"] = path_to_prepend + conf["background"]
    return conf
