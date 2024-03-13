# import numpy as np
# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# from sort import *
#
# def crossed_line(point, line_start, line_end):
#     """
#     Check if a point crosses a line segment.
#
#     Parameters:
#         point (tuple): Coordinates of the point (x, y).
#         line_start (tuple): Coordinates of the start point of the line segment (x, y).
#         line_end (tuple): Coordinates of the end point of the line segment (x, y).
#
#     Returns:
#         bool: True if the point crosses the line segment, False otherwise.
#     """
#     x, y = point
#     x1, y1 = line_start
#     x2, y2 = line_end
#
#     if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
#         # Calculate the cross product
#         cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
#         return cross_product == 0  # Point lies on the line
#     else:
#         return False  # Point is outside the line segment
#
#
# cap = cv2.VideoCapture("Videos/cars.mp4")  # For Video
# model = YOLO("../Yolo-Weights/yolov8l.pt")
#
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"]
#
# mask = cv2.imread("mask.png")
#
# # Tracking
# tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
#
# limits = [400, 187, 673, 187]
# totalCount = []
#
# # Define the two lines for counting
# line_enter = [(limits[0], limits[1]), (limits[2], limits[3])]  # Line for counting vehicles entering
# line_exit = [(limits[0], limits[1] + 300), (limits[2], limits[3] + 300)]  # Line for counting vehicles exiting
#
# while True:
#     success, img = cap.read()
#
#     mask_resized = cv2.resize(mask, (1280, 720))
#     imgRegion = cv2.bitwise_and(img, mask_resized)
#
#     imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
#     img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
#     results = model(imgRegion, stream=True)
#
#     detections = np.empty((0, 5))
#
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             w, h = x2 - x1, y2 - y1
#
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#             currentClass = classNames[cls]
#
#             if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
#                 currentArray = np.array([x1, y1, x2, y2, conf])
#                 detections = np.vstack((detections, currentArray))
#
#     resultsTracker = tracker.update(detections)
#
#     cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
#     cv2.line(img, (limits[0], limits[1] + 300), (limits[2], limits[3] + 300), (255, 0, 0), 5)
#
#     vehicles_between = 0  # Initialize count of vehicles between entry and exit lines
#
#     for result in resultsTracker:
#         x1, y1, x2, y2, id = result
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         w, h = x2 - x1, y2 - y1
#         cx, cy = x1 + w // 2, y1 + h // 2
#
#         # Check if the vehicle crosses the line for entering
#         if crossed_line([cx, cy], line_enter[0], line_enter[1]):
#             if id not in totalCount:  # Check if the vehicle ID is not already counted
#                 totalCount.append(id)
#
#         # Check if the vehicle crosses the line for exiting
#         if crossed_line([cx, cy], line_exit[0], line_exit[1]):
#             if id in totalCount:  # Check if the vehicle ID was counted while entering
#                 totalCount.remove(id)  # Remove the vehicle ID from the list
#
#         # Check if the vehicle is between the entry and exit lines
#         if limits[1] < cy < limits[1] + 200:
#             vehicles_between += 1
#
#         cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
#         cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=10)
#         cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
#
#     cv2.putText(img, f'Vehicles Between: {vehicles_between}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
#     # cv2.putText(img, f'Total Count: {len(totalCount)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
#
#     cv2.imshow("Image", img)
#     k = cv2.waitKey(1)
#     if k == ord("s") & 0xFF:
#         break

import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

def crossed_line(point, line_start, line_end):
    """
    Check if a point crosses a line segment.

    Parameters:
        point (tuple): Coordinates of the point (x, y).
        line_start (tuple): Coordinates of the start point of the line segment (x, y).
        line_end (tuple): Coordinates of the end point of the line segment (x, y).

    Returns:
        bool: True if the point crosses the line segment, False otherwise.
    """
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
        # Calculate the cross product
        cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        return cross_product == 0  # Point lies on the line
    else:
        return False  # Point is outside the line segment

def count_vehicles_between(line_enter, line_exit, resultsTracker, totalCount):
    vehicles_between = 0  # Initialize count of vehicles between entry and exit lines

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Check if the vehicle crosses the line for entering
        if crossed_line([cx, cy], line_enter[0], line_enter[1]):
            if id not in totalCount:  # Check if the vehicle ID is not already counted
                totalCount.append(id)

        # Check if the vehicle crosses the line for exiting
        if crossed_line([cx, cy], line_exit[0], line_exit[1]):
            if id in totalCount:  # Check if the vehicle ID was counted while entering
                totalCount.remove(id)  # Remove the vehicle ID from the list

        # Check if the vehicle is between the entry and exit lines
        if limits[1] < cy < limits[1] + 200:
            vehicles_between += 1

    return vehicles_between

cap = cv2.VideoCapture("Videos/cars.mp4")  # For Video
model = YOLO("../Yolo-Weights/yolov8m.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 167, 673, 167]
totalCount = []

# Define the two lines for counting
line_enter = [(limits[0], limits[1]), (limits[2], limits[3])]  # Line for counting vehicles entering
line_exit = [(limits[0], limits[1] + 300), (limits[2], limits[3] + 300)]  # Line for counting vehicles exiting

# Coordinates of the vertices of the trapezoid
x1, y1 = line_enter[0]
x2, y2 = line_enter[1]
x3, y3 = line_exit[0]
x4, y4 = line_exit[1]

# Length of the longer side (base1)
base1 = abs(x2 - x1)

# Length of the shorter side (base2)
base2 = abs(x4 - x3)

# Perpendicular distance between the two bases (height)
height = abs(y3 - y1)

# Calculate the area of the trapezoid
area_trapezoid = 0.5 * (base1 + base2) * height

print("Area of the trapezoid:", area_trapezoid)

while True:
    success, img = cap.read()

    mask_resized = cv2.resize(mask, (1280, 720))
    imgRegion = cv2.bitwise_and(img, mask_resized)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img, (limits[0], limits[1] + 300), (limits[2], limits[3] + 300), (255, 0, 0), 5)

    vehicles_between = 0  # Initialize count of vehicles between entry and exit lines

    # Check if the "e" key is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == ord('e'):
        vehicles_between = count_vehicles_between(line_enter, line_exit, resultsTracker, totalCount)

    cv2.putText(img, f'Vehicles Between: {vehicles_between}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    density = (vehicles_between * 10000) / area_trapezoid
    density_ratio = round(density, 3)
    cv2.putText(img, f'Density: {density_ratio}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    if 0.1 <= density_ratio <= 0.399:
        cv2.putText(img, f'count: 2 sec', (50, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)
    elif 0.4 <= density_ratio <= 0.699:
        cv2.putText(img, f'count: 7 sec', (50, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)
    elif density >= 0.7:
        cv2.putText(img, f'count: 10 sec', (50, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    if k == ord("s"):
        break

