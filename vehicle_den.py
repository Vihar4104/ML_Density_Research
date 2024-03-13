import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import pandas as pd
from datetime import datetime

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

# Initialize VideoCapture, YOLO model, etc.
cap = cv2.VideoCapture("Videos/video.mp4")  # For Video
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
mask = cv2.imread("mask4.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 167, 673, 167]
totalCount = []
line_enter = [(limits[0], limits[1]), (limits[2], limits[3])]  # Line for counting vehicles entering
line_exit = [(limits[0], limits[1] + 300), (limits[2], limits[3] + 300)]  # Line for counting vehicles exiting
x1, y1 = line_enter[0]
x2, y2 = line_enter[1]
x3, y3 = line_exit[0]
x4, y4 = line_exit[1]
base1 = abs(x2 - x1)
base2 = abs(x4 - x3)
height = abs(y3 - y1)
area_trapezoid = 0.5 * (base1 + base2) * height

# Read existing data from the Excel file, if exists
try:
    df = pd.read_excel("vehicle_data.xlsx")
    data = df.values.tolist()
except FileNotFoundError:
    data = []

# Create a list to store new data
new_data = []

while True:
    success, img = cap.read()

    mask_resized = cv2.resize(mask, (1280, 720))
    print(mask.shape)
    print(img.shape)
    imgRegion = cv2.bitwise_and(img, mask_resized)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img, (limits[0], limits[1] + 300), (limits[2], limits[3] + 300), (255, 0, 0), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    vehicles_between = 0
    density_ratio = 0
    k = cv2.waitKey(1) & 0xFF
    if k == ord('e'):
        Time = 0
        vehicles_between = count_vehicles_between(line_enter, line_exit, resultsTracker, totalCount)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        density = (vehicles_between * 10000) / area_trapezoid
        density_ratio = round(density, 3)
        if 0.1 <= density_ratio <= 0.399:
            Time = '2 sec'
        elif 0.4 <= density_ratio <= 0.699:
            Time = '7 sec'
        elif density >= 0.7:
            Time = '10 sec'
        new_data.append([vehicles_between, density_ratio, current_time,Time])

    cv2.putText(img, f'Vehicles Between: {vehicles_between}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(img, f'Density: {density_ratio}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    if 0.1 <= density_ratio <= 0.399:
        cv2.putText(img, f'count: 2 sec', (50, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)
    elif 0.4 <= density_ratio <= 0.699:
        cv2.putText(img, f'count: 7 sec', (50, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)
    elif density_ratio >= 0.7:
        cv2.putText(img, f'count: 10 sec', (50, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)
    cv2.imshow("Image", img)

    if k == ord("s"):
        break

# Convert new data to DataFrame
new_df = pd.DataFrame(new_data, columns=["Vehicles Between", "Density", "Current_Time","Time"])

# Concatenate new data with existing data, if any
if len(data) > 0:
    df = pd.concat([df, new_df], ignore_index=True)
else:
    df = new_df

# Save DataFrame to Excel file
df.to_excel("vehicle_data.xlsx", index=False)
