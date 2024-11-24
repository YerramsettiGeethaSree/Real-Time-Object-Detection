import cv2
import math
import time
from ultralytics import YOLO

# Start webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)  # Width
video_capture.set(4, 480)  # Height

# Load YOLO model
object_detection_model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes (ensure it matches the model classes)
object_classes = ["Person", "Bicycle", "Car", "Motorbike", "Aeroplane", "Bus", "Train", "Truck", "Boat",
                  "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat",
                  "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella",
                  "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat",
                  "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle", "Wine Glass", "Cup",
                  "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli",
                  "Carrot", "Hot Dog", "Pizza", "Donut", "Cake", "Chair", "Sofa", "Pottedplant", "Bed",
                  "Diningtable", "Toilet", "Tvmonitor", "Laptop", "Mouse", "Remote", "Keyboard", "Cell Phone",
                  "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors",
                  "Teddy Bear", "Hair Drier", "Toothbrush"
                  ]

# Set confidence threshold for detection
confidence_threshold = 0.5

# Initialize FPS calculation
prev_time = 0
show_bounding_boxes = True  # Variable to toggle bounding boxes

# Define a color palette for different object classes (excluding the first color)
colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), 
          (128, 128, 0), (128, 0, 128), (0, 128, 128), (255, 165, 0), (75, 0, 130), 
          (240, 230, 140)]

# Define the main color for the first object
main_color = (205, 170, 145)
text_color = (115, 57, 15)

while True:
    is_successful, frame = video_capture.read()
    if not is_successful:
        break

    # YOLO object detection
    detection_results = object_detection_model(frame, stream=True)

    # Current time for FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    for result in detection_results:
        detected_boxes = result.boxes
        first_object_detected = True  # Flag to indicate the first object

        for detected_box in detected_boxes:
            # Get bounding box coordinates and convert them to integers
            box_x1, box_y1, box_x2, box_y2 = map(int, detected_box.xyxy[0])

            # Get confidence score
            confidence_score = detected_box.conf[0].item()
            if confidence_score < confidence_threshold:
                continue

            # Get class index and handle possible out-of-range values
            class_index = int(detected_box.cls[0].item())
            class_name = object_classes[class_index] if class_index < len(object_classes) else "Unknown"

            # Choose color for the bounding box
            if first_object_detected:
                color = main_color  # Use main color for the first detected object
                first_object_detected = False  # Set flag to False after the first object
            else:
                color_index = class_index % len(colors)  # Cycle through colors for subsequent objects
                color = colors[color_index]

            # Draw bounding box if the toggle is on
            if show_bounding_boxes:
                # Draw bounding box with the chosen color
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, 3)

                # Add label and confidence score to bounding box with original text settings
                label = f"{class_name}: {confidence_score:.2f}"
                text_position = (box_x1, box_y1 - 10)
                cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # Display FPS on the screen in light lavender color
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 230, 250), 2)  # Light lavender color for FPS

    # Show the frame with object detections
    cv2.imshow('Webcam', frame)

    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord('b'):
        show_bounding_boxes = not show_bounding_boxes  # Toggle bounding boxes
    elif key == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
