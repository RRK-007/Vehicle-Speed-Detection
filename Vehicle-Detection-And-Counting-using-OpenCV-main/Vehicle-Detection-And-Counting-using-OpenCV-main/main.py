import cv2
import os
import time
import math
import pygame

class AdvancedEuclideanDistTracker:
    def __init__(self, max_frames_missing=5, distance_threshold=25, velocity_threshold=10):
        self.center_points = {}
        self.id_count = 0
        self.max_frames_missing = max_frames_missing
        self.distance_threshold = distance_threshold
        self.velocity_threshold = velocity_threshold

    def update(self, obj_rect):
        obj_bbx_ids = []

        # Update existing object positions
        for obj_id, (prev_center, velocity, missing_frames) in list(self.center_points.items()):
            if missing_frames >= self.max_frames_missing:
                # Remove object if it's been missing for too many frames
                del self.center_points[obj_id]
                continue

            # Try to find a matching detection for this object
            match_found = False
            min_dist = float('inf')
            closest_rect = None
            for rect in obj_rect:
                x, y, w, h = rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_rect = rect

            if closest_rect and min_dist < self.distance_threshold:
                # Update object position, velocity, and reset missing frames counter
                x, y, w, h = closest_rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                updated_velocity = (cx - prev_center[0], cy - prev_center[1])
                if math.hypot(updated_velocity[0], updated_velocity[1]) < self.velocity_threshold:
                    self.center_points[obj_id] = ((cx, cy), updated_velocity, 0)
                    obj_bbx_ids.append([x, y, w, h, obj_id])
                    match_found = True

            # If no match found, increment missing frames counter
            if not match_found:
                self.center_points[obj_id] = (prev_center, velocity, missing_frames + 1)

        # Assign IDs to new objects
        for rect in obj_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            matched = False
            for obj_id, (prev_center, _, _) in self.center_points.items():
                dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
                if dist < self.distance_threshold:
                    matched = True
                    break
            if not matched:
                self.center_points[self.id_count] = ((cx, cy), (0, 0), 0)
                obj_bbx_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        return obj_bbx_ids

# Function to calculate speed in km/hr
def calculate_speed(distance_px, time_interval, pixel_to_km_ratio):
    # Check if time interval is zero or very small
    if time_interval <= 0.001:
        return 0
    
    # Convert distance to kilometers using pixel-to-km ratio
    distance_km = distance_px * pixel_to_km_ratio
    
    # Convert time interval to hours
    time_hours = time_interval / 3600  # 1 hour = 3600 seconds
    
    # Calculate speed in km/hr
    speed_kmh = distance_km / time_hours
    
    return speed_kmh

# Create background subtractor
obj_det = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Specify whether to use webcam or video file
webcam_is_using = False

# Directory path where the video file is located
directory_path = r"C:\Users\Rajiv Ranjan Kumar\Desktop\Vehicle-Detection-And-Counting-using-OpenCV-main\Vehicle-Detection-And-Counting-using-OpenCV-main"
# Video file name
video_file_name = "highway.mp4"

# Full path to the video file
video_path = os.path.join(directory_path, video_file_name)

# Pixel-to-kilometer ratio (adjust according to your video)
pixel_to_km_ratio = 0.1  # Example: 1 pixel = 0.1 km

# Initialize previous centroid, timestamp, and distance
prev_centroids = {}

# Initialize advanced object tracker
tracker = AdvancedEuclideanDistTracker(max_frames_missing=10, distance_threshold=50, velocity_threshold=20)

# Initialize Pygame mixer
pygame.mixer.init()

# Get the directory of the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the alert.mp4 file relative to the script directory
alert_mp4_path = os.path.join(script_directory, "alert.mp4")

# Load the alert sound
alert_sound = pygame.mixer.Sound(alert_mp4_path)

# Initialize video capture
cap = cv2.VideoCapture(0) if webcam_is_using else cv2.VideoCapture(video_path)

# Check if video capture is successful
if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

# Get video frame rate
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Set a target frame rate for video playback
target_frame_rate = 50 # Adjust as needed

# Create a separate frame for displaying speed
speed_frame = None

# Main loop
while True:
    # Read frame from the video source
    ret, frame = cap.read()

    # Check if frame is successfully read
    if not ret:
        print("End of video.")
        break

    # Apply object detection to a wider region of interest (ROI)
    roi = frame[100:1000, 100:2000]  # Adjusted ROI coordinates
    mask = obj_det.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust minimum contour area
            x, y, w, h = cv2.boundingRect(contour)
            # Expand bounding boxes by 20 pixels on each side
            x -= 20
            y -= 20
            w += 40
            h += 40
            detections.append([x, y, w, h])

    # Update tracker with detections
    boxes_ids = tracker.update(detections)
    
    # Draw bounding boxes, IDs, and speeds on the frame
    for box in boxes_ids:
        x, y, w, h, obj_id = box
        centroid_x = x + w // 2
        centroid_y = y + h // 2
        
        # Calculate speed if previous centroid exists
        if obj_id in prev_centroids:
            prev_x, prev_y, prev_time_stamp = prev_centroids[obj_id]
            distance_px = ((centroid_x - prev_x) ** 2 + (centroid_y - prev_y) ** 2) ** 0.5
            time_interval = time.time() - prev_time_stamp
            speed_kmh = calculate_speed(distance_px, time_interval, pixel_to_km_ratio)
            
            # Check if speed exceeds 40 km/hr
            if speed_kmh > 220:
                alert_sound.play()
            
            # Extract only the value after the decimal point
            decimal_part = "{:.2f}".format(speed_kmh).split('.')[1]
            
            # Display speed on the screen
            cv2.putText(roi, f"Speed: {decimal_part} km/hr", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Print speed in terminal
            print(f"Object ID: {obj_id}, Speed: {speed_kmh:.2f} km/hr")
        
        # Update previous centroid and timestamp
        prev_centroids[obj_id] = (centroid_x, centroid_y, time.time())

        cv2.putText(roi, f"ID: {obj_id}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), )
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Display frames
    cv2.imshow("mask", mask)
    cv2.imshow("roi", roi)
    cv2.imshow("frame", frame)

    # Create a separate frame for speed display
    if speed_frame is None:
        speed_frame = frame.copy()
    
    # Delay frame to achieve target frame rate
    delay_time = int(1000 / target_frame_rate)  # milliseconds
    key = cv2.waitKey(delay_time)

    # Check for exit key
    if key == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
