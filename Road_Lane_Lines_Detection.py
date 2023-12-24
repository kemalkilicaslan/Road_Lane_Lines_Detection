# Road Lane Lines Detection

# Import the necessary libraries
import cv2
import numpy as np
import os

# Function to detect lane lines in a frame
def detect_lane_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect edges detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Get height and width of the frame
    height, width = frame.shape[:2]
    
    # Create a Region Of Interest(ROI)
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2, height // 2)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    
    # Apply the mask to the edges to focus on the Region Of Interest(ROI)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)
    
    # Draw detected frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    return frame

video_capture = cv2.VideoCapture("Road_Lane_Lines.mp4")

# Video features
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Name and directory of the file to be saved
output_filename = 'Detected_Road_Lane_Lines.mp4'
output_path = os.path.join(os.path.dirname(__file__), output_filename)

output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while video_capture.isOpened():
    ret, frame = video_capture.read()  # read video frame
    
    if not ret:
        break
    
    # Detect lane lines
    detected_frame = detect_lane_lines(frame)
    
    # Display detected lane lines
    cv2.imshow('Detected_Road_Lane_Lines', detected_frame)

    # Write the drawn frame to the video file to be saved
    output_video.write(detected_frame)

    # Switch off video when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all open windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
