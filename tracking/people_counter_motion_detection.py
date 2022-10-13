# Import the necessary packages
import cv2
import imutils
from object_tracker import Tracker

#read video input
#camera = cv2.VideoCapture("./input_real/usb_cam/1person_usb.mp4")

# read video input signal USB camera
camera= cv2.VideoCapture(0)

# Function to detect moving objects
def detect_motion(img, background):

    # Smoothing of the image
    img = cv2.GaussianBlur(img, (9, 9), 0)
    background = cv2.GaussianBlur(background, (9, 9), 0)

    # Compute the absolute difference between the current frame and background
    resta = cv2.absdiff(background, img)

    # Apply threshold
    thresh = cv2.threshold(resta, 50, 255, cv2.THRESH_BINARY)[1]
 
    # Apply morphological transformations to remove noise and fill holes 
    thresh = cv2.dilate(thresh, None, iterations=1)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, None)

    # Find contours
    contours,_ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Initialize variables
background = None
max_disappeared = 10
max_distance = 200
max_history = 10
limit = None
W = None
H = None
tracker = Tracker(limit, max_disappeared, max_distance, max_history)

# Loop over the frames of the video
while True:
    
    # Capture frame by frame
    (grabbed, frame) = camera.read()

    # Resize the frame
    frame = imutils.resize(frame, width=400)

    # Get frame shape and update the limit
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        posLineY = H-80
        tracker.limit = [(0,posLineY), (W, posLineY)]

    # Break if reached the end of the video
    if not grabbed:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the first frame this is the background
    if background is None:
        background = gray
        continue

    # Detect moving objects
    contours = detect_motion(gray, background)

    # Initialize detection array
    people_data = []

    # Loop over found contours
    for c in contours:

        # Filter objects that have a certain number of pixels
        if cv2.contourArea(c) > 7000 and cv2.contourArea(c) < 24000:
            # Compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)

            # Filter objects with a certain width
            if w >=100 and w <= 200:
                
                # Compute the upper-left corner and the lower-right corner 
                # of the bounding box
                startX = x; startY = y; endX = x+w; endY = y+h
                bbox = [int(startX),int(startY), int(endX), int(endY)]

                # Compute the centroid
                cx = int((2*x + w) / 2.0)
                cy = int((2*y + h) / 2.0)
                centroid = (cx, cy)

                # Add centroids and bounding boxes
                people_data.append((centroid, bbox))
    
    # Update tracker object
    tracker.update(people_data)

    # Draw bounding boxes 
    for object_id in list(tracker.centroids.keys()):
        x1,y1,x2,y2 = tracker.bboxes[object_id]
        cv2.rectangle(frame, (x1,y1), (x2,y2),(0, 255, 0), 2)

        # Draw and centroids
        for centroid in tracker.centroids[object_id]:
            cv2.circle(frame, (centroid[0], centroid[1]), 4,(int(tracker.colors[object_id][0]), int(tracker.colors[object_id][1]), int(tracker.colors[object_id][2])), -1)

        
        # Draw  the ID of the object
        text = "ID {}".format(object_id)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(tracker.colors[object_id][0]), int(tracker.colors[object_id][1]), int(tracker.colors[object_id][2])), 2)
        
    
    # Draw the limit
    cv2.line(frame, (0,posLineY), (W, posLineY), (0, 255, 255), 2)


    # Count people
    tracker.count_people()

    # Draw count   
    cv2.putText(frame,  f"In: {tracker.people_in}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame,  f"Out: {tracker.people_out}", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame with the moving objects detected
    cv2.imshow("Detect motion", frame)

    # If the 'q' key is pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
 
# Release the camera and close all the windows
camera.release()
cv2.destroyAllWindows()