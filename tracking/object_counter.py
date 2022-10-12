from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, cudaDrawCircle, cudaDrawLine, cudaFont
from object_tracker import Tracker

# Model parameters
model = "ssd-inception-v2"
model_threshold = 0.5
# Tracker parameters
max_disappeared = 10
max_distance = 200
max_history = 10
limit = [(0,425),(1280,425)]
# System parameters
fps_detection_rate = 5

# The sources can be videos or cameras:
#   cap = videoSource("csi://0")
#   cap = videoSource("video.mp4")
cap = videoSource("/dev/video1")

# The output can be a file or display:
#   display = videoOutput("out.mp4")
display = videoOutput("display://0")

# Initialize the cudaFont
font = cudaFont()

# Initialize the tracker object
tracker = Tracker(limit, max_disappeared, max_distance, max_history)
# Initialize the model
net = detectNet(model, threshold=model_threshold)

# Main loop - initialize the count so the first frame is processed
fps_count = fps_detection_rate
while True:
    # Capture a frame
    img = cap.Capture()
    # Detect and track
    if fps_count == fps_detection_rate:
        fps_count = 0
        detections = net.Detect(img, overlay = "OVERLAY_NONE")
        people_detections = [d for d in detections if d.ClassID == 1]
        people_data = []
        for d in people_detections:
            bbox = [int(d.Left),int(d.Top), int(d.Right), int(d.Bottom)]
            centroid = (int(d.Center[0]), int(d.Center[1]))
            people_data.append((centroid, bbox))
        tracker.update(people_data)

    # Draw bounding boxes and centroids
    for object_id in list(tracker.centroids.keys()):
        x1,y1,x2,y2 = tracker.bboxes[object_id]
        cudaDrawLine(img, (x1,y1), (x2,y1), tracker.colors[object_id], 2)
        cudaDrawLine(img, (x2,y1), (x2,y2), tracker.colors[object_id], 2)
        cudaDrawLine(img, (x2,y2), (x1,y2), tracker.colors[object_id], 2)
        cudaDrawLine(img, (x1,y2), (x1,y1), tracker.colors[object_id], 2)

        for centroid in tracker.centroids[object_id]:
            cudaDrawCircle(img, (int(centroid[0]), int(centroid[1])), 3, tracker.colors[object_id])

        font.OverlayText(img, 50, 50, f"ID {object_id}", x1-10, y1-10, font.White, font.Gray40)
    
    # Draw the limit
    cudaDrawLine(img, limit[0], limit[1], (0,255,0,255), 3)

    # Count people
    tracker.count_people()
    font.OverlayText(img, img.width, img.height, f"In: {tracker.people_in}", 5, 5, font.White, font.Gray40)
    font.OverlayText(img, img.width, img.height, f"Out: {tracker.people_out}", 5, 55, font.White, font.Gray40)

    # Render the result
    display.Render(img)
    display.SetStatus("{:s} | Network {:.0f} FPS".format("Mobilenet", net.GetNetworkFPS()))
    fps_count += 1

    if not cap.IsStreaming() or not display.IsStreaming():
        break
