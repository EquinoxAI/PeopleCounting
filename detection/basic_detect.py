import jetson.inference
import jetson.utils

#The detectNet object allows you to use a deep learning model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

#Use a USB camera as input
camera = jetson.utils.videoSource("csi://0")
#Use the display as output
display = jetson.utils.videoOutput("display://0")

while True:
    #Capture a frame
    img = camera.Capture()
    #Use the model with the frame as input
    detections = net.Detect(img)
    
    #Render the frame with the detected objects
    display.Render(img)
    #Set the window title
    display.SetStatus("Network {:.0f} FPS".format(net.GetNetworkFPS()))
    #Stop if the camera or display stop streaming data
    if not camera.IsStreaming() or not display.IsStreaming():
        break