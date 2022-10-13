# PeopleCounting

Control and configuration of the Jetson Nano were done through a remote connection, using VNC Viewer and the VS-code Remote-SSH extension. In our post [LGetting started with Jetson Nano and Jetson Inference](https://asesoftware-my.sharepoint.com/personal/ndiaz_asesoftware_com/Documents/WhoAmEye/Artículo1-Instalación%20y%20configuración.docx), you will find all the steps needed to configure both tools, as well as the basic installation and configuration of Jetson Inference, a library specifically developed to implement deep learning models on Jetson Nano.

Detection and tracking
We can divide this application into two main tasks: people detection and object tracking. First, we use a deep learning model and digital image processing techniques to detect moving people and get their location in a particular frame. We then save the detection’s bounding boxes and centroids and then keep track of their movement over the following frames using object tracking algorithms, as illustrated in Fig. 3. In the following section, we will implement our two posts about detection People detection using Jetson Inference and OpenCV and tracking Object tracking and counting using Python to develop a people counting system.

 ![detec_tracking](https://user-images.githubusercontent.com/107493543/195665143-c33c2866-c0c9-438e-a107-5bdb0bc8456d.jpg)

Fig. 3. People counting process.
