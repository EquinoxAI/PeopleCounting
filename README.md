# People counting on Jetson Nano
![out](https://user-images.githubusercontent.com/107493543/195707863-c246c4b4-bcde-4d65-a986-60d0d051bb01.gif)

In this repo you can find the codes, to implement a monitoring system in Jetson Nano that tracks the flow of people using digital image processing techniques and deep learning. In addition, you can review our posts [People counting on Jetson Nano](https://asesoftware-my.sharepoint.com/personal/ndiaz_asesoftware_com/Documents/WhoAmEye/Art%C3%ADculo4-Resultados.docx?web=1).

## Hardware and software used
For the development of this monitoring system, we used a *Jetson Nano Developer Kit A01*, the video input signal is acquired through a *Logitech HD Pro C920 webcam* with 1080p resolution, also this monitoring algorithms were implemented in Python 3.6 using OpenCV and Jetson Utils. The information about the NVIDIA Jetson environment is in the following table.

| **Software**          | **Version**   |
| --------------------- | ------------- |
| **L4T 32.4.4**        | JetPack 4.4.1 |
| **Ubuntu**            | 18.04.5 LTS   |
| **Kernel**            | 4.9.140-tegra |
| **CUDA**              | 10.2.89       |
| **CUDA Architecture** | 5.3           |
| **OpenCV**            | 4.5.0         |
| **OpenCV Cuda**       | YES           |
| **CUDNN**             | 8.0.0.180     |
| **TensorRT**          | 7.1.3.0       |
| **Vision Works**      | 1.6.0.501     |
| **VPI**               | 4.4.1-b50     |
| **Vulcan**            | 1.2.70        |

*Note: If your OpenCV version is different or you are experiencing problems, you can check the following [tutorial](https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html) to install the version used in this implementation.*

## Control and configuration
Control and configuration of the Jetson Nano were done through a remote connection, using VNC Viewer and the VS-code Remote-SSH extension. In our post [*Getting started with Jetson Nano and Jetson Inference*](https://asesoftware-my.sharepoint.com/personal/ndiaz_asesoftware_com/Documents/WhoAmEye/Artículo1-Instalación%20y%20configuración.docx), you will find all the steps needed to configure both tools, as well as the basic installation and configuration of Jetson Inference, a library specifically developed to implement deep learning models on Jetson Nano.

## Detection and tracking
We can divide this application into two main tasks: people detection and object tracking. First, we use a deep learning model and digital image processing techniques to detect moving people and get their location in a particular frame. We then save the detection’s bounding boxes and centroids and then keep track of their movement over the following frames using object tracking algorithms, as illustrated in the following figure. In addition, you can review our posts about detection [*People detection using Jetson Inference and OpenCV*](https://asesoftware-my.sharepoint.com/personal/ndiaz_asesoftware_com/Documents/WhoAmEye/Art%C3%ADculo2-Detecci%C3%B3n.docx?web=1) and tracking [*Object tracking and counting using Python*](https://asesoftware-my.sharepoint.com/personal/ndiaz_asesoftware_com/Documents/WhoAmEye/Art%C3%ADculo3-Tracking.docx?web=1) to develop this **people counting system**.

 ![detec_tracking](https://user-images.githubusercontent.com/107493543/195665143-c33c2866-c0c9-438e-a107-5bdb0bc8456d.jpg)

