Packages needed
-----------------

1. OpenCV
2. python flann library

USAGE:
-------

Open Main1.py and specify the path for the object to be tracked

There are two settings now

1. Track in a recorded video
2. Track from webcam feed
  2.1. Uncomment line 30 and 33 for tracking on a recorded video
  2.2.  Uncomment line 31 and 34 for tracking on live webcam feed
3. Tracked Images are recored in the directory one above the present directory 
4. Re Initialised images are recorded in the same directory
   4.1. Reinitialised images have the keypoints that are used for RANSAC calculation 
   4.2. The reinitialised Bounding Box