import numpy as np
import cv2
import time 
from matplotlib import pyplot as plt
import imutils
from collections import deque
import argparse
import pandas as pd
import random


cap = cv2.VideoCapture('cardi.MP4')
ap = argparse.ArgumentParser()

#arguments to start with
ap.add_argument("-b", "--buffer", type=int, default=5000,
    help="max buffer size")
args = vars(ap.parse_args())

# create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()  

# where the centroids will be stored
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

#setting variables before the image processing
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)

print(frames_count, fps, width, height)

# creates a pandas data frame with the number of rows the same length as frame count
df = pd.DataFrame(index=range(int(frames_count)))
df.index.name = "Frames"



framenumber = 0  # keeps track of current frame
carids = []  # blank list to add car ids
totalcars = 0  # keeps track of total cars


#capturing data
while(True):

        



# Capture two frames
    ret, frame1 = cap.read()  # first image
    
    time.sleep(1/25)          # slight delay
    ret, frame2 = cap.read()  # second image 
    image = cv2.resize(frame1, (0, 0), None, 1,1) 



#getting the difference as the basis for movement
    diff = cv2.absdiff(frame1,frame2)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th = 17
    imask =  diff > th
    canvas = np.zeros_like(frame2, np.uint8)
    canvas[imask] = frame1[imask]
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    #canvas = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    fgmask = fgbg.apply(mask)  # uses the background subtraction

        # applies different thresholds to fgmask to try and isolate cars
        
        # just have to keep playing around with settings until cars are easily identifiable
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))  # kernel to apply to the morphology
    
    
    
    
    dilation = cv2.dilate(fgmask, kernel)  
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
  
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
   # c = cv2.morphologyEx(d, cv2.MORPH_CLOSE, kernel)

    im =closing

   





    framenumber = framenumber + 1
    cv2.imshow('Intersection Flow Prediction',im)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
df.to_csv('grounddata.csv', sep=',')
