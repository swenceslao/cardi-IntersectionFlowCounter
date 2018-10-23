import numpy as np
import cv2
import time 
from matplotlib import pyplot as plt
import imutils
from collections import deque
import argparse
import pandas as pd


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
    th = 10
    imask =  mask > th
    canvas = np.zeros_like(frame2, np.uint8)
    canvas[imask] = frame1[imask]
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
# convert the grayscale image to binary image
    ret,thresh = cv2.threshold(mask,0,255,0)


# creates contours/blobs
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
# use convex hull to create polygon around contours
    hull = [cv2.convexHull(c) for c in contours]

# draw contours
    cv2.drawContours(mask, hull, -1, (0, 255, 0), 3)
    
    cxx = np.zeros(len(contours))
    cyy = np.zeros(len(contours))



# line created to stop counting contours, needed as cars in distance become one big contour
    lineypos = 400
    cv2.line(image, (-100, lineypos), (width, -120), (255, 0, 0), 3)   # bluee
    #cv2.line(canvas, (-50, 100), (width, 100), (255, 0, 0), 5)
    #cv2.line(image, (-50, 700), (width, 700), (0, 0, 255), 3)   # red

# line y position created to count contours
    lineypos2 = -700
    cv2.line(image, (-150, lineypos2), (width, 700), (0, 255, 0), 3)    # green
    cv2.line(image, (-150, -100), (width, 1800), (255, 255,0), 3)
    
    for j in range(len(contours)): 

        if hierarchy[0, j, 3] == -1:
            cnt=contours[j]
            
            area = cv2.contourArea(cnt)
           
            if area > 500:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#getting variables for the centroids
                cx = x + w/2
                cy = y + h/2
                cen = (cx,cy)
                cv2.circle(image, (cx,cy), 7, (255,0,0), -1)

                cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    .5, (0, 0, 255), 1)
                

                cxx[j] = cx
                cyy[j] = cy
                pts.appendleft(cen)
    #storing centroid positions
        # eliminates zero entries (centroids that were not added)
    
    for i in np.arange(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
         
                # check to see if enough points have been accumulated in
                # the buffer
                if counter >= 10 and i == 1 and pts[-10] is not None:
                    # compute the difference between the x and y
                    # coordinates and re-initialize the direction
                    # text variables
                    dX = pts[-10][0] - pts[i][0]
                    dY = pts[-10][1] - pts[i][1]
                    (dirX, dirY) = ("", "")
         
                    # ensure there is significant movement in the
                    # x-direction
                    if np.abs(dX) > 10:
                        dirX = "East" if np.sign(dX) == 1 else "West"
         
                    # ensure there is significant movement in the
                    # y-direction
                    if np.abs(dY) > 10:
                        dirY = "North" if np.sign(dY) == 1 else "South"
         
                    # handle when both directions are non-empty
                    if dirX != "" and dirY != "":
                        direction = "{}-{}".format(dirY, dirX)
         
                    # otherwise, only one direction is non-empty
                    else:
                        direction = dirX if dirX != "" else dirY
                    # otherwise, compute the thickness of the line and
             

                # draw the centroid tracker
                cv2.circle(image, (pts[i - 1]), 2, (0,0,255), -1)


              

            







    cv2.imshow('Intersection Flow Prediction',image)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()