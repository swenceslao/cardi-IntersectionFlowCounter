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
    th =25
    imask =  mask > th
    canvas = np.zeros_like(frame2, np.uint8)
    canvas[imask] = frame1[imask]
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    #canvas = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
  # transforms
    fgmask = fgbg.apply(mask)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))  
    #dilation = cv2.dilate(fgmask, kernel) 
   
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    #opening = cv2.morphologyEx(closing, cv2.MORPH_ERODE, kernel)
    mask =closing
    
# variable for contours
    ret,thresh = cv2.threshold(mask,0,255,0)

# creates contours/blobs
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
# use convex hull to create polygon around contours
    hull = [cv2.convexHull(c) for c in contours]

# draw contours
    cv2.drawContours(mask, hull, -1, (0, 255, 0), 2)
    
    cxx = np.zeros(len(contours))
    cyy = np.zeros(len(contours))

# line created to stop counting contours, needed as cars in distance become one big contour
    lineypos = 400
    cv2.line(image, (-100, lineypos), (width, -120), (255, 0, 0), 3)   # blue
    lineypos2 = -700
    cv2.line(image, (-150, lineypos2), (width, 700), (0, 255, 0), 3)    # green
    cv2.line(image, (-150, -100), (width, 1800), (255, 255,0), 3)

#creating centroids and boxes
    for j in range(len(contours)): 

        if hierarchy[0, j, 3] == -1:
            cnt=contours[j]
            
            area = cv2.contourArea(cnt)
           
            if 500 < area < 50000:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#getting variables for the centroids
                cx = int(x + w/2)
                cy = int(y + h/2)
                cen = (cx,cy)
                cv2.circle(image, (cx,cy), 7, (255,0,0), -1)

                cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    .5, (0, 0, 255), 1)
                

                cxx[j] = cx
                cyy[j] = cy
                pts.appendleft(cen)
    
#this is for plotting the past centroid positions

    for i in np.arange(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
               
                # draw the centroid tracker
                cv2.circle(image, (pts[i - 1]), 2, (0,0,255), -1)


#drawing hte current centroid
   
    cxx = cxx[cxx != 0]
    cyy = cyy[cyy != 0]    
    minx_index2 = []
    miny_index2 = []
    maxrad = 30

# if there are centroids in the specified area
    if len(cxx):  
            if not carids:  # if carids is empty
                
                # loops through all centroids
                for i in range(len(cxx)):  
                    carids.append(i)  

                    df[str(carids[i])] = ""
                    df.at[int(framenumber), str(carids[i])] = [cxx[i], cyy[i]]
                   

                    totalcars = carids[i] + 1 
            else:
                        dx = np.zeros((len(cxx), len(carids))) 
                        dy = np.zeros((len(cyy), len(carids)))  

                        for i in range(len(cxx)):  

                            for j in range(len(carids)): 

                        # acquires centroid from previous frame for specific carid
                                oldcxcy = df.iloc[int(framenumber - 1)][str(carids[j])]

                        # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                                curcxcy = np.array([cxx[i], cyy[i]])

                                if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                                      continue  # continue to next carid

                                else:  # calculate centroid deltas to compare to current frame position later

                                     dx[i, j] = oldcxcy[0] - curcxcy[0]
                                     dy[i, j] = oldcxcy[1] - curcxcy[1]

                        for j in range(len(carids)):  # loops through all current car ids

                            sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                    # finds which index carid had the min difference and this is true index
                            correctindextrue = np.argmin(np.abs(sumsum))
                            minx_index = correctindextrue
                            miny_index = correctindextrue

                    # acquires delta values of the minimum deltas in order to check if it is within radius later on
                            mindx = dx[minx_index, j]
                            mindy = dy[miny_index, j]

                            if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                        # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                        # delta could be zero if centroid didn't move

                                continue  # continue to next carid

                            else:

                        # if delta values are less than maximum radius then add that centroid to that specific carid
                                if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:

                            # adds centroid to corresponding previously existing carid
                                    df.at[int(framenumber), str(carids[j])] = [cxx[minx_index], cyy[miny_index]]
                                    minx_index2.append(minx_index)  # appends all the indices that were added to previous carids
                                    miny_index2.append(miny_index)

                        for i in range(len(cxx)):  # loops through all centroids

                    # if centroid is not in the minindex list then another car needs to be added
                            if i not in minx_index2 and miny_index2:

                                df[str(totalcars)] = ""  # create another column with total cars
                                totalcars = totalcars + 1  # adds another total car the count
                                t = totalcars - 1  # t is a placeholder to total cars
                                carids.append(t)  # append to list of car ids
                                df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # add centroid to the new car id

                            elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                        # checks if current centroid exists but previous centroid does not
                        # new car to be added in case minx_index2 is empty

                                df[str(totalcars)] = ""  # create another column with total cars
                                totalcars = totalcars + 1  # adds another total car the count
                                t = totalcars - 1  # t is a placeholder to total cars
                                carids.append(t)  # append to list of car ids
                                df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # add centroid to the new car id

        # The section below labels the centroids on screen

    currentcars = 0  # current cars on screen
    currentcarsindex = []  # current cars on screen carid index

    for i in range(len(carids)):  # loops through all carids

            if df.at[int(framenumber), str(carids[i])] != '':
                # checks the current frame to see which car ids are active
                # by checking in centroid exists on current frame for certain car id

                currentcars = currentcars + 1  # adds another to current cars on screen
                currentcarsindex.append(i)  # adds car ids to current cars on screen

    for i in range(currentcars):  # loops through all current car ids on screen

            # grabs centroid of certain carid for current frame
            curcent = df.iloc[int(framenumber)][str(carids[currentcarsindex[i]])]

            # grabs centroid of certain carid for previous frame
            oldcent = df.iloc[int(framenumber - 1)][str(carids[currentcarsindex[i]])]

            if curcent:  # if there is a current centroid

                # On-screen text for current centroid
                #cv2.putText(image, "Centroid" + str(curcent[0]) + "," + str(curcent[1]),
                            #(int(curcent[0]), int(curcent[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                cv2.putText(image, "ID:" + str(carids[currentcarsindex[i]]), (int(curcent[0]), int(curcent[1] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)
                if oldcent:  # checks if old centroid exists
                    # adds radius box from previous centroid to current centroid for visualization
                    xstart = oldcent[0] - maxrad
                    ystart = oldcent[1] - maxrad
                    xwidth = oldcent[0] + maxrad
                    yheight = oldcent[1] + maxrad
                    #cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0), 1)

                  





    framenumber = framenumber + 1
    cv2.imshow('Intersection Flow Prediction',image)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
df.to_csv('grounddata1.csv', sep=',')
