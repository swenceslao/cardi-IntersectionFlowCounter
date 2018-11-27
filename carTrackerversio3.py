import numpy as np
import cv2
import time 
from matplotlib import pyplot as plt
import imutils
from collections import deque
import argparse
import pandas as pd
import random
from math import atan2,degrees

# ============== GLOBAL VARIABLES
#   setting variables before the image processing
cap = cv2.VideoCapture("cardi.MP4")
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=5000,help="max buffer size")
args = vars(ap.parse_args())
fgbg = cv2.createBackgroundSubtractorMOG2()  # create background subtracto
pts = deque(maxlen=args["buffer"])# where the centroids will be stored
mk = deque(maxlen=args["buffer"])
lastcoord = deque(maxlen=args["buffer"])
theta = np.radians(42)
xx, yy = np.cos(theta), np.sin(theta)
R = np.array(((xx,-yy), (yy, xx)))
df = pd.DataFrame(index=range(int(frames_count)))
df.index.name = "Frames"

df2 = pd.DataFrame(columns=['carid','startx', 'starty','sframe','endx', 'eframe' ,'endy'])


framenumber = 0  # keeps track of current frame
carids = []  # blank list to add car ids
totalcars = 0  # keeps track of total cars


def GetAngleOfLineBetweenTwoPoints(p1, p2):
        xDiff = p2.x - p1.x
        yDiff = p2.y - p1.y
        return degrees(atan2(yDiff, xDiff))


def counter(mk,lastcoord):
    for i in np.arange(1, len(mk)):
            if mk[i - 1] is None or mk[i] is None:
                    continue
            cid,fno = mk[i - 1]
            start,y = df.iloc[int(fno)][str(cid)]
            
            df2.at[i,'carid'] = cid
            df2.at[i,'sframe'] = fno 
            df2.at[i,'startx'] = start
            df2.at[i,'starty'] = y

          
                



                # draw the centroid tracker
                
# mark the framenumber once a new carid is assigned , then , copy centroid
# remember the carid and theframe number it once started 
# loop through the framenumber wth the id until end
# get the framenumber of last record, get the centroid

#def prediction():



#def chartGen():


def trackCarCent(pts, image):
    #this is for plotting the past centroid positions

    for i in np.arange(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                x,y = pts[i - 1]
                # draw the centroid tracker
                cv2.circle(image,(int(x), int(y)), 2, (0,0,255), -1)


def getDist(frame1 ,frame2):
    image1 = np.float32(frame1)
    image2 = np.float32(frame2)
    diff = image1-image2
    norm = np.sqrt(diff[:,:,0]**2 + diff[:,:,1]**2+diff[:,:,2]**2) / np.sqrt(255**2+ 255**2 + 255**2)
    dist = np.uint8(norm*255)
    return dist

#works with the dist
def morphTrans(image):
    blur = cv2.GaussianBlur(dist,(9,9),0)
    #grayIm = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th = 10
    imask =  blur > th
    canvas = np.zeros_like(frame1, np.uint8)
    canvas[imask] = frame[imask]
    grayIm = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
    dilation = cv2.dilate(canvas, kernel) 
    opening = cv2.morphologyEx(dilation, cv2.MORPH_ERODE, kernel)
    mask =closing
    return mask

#works with basic transform
def morphTrans1(image):
     
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
     
    
    fgmask = fgbg.apply(image)
    blur = cv2.GaussianBlur(fgmask,(9,9),0)
    fgmask = fgbg.apply(blur)
    
    
    mask = fgmask
    return mask

def drawContour(mask, image):
    

    ret,thresh = cv2.threshold(mask,0,255,0)
    # creates contours/blobs
    __, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    # use convex hull to create polygon around contours
    hull = [cv2.convexHull(c) for c in contours]
    # draw contours
    cv2.drawContours(mask, hull, -1, (0, 255, 0), 2)

#to store centroids    
    cxx = np.zeros(len(contours))
    cyy = np.zeros(len(contours))

    for j in range(len(contours)): 

            if hierarchy[0, j, 3] == -1:
                cnt=contours[j]
                
                area = cv2.contourArea(cnt)
               
                if 2500 < area < 10000:
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

    return image , cxx ,cyy

def getCent(cxx,cyy,carids,df,totalcars,framenumber):
    cxx = cxx[cxx != 0]
    cyy = cyy[cyy != 0]    
    minx_index2 = []
    miny_index2 = []
    maxrad =50
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
            #improve here for kN
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
                                    carStart = (t,framenumber)
                                    mk.appendleft(carStart)
                                    df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # add centroid to the new car id

                                elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                                    df[str(totalcars)] = ""  # create another column with total cars
                                    totalcars = totalcars + 1  # adds another total car the count
                                    t = totalcars - 1  # t is a placeholder to total cars
                                    carids.append(t)  # append to list of car ids
                                    carStart = (t,framenumber)
                                    mk.appendleft(carStart)

                                    df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # add centroid to the new car id
    return totalcars, carids

def carIDcounter(image,carids, framenumber,df,currentcars,currentcarsindex):
    
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
                cv2.putText(image, "ID:" + str(carids[currentcarsindex[i]]), (int(curcent[0]), int(curcent[1] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)
               

               # cv2.putText(image, "CarID in the area: ", (0, 115 ), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
               #cv2.putText(image, "  " + str(carids[currentcarsindex[i]]), (5+ (50*i) , 150), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
    return df , carids, currentcars,currentcarsindex
                

def textDisp(framenumber,image,currentcars,carids,fps,frames_count):
    cv2.rectangle(image, (0, 0), (1500, 100), (0,0,0), -1)  # background rectangle for on-screen text

    cv2.putText(image, "Cars in Area: " + str(currentcars), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
    cv2.putText(image, "Total Cars Detected: " + str(len(carids)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255,255,255), 1)
    cv2.putText(image, "Frame: " + str(framenumber) + ' of ' + str(frames_count), (0, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255,255,255), 1)
    cv2.putText(image, 'Time: ' + str(round(framenumber / fps, 2)) + ' sec of ' + str(round(frames_count / fps, 2))
                    + ' sec', (0, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)


    cv2.putText(image, "Total Cars turned N: " , (251, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
    cv2.putText(image, "Will go to N: " , (251, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255,255,255), 1)
   

    cv2.putText(image, "Total Cars turned S: " , (501, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
    cv2.putText(image, "Will go to S:  " , (501, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255,255,255), 1)


    cv2.putText(image, "Total Cars turned L: " , (751, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
    cv2.putText(image, "Will go to L: " , (751, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,255,255), 1)
   

    cv2.putText(image, "Total Cars turned R: " , (1001, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
    cv2.putText(image, "Will go to R:  " , (1001, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255,255,255), 1)
#================ MAIN ==================================

_ , frame1 = cap.read()
_ , frame2 = cap.read()
    #capturing data

while(True):

# Capture two frames
    _ , frame1 = cap.read()
    time.sleep(1/25)          # slight delay
    _ , frame2 = cap.read()

#resizing
    image = cv2.resize(frame1, (0, 0), None, 1,1)
    #image2 = cv2.resize(frame2, (0, 0), None, .3,.3)

#background subtraction
     
    diff = cv2.absdiff(frame1,frame2)  
    canvas = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask = morphTrans1(canvas)
    

    im, cxx , cyy = drawContour(mask, image)
    trackCarCent(pts,image)
    totalcars,carids=  getCent(cxx,cyy,carids,df,totalcars,framenumber)

    currentcars = 0  # current cars on screen
    currentcarsindex = []  # current cars on screen carid index

   # carIDcounter(carids, framenumber,df,currentcars,currentcarsindex)    
    df , carids, currentcars,currentcarsindex = carIDcounter(image,carids, framenumber,df,currentcars,currentcarsindex)
    textDisp(framenumber,image,currentcars,carids,fps,frames_count)

    counter(mk,lastcoord)
    framenumber = framenumber + 1
    

  


#showing the video transformation
    cv2.imshow('Morph',mask)
    cv2.imshow('Track Movements',im)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

df.to_csv('current.csv', sep=',')
df2.to_csv('dummy.csv', sep=',')
cap.release()
cv2.destroyAllWindows()
print(lastcoord)

