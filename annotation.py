import cv2 
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.pylab as pylab
import sys
import os
import csv 


dict = {}
anot_vals = []
pts = []
frame = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
draw = False
start = False
ix ,iy = -1, -1 

def annotate(event,x,y,flags,params):
    global draw, ix,iy,pts,anots_vals
    # Store the height and width of the image
    width = frame.shape[0]
    height = frame.shape[1]
    #click if you want to label an image
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        ix,iy = x,y
        pts.append(int(ix))
        pts.append(int(iy))
    elif event == cv2.EVENT_MOUSEMOVE:  
        if draw == True: 
            print('drawing')     
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        cv2.rectangle(frame,(ix,iy),(x,y),(0,255,0),3)
        pts.append(int(x))
        pts.append(int(y))   
        anot_vals.append(pts)
        pts= []


cv2.namedWindow('Image')
cv2.setMouseCallback('Image',annotate)
while True:
    cv2.imshow('Image',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break 
dict[sys.argv[1]] = anot_vals    
print(dict)
f = open("test.txt","a")
f.write( str(dict) + "\n" )
f.close()

cv2.destroyAllWindows()
# if __name__ == "__main__": 
#     main()    
