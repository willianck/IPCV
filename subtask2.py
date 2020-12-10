import cv2
import numpy as np
import matplotlib.pyplot as pyplot
import math
import matplotlib.pylab as pylab
import os
import sys

cascade_name =cv2.CascadeClassifier('samples/dartcascade/cascade.xml')

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Viola Jones dartboard detector

def main():
    #load image
    frame = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
     #convert to gray scale
    frame_grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     #swap to RGB
    frame_grey = convertToRGB(frame_grey)
     # run the classifier 
    #  frames_rects = cascade_name.detectMultiScale(frame_grey,scaleFactor=1.1,minNeighbors =10,minSize=(50,50),maxSize=(500,500),flags=cv2.CASCADE_SCALE_IMAGE)
    frames_rects = cascade_name.detectMultiScale(frame_grey,1.1, 1, 0|cv2.CASCADE_SCALE_IMAGE, (50, 50), (500,500) )
     #visualize the classifier 
    for (x,y,w,h) in frames_rects:
          print(x,y,x+w,y+h)
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imwrite('dart_classified.jpg',frame)


if __name__ == "__main__": 
    main()

     