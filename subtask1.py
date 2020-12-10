import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.pylab as pylab
import os
import sys
import matplotlib.patches as patches
import ast

cascade_name =cv2.CascadeClassifier('samples/dartcascade/cascade.xml')

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# tp,fp,fn
metrics = [0,0,0]

    # Viola Jones dartboard detector
def VJ_detector(image, obj_classifier):
    boxes = []
    #run classifier
    obj = obj_classifier.detectMultiScale(image, 1.1, 1, 0|cv2.CASCADE_SCALE_IMAGE, (50,50), (500,500))
    for (x,y,w,h) in obj:
        boxes.append([x,y,x+w,y+h])
    return boxes

def rectangle_intersection(box1, box2):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # compute the area of intersection rectangle
    intersec_area = (x2 - x1) * (y2 - y1)
    # compute the area of both the viola and hough rectangles
    box1_area = (box1[2] - box1[0] + 1 ) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersec_area / float(box1_area + box2_area - intersec_area)
    # return the intersection over union value
    return iou

    # Calculate the true positive rate from [TP, FP, FN]
def tpr(metrics):
    tp = metrics[0]
    fn = metrics[2]  
    if tp+fn == 0 or tp==0:
        return 0
    tpr = metrics[0]/(metrics[0] + metrics[2])
    return tpr

# Calculate the F1-score from [TP, FP, FN]
def f1score(metrics):
    tp = metrics[0]
    fp = metrics[1]
    fn = metrics[2]
    if tp == 0:
        return 0
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1score = 2*((precision*recall)/(precision+recall))
    return f1score

def eval(box1,box2,iou,threshold=0.4):
    if iou>=threshold:
        # true positive
        metrics[0]+=1
    else: 
        #  false positive
        metrics[1]+= 1    

#  if false  inner condition is false then they overlap 
def check_if_intersect(box1,box2):
    return not(box2[0] > box1[2] \
        or box2[2] < box1[0] \
        or box2[1] > box1[3] \
        or box2[3] < box1[1])

def main():
    file = open("annotated_dartboards.txt", "r")
    contents = file.read()
    # dictionary of all the ground truths 
    dictionary = ast.literal_eval(contents)
    file.close()
    frame = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
    groundtruth = dictionary[sys.argv[1]]    
    #  #convert to gray scale
    frame_grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     #swap to RGB
    frame_grey = convertToRGB(frame_grey)
     # run the classifier 
    frames_rects = cascade_name.detectMultiScale(frame_grey,1.1, 1, 0|cv2.CASCADE_SCALE_IMAGE, (50, 50), (500,500) )
    boxes1 = VJ_detector(frame,cascade_name)
    boxes2 = groundtruth
    print(boxes2)
    for b1 in range(len(boxes1)):
        for b2 in range(len(boxes2)):
            x0,y0,x1,y1 = boxes1[b1]
            x2,y2,x3,y3 = boxes2[b2]
            cv2.rectangle(frame,(x0,y0),(x1,y1),(0,255,0),3)
            cv2.rectangle(frame,(x2,y2),(x3,y3),(0,0,255),3)
            if(check_if_intersect(boxes1[b1],boxes2[b2]) ): 
                 iou = rectangle_intersection(boxes1[b1],boxes2[b2])
                 eval(boxes1[b1],boxes2[b2],iou)
             # classifier does not intersect with ground truth ( false negative)
            else: 
                metrics[2] +=1  
    


    true_positive_rate = tpr(metrics)
    f1= f1score(metrics)
    print('dart board 11')
    print('dart board true positive :',metrics[0])
    print('dart board false positive :',metrics[1])
    print('dart board false negative :',metrics[2])
    print('dart board True positive rate:',true_positive_rate)
    print('dart board f1 score:',f1)
    cv2.imwrite('dartboard-tested/dart11tested.jpg',frame)
    


if __name__ == "__main__": 
    main()
