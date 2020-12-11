#%%
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import os
import sys
import time
from collections import defaultdict

#%%
cascade_name =cv2.CascadeClassifier('samples/dartcascade/cascade.xml')

#  helper functions 
#%%
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1  step is noise reduction we can use the cv2 function gaussian blur kernel to do so 
def gaussian_blur(img,k_size):
    img = cv2.GaussianBlur(img,(k_size,k_size),cv2.BORDER_DEFAULT)
    return img

# Plot a barchart of the results
def f1bar(result1, result2, whichimgs):
    image_labels = []
    for i in whichimgs:
        each_image = 'dart'+str(i)+'jpg.'
        image_labels.append(each_image)

    indices = np.arange(len(image_labels))
    width = 0.35

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bar1 = ax.bar(indices, result1, width, color = 'royalblue', label = 'VJ F1-Score')
    bar2 = ax.bar(indices+width, result2, width, color = 'seagreen', label = 'VJHT F1-Score')
    plt.xticks(indices+width/2, image_labels, rotation = 'vertical')
    plt.legend(loc = 'lower left', bbox_to_anchor=(0,1.02,1,0.2), mode = 'expand', ncol = 2)
    plt.show()



def sobel_edge_filter(img):
    sobelk_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobelk_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    deriv_x = convolution2D(img,sobelk_x)
    deriv_y = convolution2D(img,sobelk_y)
    mag_G = np.hypot(deriv_x,deriv_y)
    # normalise magnitude
    mag_G = mag_G / mag_G.max() * 255
    theta = np.arctan2(deriv_y, deriv_x)
    return (mag_G, theta)


# improving the sobel detector to a canny-edge detector 
def canny_edge_filter(sobel_output,theta,min_ratio,max_ratio,weak_pixel_value,strong_pixel_value):
   output_suppresion = non_max_suppresion(sobel_output,theta)
   output_threshold = threshold(output_suppresion,min_ratio,max_ratio,weak_pixel_value)
   final_output =  hysteris_thresholding(output_threshold,strong_pixel_value,weak_pixel_value)
   return final_output

#%%
#  hough transform for detecting circles
def hough_circle_transform(gradient_magnitude,gradient_direction):
    width = gradient_magnitude.shape[0]
    height = gradient_magnitude.shape[1]
    R_max = 80
    R_min = 60
    # 3 dimensional hough_space
    hough_space = np.zeros((width,height,R_max+1))
    radius_theta = []
    steps = 360
    theta = np.deg2rad(np.arange(360))
    for r in range(R_min,R_max+1):
        for t in range(steps):
            radius_theta.append( (r, int(r*np.cos(2*np.pi*t/steps)), int(r*np.sin(2*np.pi*t/steps)) ) ) 
    for i in range(0,width):
        print(i)       
        for j in range(0,height):
            if gradient_magnitude[i,j] == 255:
                 for r, r_cost, r_sint in radius_theta:
                    a = int(i-r_cost)
                    b = int(j-r_sint)
            # accumulator so then centre r with max value determines that there is a circle
                    if a >= 0 and a < width and b >= 0 and b < height:
                        hough_space[a,b,r]+= 1
    #find circle with max count in hough space
    return hough_space

#%%
# -------------------------------- copied function from kheeran
# Creating 2D Hough Space to display
def houghspace_2D (hough_space):
   return np.sum(hough_space,axis=2)
# -----------------------------------------------------

#  we threshold the image to make the edges we want more clearer and discard the edges 
#  we have no use for 
def threshold(sobel_output,min_ratio,max_ratio,weak_pixel_value):
    highthreshold = sobel_output.max() * max_ratio
    lowthreshold =  sobel_output.max() * min_ratio
    result_matrix = np.zeros(sobel_output.shape)
    # Set high thresholded pixels that we particular want with a high value pixel
    strong_pixel_value = 255 
    # set in between thresholded pixels  we would  want to keep to a fair pixel value
    s_i, s_j = np.where(sobel_output>=highthreshold)
    w_i, w_j  = np.where(np.logical_and(sobel_output<highthreshold,sobel_output>=lowthreshold))
    result_matrix[s_i,s_j] = strong_pixel_value
    result_matrix[w_i,w_j] = weak_pixel_value
    return result_matrix


#   check  neighbouring pixels to pixels with between threshold and highthreshold values
#   to decide if they are worth considering as edge pixels (highthresholded) and discard the 
#  ones which are not
def hysteris_thresholding(image,strong_pixel_value,weak_pixel_value):
    width = image.shape[0]
    height = image.shape[1]
    top = image.copy()
    bottom = image.copy()
    right = image.copy()
    left = image.copy()

    for i in range(1,width-1):
       for j in range(1,height-1):
           apply_hysteris(top,strong_pixel_value,weak_pixel_value,i,j)

    for i in range(width-1,0,-1):
        for j in range(height-1,0,-1):
             apply_hysteris(bottom,strong_pixel_value,weak_pixel_value,i,j)   

    for i in range(1,width-1):
        for j in range(height-1,0,-1):
              apply_hysteris(right,strong_pixel_value,weak_pixel_value,i,j)             
    
    for i in range(width-1,0,-1):
        for j in range(1,height-1):
               apply_hysteris(left,strong_pixel_value,weak_pixel_value,i,j)
    
    output = top + bottom + right + left 
    output[output>strong_pixel_value] = strong_pixel_value
    return output       


def apply_hysteris(image,strong_pixel_value,weak_pixel_value,row,col):
      if image[row,col] == weak_pixel_value:
                if(check_pixel_neighbours(image,row,col,strong_pixel_value)):
                    image[row,col] = strong_pixel_value
                else:
                    image[row,col] = 0   
 
def check_pixel_neighbours(image,i,j,strong):
    return image[i-1,j] == strong or image[i+1,j] == strong or image[i-1,j-1] == strong or \
           image[i+1,j+1] == strong or image[i,j-1] == strong or image[i,j+1] == strong or \
           image[i+1,j-1] == strong or image[i-1,j+1] == strong    



def get_normalised_edge_direction(angle):
    edge = angle - (np.pi/2)
    for i in range(0,edge.shape[0]):
        for j in range(0,edge.shape[1]):
            edge[i,j] = math.degrees(edge[i,j])
    #  normalise edge-angle to be between 0 and 360
    edge = edge / edge.max() * 360
    return edge



# we want to only keep the pixel values along the edges that are most intense
# hence we look along each edge direction and check neighbouring pixel values in that direction
# setting to zero lower-intensity pixels and keeping value of the current max intense pixel
def non_max_suppresion(sobel_output,gradient_direction):
    result_matrix = np.zeros(sobel_output.shape)
    width = sobel_output.shape[0]
    height = sobel_output.shape[1]
    edge_direction = get_normalised_edge_direction(gradient_direction)
    # pixels have 8 neighbours so we check  for each  360/8 edge-angle starting at 0
    for i in range(1,width-1):
       for j in range(1,height-1): 
             #  angle 0 and 360 ( horizontal edge)
            if (0 <= edge_direction[i,j] < 45) or  (180 <= edge_direction[i,j] < 225) or (edge_direction[i,j] == 360):
                    Pneighbour_pixel = sobel_output[i-1,j] 
                    Nneighbour_pixel = sobel_output[i+1,j]

              # angle 45 and  225  ( right diagonal edge)
            elif (45 <= edge_direction[i,j] < 90 or  225 <= edge_direction[i,j]< 270):
                    Pneighbour_pixel = sobel_output[i-1,j-1] 
                    Nneighbour_pixel = sobel_output[i+1,j+1]
              #  angle 90 and 270 ( vertical edge)
            elif (90 <= edge_direction[i,j] < 135 or 270 <= edge_direction[i,j] < 315): 
                    Pneighbour_pixel = sobel_output[i,j-1] 
                    Nneighbour_pixel = sobel_output[i,j+1]
             #  angle 135 and 315 ( left diagonal edge)
              # ( 135 <= edge_direction[i,j] < 180 or 315 <= edge_direction[i,j] < 360 )
            else:   
                    Pneighbour_pixel = sobel_output[i-1,j+1] 
                    Nneighbour_pixel = sobel_output[i+1,j-1]
        # compare pixel and set to max or zero
            if (sobel_output[i,j] >= Pneighbour_pixel) and (sobel_output[i,j] >= Nneighbour_pixel):
                result_matrix[i,j] = sobel_output[i,j]
            else:
                 result_matrix[i,j] = 0    
    return result_matrix             


def convolution2D(orig_matrix,kernel):
    output = np.zeros(orig_matrix.shape)
    kernel = np.flipud(np.fliplr(kernel))
    kernel_size = kernel.shape
    width  = orig_matrix.shape[0]
    height = orig_matrix.shape[1]
    if kernel_size[0] == kernel_size[1]:
        if kernel_size[0] > 2:
             orig_padded = np.pad(orig_matrix, kernel_size[0]-2, mode='constant')

    # Loop over every pixel of the image
    for x in range(width):
        for y in range(height):
            # element-wise multiplication of the flipped kernel and the image
            output[x, y] = (kernel * orig_padded[ x: x + kernel_size[0],y: y + kernel_size[1]]).sum()            
    return output


#%%
def ht_to_rectangle(frame,circles,min_dis = 50):
    color = (0,255,0)
    boxes = []
    # nc =  maxrad-minrad #num circles
    # r = (rindex+minrad+1)*maxrad/nc
    # circle0 = (x0-int(r),y0-int(r),int(2*r),int(2*r)
    for circ in circles:
        a,b,r = circ
        # calculating rectangle corners
        ulx, uly = (a-r),(b+r)
        urx, ury = (a+r),(b+r)
        lrx, lry = (a+r),(b-r)
        llx, lly = (a-r), (b-r)
        x1=min(ulx,urx,lrx,llx)
        x2=max(ulx,urx,lrx,llx)
        y1=min(uly,ury,lry,lly)
        y2=max(uly,ury,lry,lly)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
        boxes.append([x1,y1,x2,y2])
        # cv2.circle(frame,(a,b),r,color=color,thickness=4)
        # cv2.circle(frame, (a,b),0, color=(0, 0, 255), thickness=4) 
    
    cv2.imwrite('circle_dart1.jpg',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return boxes

    # Viola Jones dartboard detector
def VJ_detector(image, obj_classifier):
    boxes = []
    #run classifier
    obj = obj_classifier.detectMultiScale(image, 1.1, 1, 0|cv2.CASCADE_SCALE_IMAGE, (50,50), (500,500))
    for (x,y,w,h) in obj:
        boxes.append([x,y,x+w,y+h])
    return boxes



def get_best_circle(hs_2d,hs_3d):
    circles = []
    count  = 3
    hs_2d_copy = np.copy(hs_2d) 
    while(count>0):
        # get best center
        a, b = np.unravel_index(hs_2d_copy.argmax(), hs_2d_copy.shape)
        #   get best radius
        r = np.unravel_index(hs_3d[a,b,:].argmax(),hs_3d[a,b,:].shape)
        r = r[0]
        if all( np.sqrt( (a-x)**2 + (b-y) **2 ) > r + radius + 50  for  x,   y, radius in circles):
            circles.append((a,b,r))
        #set best center  accumulator value to zero because we found it  
            hs_2d_copy[a,b] = 0
            count = count - 1
        else: 
            #  best center collides with other cicrles set to zero and look for others
            hs_2d_copy[a,b] = 0
    return circles

#%%
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

def combineBoundingBox(box1, box2):
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = box2[0] + box2[2] - box1[0]
    h = max(box1[1] + box1[3], box2[1] + box2[3]) - y
    return (x, y, w, h)

#  if false they overlap 
def check_if_intersect(box1,box2):
    return not(box2[0] > box1[2] \
        or box2[2] < box1[0] \
        or box2[1] > box1[3] \
        or box2[3] < box1[1])
 
# def main():
#    frame = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
#    frame_grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    smooth_img = gaussian_blur(frame_grey,7)
#    grad_mag, grad_direc = sobel_edge_filter(smooth_img)
#    canny_output = canny_edge_filter(grad_mag,grad_direc,0.1,0.4,25,255)
#    cv2.imwrite('canny_output.jpg',canny_output)
#    t0 = time.time() 
#    hs = hough_circle_transform(canny_output,grad_direc)
#    t1 = time.time()
#    print(t1-t0,'time taken in seconds')
#    hs_2d = houghspace_2D(hs)
#    cv2.imwrite('hough_circle_transform.jpg',hs_2d)   
#    circles = get_best_circle(hs_2d,hs)
#    display_ht_to_rectangle(frame,circles)

# if __name__ == "__main__": 
#     main()

#%%
frame = cv2.imread('dart0.jpg',cv2.IMREAD_COLOR)
frame_grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
#%%
smooth_img = gaussian_blur(frame_grey,3)
grad_mag, grad_direc = sobel_edge_filter(smooth_img)
canny_output = canny_edge_filter(grad_mag,grad_direc,0.1,0.4,25,255)
cv2.imwrite('canny_output.jpg',canny_output)

#%%
t0 = time.time() 
hs = hough_circle_transform(canny_output,grad_direc)
t1 = time.time()
print(t1-t0,'time taken in seconds')
hs_2d = houghspace_2D(hs)
cv2.imwrite('hough_circle_transform.jpg',hs_2d)   
#%%
circles = get_best_circle(hs_2d,hs)
ht_darts = ht_to_rectangle(frame,circles)
viola_darts = VJ_detector(frame,cascade_name)   
# %%
#  combining viola jones detector and circle hough transform 
def final_box(ht_darts,viola_darts,threshold=0.3):
    combined_boxes = []
    for  i  in range(len(ht_darts)):
        for j in range(len(viola_darts)):
            if(check_if_intersect(ht_darts[i],viola_darts[j])):
                iou = rectangle_intersection(ht_darts[i],viola_darts[j])
                print(iou)
                if(iou>=threshold):
                    x,y,x1,y1 = combineBoundingBox(ht_darts[i],viola_darts[j])
                    combined_boxes.append([x,y,x1,y1])
    return combined_boxes
            # viola and hough_transform combined , break from loop

best_box = final_box(ht_darts,viola_darts)

#%%

def final_display(box,v_box):
    # best box list is empty
    if not box:
        # display viola-detector box
        for i in range(len(v_box)):
            b = v_box[i]
            cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),(0,255,0),3)
    # display best boxes
    else:
        for i in range(len(box)):
            b = box[i]
            cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),(255,0,0),3)        
    cv2.imwrite('combined-detector.jpg',frame)

final_display(best_box,viola_darts)

# %%
def threshold_hough_space(hs_3d,R_min=60,R_max=80,threshold= 150):
    a,b,r = hs_3d.shape
    hs_copy = np.copy(hs_3d)
    for x in range(0,a):
        for y  in range(0,b):
            for z in range(R_min,R_max):
                if hs_copy[x,y,z] <=threshold:
                   hs_copy[x,y,z] = 0 
    return hs_copy
trs_hs = threshold_hough_space(hs)
trs_hs_2d = houghspace_2D(trs_hs)
cv2.imwrite('2d-thresholded.jpg',trs_hs_2d)

# %%
