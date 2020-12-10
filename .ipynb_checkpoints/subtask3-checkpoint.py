import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
from collections import defaultdict


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1  step is noise reduction we can use the cv2 function gaussian blur kernel to do so 
def gaussian_blur(img,k_size):
    img = cv2.GaussianBlur(img,(k_size,k_size),cv2.BORDER_DEFAULT)
    return img




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



#  hough transform for detecting circles
def hough_circle_transform(gradient_magnitude):
    width = gradient_magnitude.shape[0]
    height = gradient_magnitude.shape[1]
    #safe limit as the highest radius of a circle in the image
    R_max = 55
    R_min = 50
    # 3 dimensional hough_space
    hough_space = np.zeros((width,height,R_max+1))
    # accumulator
    acc = defaultdict(int)
    radius_theta = []
    theta = np.deg2rad(np.arange(360))
    for r in range(R_min,R_max+1):
        for t in theta:
            radius_theta.append( (r, r*np.cos(t), r*np.sin(t) ) )

  
   
    for i in range(0,width):
        print(i)     
        for j in range(0,height):
            if gradient_magnitude[i,j] == 255:
                for r, rcos_t, rsin_t in radius_theta:
                    a = int( i- rcos_t)
                    b = int( j- rsin_t)
                # accumulator so then centre r with max value determines that there is a circle
                    if a >= 0 and a < width and b >= 0 and b < height:
                        hough_space[a,b,r] +=1
                        acc[(a, b, r)] += 1
                

    #find circle with max count in hough space
    return hough_space,acc

# -------------------------------- copied function from kheeran
#Creating 2D Hough Space to display
def houghspace_2D (hough_space):
    a, b, r = hough_space.shape #to calculate the no of elements
    hspace = np.zeros((a,b))
    for i in range (0,a):
        for j in range (0,b):
            sum = 0
            for k in range (0,r):
                if hough_space[i,j,k] !=0:
                   sum += hough_space[i,j,k]
            hspace[i,j] = sum
                
    norm = np.amax(hspace)
    hspace = hspace*255/norm
    return hspace
# -----------------------------------------------------




def display_circle(frame,hough_space,acc,circle_threshold=0.45,steps=200):
    # x, y  = np.unravel_index(hough_space.argmax(), hough_space.shape)

    # r = []    
    # print('max threshold', hough_space[x,y,r])
   
    # circles = []
    # for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    #     a, b, r = k
    #     if hough_space[a,b,r] / steps >= circle_threshold and all( (a - x) ** 2 + (b - y) ** 2 > r_hat ** 2 for x,   y, r_hat in circles):
    #             circles.append((a,b,r))    
    # fig, ax = plt.subplots()
    # plt.title('Circle')
    # plt.imshow(frame)
    # for circ in circles:
    #    a,b,r= circ
    #    circle = plt.Circle((a,b),r,color='blue',fill=False)
    #    ax.set_aspect(1)
    color = ['blue', 'red', 'green', 'yellow']

    fig, ax  = plt.subplots()
    plt.imshow(frame)
    for i in range(0,hough_space.shape[0]):
        for j in range(0,hough_space.shape[1]):
           sum = hough_space[i][j]
           if sum < 70: 
               pass
           elif sum >= 70 and sum < 80:
               print('70')
               plt.scatter(i,j, color=color[0])
           elif sum >= 80 and sum< 90:
               print('80')
               plt.scatter(i,j,color=color[1])
           elif sum >= 90 and sum < 100:
               print('90')
               plt.scatter(i,j,color=color[2])
           else:
               print('101')
               plt.scatter(i,j,color=color[3])

    plt.title('Circle')
    plt.show()

    
 
def hough_line_transform():
    pass



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


# def hysteris_thresholding(image,strong_pixel_value,weak_pixel_value):
#     top_corner = np.copy(image)
#     for i in range(1,image.shape[0]-1):
#        for j in range(1,image.shape[1]-1):
#             if image[i,j] == weak_pixel_value:
#                 if(check_pixel_neighbours(image,i,j,strong_pixel_value)):
#                     image[i,j] = strong_pixel_value
#                 else:
#                     image[i,j] = 0   
#     return image       




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


# canny edge detector  get rid of the edges we are not really interested in and keep the only good parts 
# making it better than the sobel edge detector






def main():
    #load image
    frame = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
       #swap to RGB
    frame = convertToRGB(frame)
     #convert to gray scale
    frame_grey = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
  
    # perform gaussian blur 
    smooth_img = gaussian_blur(frame_grey,5)
    grad_mag, grad_direc = sobel_edge_filter(smooth_img)
    canny_output = canny_edge_filter(grad_mag,grad_direc,0.1,0.35,25,255)
    hs,acc = hough_circle_transform(canny_output)
    hs_2d = houghspace_2D(hs)
    # cv2.imwrite('houghs_space_dart1.jpg',hs_2d)
    # cv2.imshow('cany-output',canny_output)
    display_circle(frame,hs_2d,acc)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
    



    


if __name__ == "__main__":
    main()






