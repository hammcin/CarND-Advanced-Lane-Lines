import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def find_cal_pts(images, nx=9, ny=6):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in images:
        # Read in each image
        img = mpimg.imread(fname)

        # Convert image to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    return imgpoints, objpoints

# Function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

# Function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    mag = np.sqrt(sobelx**2 + sobely**2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_mag = np.uint8(255*mag/np.max(mag))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_mag)
    binary_output[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

# Function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_orient = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_orient)
    binary_output[(grad_orient >= thresh[0]) & (grad_orient <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

# Function that thresholds the S-channel of HLS
# Uses exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):

    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]

    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    return binary_output

# Pipeline
def pipeline(img, s_thresh = (215, 240), sobelx_thresh=(20, 100), r_thresh = (225, 255), ksize=3):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    r_channel = img[:,:,0]

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=sobelx_thresh)

    combined = np.zeros_like(gradx)
    combined[(gradx == 1)] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

    color_combined = np.zeros_like(s_binary)
    color_combined[(s_binary == 1) | (r_binary == 1)] = 1

    # Stack each channel
    # color_binary = np.dstack(( np.zeros_like(combined), combined, s_binary)) * 255

    color_binary = np.zeros_like(combined)
    color_binary[(combined == 1) | (color_combined == 1)] = 1

    return color_binary

def hist(img, h_frac=(1/2)):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[int(h_frac*img.shape[0]):,:]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)

    return histogram

def find_lane_pixels(binary_warped, nwindows = 9, margin = 100, minpix = 50, h_frac=(1/2)):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(h_frac*binary_warped.shape[0]):,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = np.maximum(0, leftx_current - margin)
        win_xleft_high = np.minimum(leftx_current + margin, (binary_warped.shape[1]-1))
        win_xright_low = np.maximum(rightx_current - margin, 0)
        win_xright_high = np.minimum(rightx_current + margin, (binary_warped.shape[1]-1))

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                      (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
                      (win_xright_high,win_y_high),(0,255,0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img

def find_lane_pixels_conv(binary_warped, nwindows = 9, margin = 100, minpix = 50,
                          window_width = 50, h_frac=(1/2)):
    # Take a histogram of the bottom half of the image
    # histogram = np.sum(binary_warped[int(h_frac*binary_warped.shape[0]):,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # midpoint = np.int(histogram.shape[0]//2)
    # leftx_base = np.argmax(histogram[:midpoint])
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(binary_warped[int(h_frac*binary_warped.shape[0]):,:int((1/2)*binary_warped.shape[1])], axis=0)
    leftx_base = np.argmax(np.convolve(window,l_sum))-int(window_width/2)
    r_sum = np.sum(binary_warped[int(h_frac*binary_warped.shape[0]):,int((1/2)*binary_warped.shape[1]):], axis=0)
    rightx_base = np.argmax(np.convolve(window,r_sum))-int(window_width/2)+int(binary_warped.shape[1]/2)

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = np.maximum(0, leftx_current - margin)
        win_xleft_high = np.minimum(leftx_current + margin, (binary_warped.shape[1]-1))
        win_xright_low = np.maximum(rightx_current - margin, 0)
        win_xright_high = np.minimum(rightx_current + margin, (binary_warped.shape[1]-1))

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                      (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
                      (win_xright_high,win_y_high),(0,255,0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        # if len(good_left_inds) > minpix:
        #     leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        # if len(good_right_inds) > minpix:
        #     rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # convolve the window into the vertical slice of the image
        image_layer = np.sum(binary_warped[win_y_low:win_y_high,:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = int(window_width/2)

        if len(good_left_inds) > minpix:
            l_min_index = int(max(leftx_current+offset-margin,0))
            l_max_index = int(min(leftx_current+offset+margin,binary_warped.shape[1]))
            leftx_current = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

        # Find the best right centroid by using past right center as a reference
        if len(good_right_inds) > minpix:
            r_min_index = int(max(rightx_current+offset-margin,0))
            r_max_index = int(min(rightx_current+offset+margin,binary_warped.shape[1]))
            rightx_current = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped, left_fit, right_fit):
    # Find our lane pixels first
    # leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, nwindows, margin, minpix, h_frac)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    left_xbounds = (left_fitx >= 0) & (left_fitx < binary_warped.shape[1])
    right_xbounds = (right_fitx >= 0) & (right_fitx < binary_warped.shape[1])

    ## Visualization ##
    # Colors in the left and right lane regions
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return (left_fitx[left_xbounds], ploty[left_xbounds]), (right_fitx[right_xbounds], ploty[right_xbounds])

def fit_polynomial_conv(binary_warped, left_fit, right_fit):
    # Find our lane pixels first
    # leftx, lefty, rightx, righty, out_img = find_lane_pixels_conv(binary_warped, nwindows, margin, minpix, window_width, h_frac)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    left_xbounds = (left_fitx >= 0) & (left_fitx < binary_warped.shape[1])
    right_xbounds = (right_fitx >= 0) & (right_fitx < binary_warped.shape[1])

    ## Visualization ##
    # Colors in the left and right lane regions
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return (left_fitx[left_xbounds], ploty[left_xbounds]), (right_fitx[right_xbounds], ploty[right_xbounds])

def fit_poly(img_shape, left_fit, right_fit):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit, margin=100):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    # margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = []
    right_lane_inds = []
    for i in range(len(nonzerox)):
        predict_x_left = (left_fit[0]*(nonzeroy[i]**2) + left_fit[1]*nonzeroy[i]
            + left_fit[2])
        predict_x_right = (right_fit[0]*(nonzeroy[i]**2) + right_fit[1]*nonzeroy[i]
            + right_fit[2])
        if ((nonzerox[i] >= (np.maximum(predict_x_left - margin, 0)))
            & (nonzerox[i] < (np.minimum(predict_x_left + margin, (binary_warped.shape[1]-1))))):
            left_lane_inds.append(i)
        if ((nonzerox[i] >= (np.maximum(predict_x_right - margin, 0)))
            & (nonzerox[i] < (np.minimum(predict_x_right + margin, (binary_warped.shape[1]-1))))):
            right_lane_inds.append(i)
    left_lane_inds = np.array(left_lane_inds)
    right_lane_inds = np.array(right_lane_inds)

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img

def visual_around_poly(binary_warped, left_fit, right_fit, out_img, margin=100):

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, left_fit, right_fit)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()

    left_lmargin = np.maximum(left_fitx-margin, 0)
    left_rmargin = np.minimum(left_fitx+margin, (binary_warped.shape[1]-1))
    lmargin_in_img = left_lmargin != left_rmargin
    left_fitx = left_fitx[lmargin_in_img]
    ploty_left = ploty[lmargin_in_img]

    right_lmargin = np.maximum(right_fitx-margin, 0)
    right_rmargin = np.minimum(right_fitx+margin, (binary_warped.shape[1]-1))
    rmargin_in_img = right_lmargin != right_rmargin
    right_fitx = right_fitx[rmargin_in_img]
    ploty_right = ploty[rmargin_in_img]

    left_line_window1 = np.array([np.transpose(np.vstack([np.maximum(left_fitx-margin, 0), ploty_left]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([np.minimum(left_fitx+margin, (binary_warped.shape[1]-1)),
                              ploty_left])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([np.maximum(right_fitx-margin, 0), ploty_right]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([np.minimum(right_fitx+margin, (binary_warped.shape[1]-1)),
                              ploty_right])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    left_xbounds = (left_fitx >= 0) & (left_fitx < binary_warped.shape[1])
    right_xbounds = (right_fitx >= 0) & (right_fitx < binary_warped.shape[1])

    # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    left_xy = (left_fitx[left_xbounds], ploty_left[left_xbounds])
    right_xy = (right_fitx[right_xbounds], ploty_right[right_xbounds])

    return result, left_xy, right_xy

def measure_curvature_pixels(ploty, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    # (1 + (2*A*y + B)^2)^(3/2)/abs(2*A)
    A_left = left_fit[0]
    B_left = left_fit[1]
    left_curverad = (1 + (2*A_left*y_eval + B_left)**2)**(3.0/2)/np.abs(2*A_left)
    A_right = right_fit[0]
    B_right = right_fit[1]
    right_curverad = (1 + (2*A_right*y_eval + B_right)**2)**(3.0/2)/np.abs(2*A_right)

    return left_curverad, right_curverad

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr, ym_per_pix=(30/720), xm_per_pix=(3.7/700)):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    # (1 + (2*A*y + B)^2)^(3/2)/abs(2*A)
    y_m = y_eval*ym_per_pix
    A_left = left_fit_cr[0]
    B_left = left_fit_cr[1]
    left_curverad = (1 + (2*A_left*y_m + B_left)**2)**(3.0/2)/np.abs(2*A_left)
    A_right = right_fit_cr[0]
    B_right = right_fit_cr[1]
    right_curverad = (1 + (2*A_right*y_m + B_right)**2)**(3.0/2)/np.abs(2*A_right)

    return left_curverad, right_curverad
