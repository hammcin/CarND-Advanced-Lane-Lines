## Advanced Lane Finding Project

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_find_corners.jpg "Corners"
[image2]: ./output_images/calibration_distort_correct.jpg "Calibration Distortion"
[image3]: ./output_images/test5_distort_correct.jpg "Test Distortion"
[image4]: ./output_images/test5_binary.jpg "Binary"
[image5]: ./output_images/test5_warped.jpg "Warped Binary"
[image6]: ./output_images/test5_detect_lane_slide.jpg "Sliding Window"
[image7]: ./output_images/test5_detect_lane_prev.jpg "Search Around Polynomial"
[image8]: ./output_images/test5_curve_pos.jpg "Result"
[video1]: ./output_videos/project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

For this step, I used the function `find_cal_pts()` in the file `advanced_lane_line_helper_functions.py` (lines # 6-31) to find the image points and object points.  I then used the image and object points in the OpenCV function `cv2.calibrateCamera()` to find the camera matrix and distortion coefficients.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the calibration image using the `cv2.undistort()` function and obtained this result:

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

After calculating the camera calibration and distortion coefficients using the calibration images, I applied this distortion correction to one of the test images using the `cv2.undistort()` function.  Here is an example of a distortion corrected test image:

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

For this step, I used the function `pipeline()` in the file `advanced_lane_line_helper_functions.py` (lines # 126-157).  First, I took the gradient in the x direction using a Sobel filter.  The Sobel filter had a kernel size of 3.  I then, applied a threshold to the gradient image that only retained pixels with a magnitude between 20 and 100.  This step was implemented in the function `abs_sobel_thresh()` in the file `advanced_lane_line_helper_functions.py` (lines # 33-58).

Then, I converted the RGB image to HLS color space and isolated the S channel.  I applied a threshold to this color channel to retain pixels whose value was in the range 215-240.  I, then, isolated the R channel of the RGB image and applied a threshold so that pixels whose value was in the range 225-255 were retained.  Finally, I combined these three images (x gradient threshold, S channel threshold, R channel threshold) so that if a pixel was actviated in any of these three images, it was also activated in the output image.

Here's an example of my output for this step:

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_line_params_advanced()`, which appears in lines # 181-241 in the file `lane_lines_helper_functions.py`.  This function takes as input an undistorted image of straight lane lines.  This function then uses the approach from the last project to detect the lane lines using the Hough Transform.  This function also accepts a number of parameters related to detecting lane lines.  This function uses the parameters of the detected lane lines to find the vertices of the trapezoid formed by the lane lines.  These vertices are used as the source (`src`) points for the perspective transform.  The input parameter trans_im_h (0.625) of this function controls the height of the trapezoid.  Finally, the input parameter margin (300 pixels), as well as the height and width of the input image, is used to determine the destination (`dst`) points for the perspective transform.

The function `get_line_params_advanced()` in the file `lane_lines_helper_functions.py` returned the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 205, 720      | 300, 720      |
| 598, 450      | 300, 0        |
| 681, 450      | 980, 0        |
| 1116, 720     | 980, 720      |

With these source and destination points, the OpenCV function `cv2.getPerspectiveTransform()` could be used to get the transformation matrix as well as its inverse.  The function `cv2.warpPerspective()` could be used with the transformation matrix to perform the perspective transform.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for identifying lane-line-pixels is in the function `find_lane_pixels()` in the file `advanced_lane_line_helper_functions.py` (lines # 170-248).  `find_lane_pixels()` uses a sliding window approach to identify lane-line pixels.  First I calculated a histogram of the lower portion of the image.  The parameter h_frac (0.75) determined how much of the image was used to calculate the histogram.  I then split the histogram in half.  I found the position of the maximum value in the left and right halves of the histogram.  I used these as the starting position of the left and right lane lines.

I then, successively, calculated the boundaries of my windows for the left and right lane lines, found all the activated pixels inside the bounds of my windows, re-centered the windows and slid them to the next position in the image.  This function returned the positions of the pixels that belonged to the left and right lane lines.

I then used the Numpy function `np.polyfit()` to fit a second-order polynomial to the left and right lane line pixels.

The function `fit_polynomial()` in the file `advanced_lane_line_helper_functions.py` (lines # 357-388) used the polynomial fit from `np.polyfit()` to plot the detected polynomials throughout the image.

Below, find example output for this step using the sliding window approach:

![alt text][image6]

Once I had a polynomial fit to the lane lines in a previous image, I used the function `search_around_poly()` in the file `advanced_lane_line_helper_functions.py` (lines # 435-478) to detect lane line pixels around the previously fit polynomial in a successive image.  For each activated pixel in the input image, this function uses the y-value of the activated pixel to calculate the predicted x-value for the left and right fit polynomials.  The function, then, checks whether the x-value of the activated pixel lies within a margin around the predicted x-value of the left and right fit polynomials.  If this is the case, the position of the activated pixel is returned as a detected pixel of either the left or right lane lines.

As before, I used the Numpy function `np.polyfit()` to fit a second-order polynomial to the left and right lane line pixels.

The function `fit_poly()` in the file `advanced_lane_line_helper_functions.py` (lines # 423-433), which is called by the function `visual_around_poly()` in the file `advanced_lane_line_helper_functions.py` (lines # 480-533), used the polynomial fit from `np.polyfit()` to plot the detected polynomials throughout the image.

Below, find example output for this step using the approach of searching around a previously detected polynomial:

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code to calculate the radius of curvature is contained in the 50th code cell and the code to calculate the position of the vehicle is in the 53rd code cell of the IPython notebook located in "./Advanced Lane Finding.ipynb".

To find the radius of curvature in meters, I assumed that the number of meters per pixel in the y-direction was equal to ym_per_pix=(30/720) and the number of meters per pixel in the x-direction was equal to xm_per_pix=(3.7/700).  After I detected the pixels of the left and right lane lines using the function `find_lane_pixels()` in the file `advanced_lane_line_helper_functions.py` (lines # 170-248), I converted the units of the locations of the detected pixels to meters.  I then fit a second-order polynomial using the Numpy function `np.polyfit()` to the detected pixel locations in units of meters.  I then used the function `measure_curvature_real()` in the file `advanced_lane_line_helper_functions.py` (lines # 555-574) to calculate the radius of curvature in meters.  The function `measure_curvature_real()` uses the following formula to calculate the radius of curvature: (1 + (2*A*y + B)^2)^(3/2)/abs(2*A).

To calculate the position of the vehicle with respect to center, I calculated the difference between the x-position of the midway point between the detected lane lines at the bottom of the image and the center point of the image (half the width of the image).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is contained in the 59th code cell of the IPython notebook located in "./Advanced Lane Finding.ipynb".  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video_out.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One major problem I had with this project was performing the perspective transform of the binary image.  The major problem I ran into while implementing this step was that I had to be careful about the dimensions of the trapezoid defined by the source points.  If I chose inappropriate dimensions for the trapezoid, then the `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` functions did not behave as expected.  Additionally, the dimensions of the trapezoid defined by the source points had a major influence on the polynomial fits and the radius of curvature I detected.  In the end, I found that it was better for me to choose my source points so that warping a lane lines image resulted in the lane lines not being exactly parallel in the warped image because it improved my polynomial fit.

My pipeline is most likely to fail on roads where the pavement changes color or where objects such as trees cast a shadow on the road.

To make my pipeline more robust, I could use convolutions to detect the location of the lane lines in the image.  Also, I could find better criteria for detecting outlier lane line detections.  Currently, my pipeline uses a check on the width of the detected lane lines at the bottom and the top of the image and the ratio of the curvature of the left and right lane lines to detect outlier detections, in which case these outlier detections are ignored.  Also, I may need to average together lane line detections over a larger number of frames (currently, use 3) or use a weighted average instead.
