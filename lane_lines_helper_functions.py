import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def get_line_params(lines):
    m_list = []
    b_list = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            m = (y2-y1)/(x2-x1)
            b = y1 - m*x1

            m_list.append(m)
            b_list.append(b)

    m_ave = 0
    b_ave = 0

    if len(m_list)>0:
        m_ave = sum(m_list)/len(m_list)
        b_ave = sum(b_list)/len(b_list)

    return [(m_ave,b_ave)]

def points_from_params(start, stop, m, b, y_min, y_max):
    x_vals = []
    y_vals = []

    for x in range(start, stop):
        y = int(m*x + b)
        if (y >= y_min) and (y < y_max):
            x_vals.append(x)
            y_vals.append(y)

    if len(x_vals)>0:
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
    else:
        x_vals = np.array([0])
        y_vals = np.array([0])

    return x_vals, y_vals

def extrapolate_line(img_shape, line_params):

    lines_new = []

    for m_ave, b_ave in line_params:
        start = 0
        stop = int(img_shape[1])
        y_min=(img_shape[0]/2)+50
        y_max=img_shape[0]
        x_vals, y_vals = points_from_params(start, stop, m_ave, b_ave, y_min, y_max)

        i1 = np.argmin(y_vals)
        i2 = np.argmax(y_vals)
        x1 = x_vals[i1]
        y1 = y_vals[i1]
        x2 = x_vals[i2]
        y2 = y_vals[i2]

        lines_new.append(np.array([[x1,y1,x2,y2]]))

    lines_new = np.array(lines_new)

    return lines_new

def lines_ave(img_shape, lines):

    line_params = get_line_params(lines)

    lines_new = extrapolate_line(img_shape, line_params)

    return lines_new

def draw_lines_new(img, lines, color=[255, 0, 0], thickness=2):
    lines_new = lines_ave(img.shape, lines)
    for line in lines_new:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines_new(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines_new(line_img, lines)
    return line_img

def get_line_params_advanced(img, rho, theta, threshold, min_line_len, max_line_gap, trans_im_h=(88/128), margin=200):

    gray = np.zeros_like(img[:,:,0])
    gray = grayscale(img)

    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    imshape = img.shape

    bottom_left_trap = (0, imshape[0]-60)
    top_left_trap = ((imshape[1]/2)-10, (imshape[0]/2)+70)
    top_right_trap = ((imshape[1]/2)+10, (imshape[0]/2)+70)
    bottom_right_trap = (imshape[1], imshape[0]-60)

    vertices = np.array([[bottom_left_trap,top_left_trap, top_right_trap, bottom_right_trap]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    masked_edges_left = masked_edges[:,:int(masked_edges.shape[1]/2)]

    masked_edges_right = masked_edges[:,int(masked_edges.shape[1]/2):]

    lines_left = cv2.HoughLinesP(masked_edges_left, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                 maxLineGap=max_line_gap)

    line_params_left = get_line_params(lines_left)
    m_left = line_params_left[0][0]
    b_left = line_params_left[0][1]

    lines_right = cv2.HoughLinesP(masked_edges_right, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    line_params_right = get_line_params(lines_right)
    m_right = line_params_right[0][0]
    b_right = line_params_right[0][1]

    y_top_draw = int(trans_im_h*img.shape[0])
    y_bottom_draw = img.shape[0]

    x_top_left_draw = int((y_top_draw - b_left)/m_left)
    x_bottom_left_draw = int((y_bottom_draw - b_left)/m_left)
    x_top_right_draw = int((y_top_draw - b_right)/m_right + (img.shape[1]/2))
    x_bottom_right_draw = int((y_bottom_draw - b_right)/m_right + (img.shape[1]/2))

    src = np.float32(
        [[x_bottom_left_draw, y_bottom_draw],
         [x_top_left_draw, y_top_draw],
         [x_top_right_draw, y_top_draw],
         [x_bottom_right_draw, y_bottom_draw]])

    dst = np.float32(
        [[margin,edges.shape[0]],
         [margin,0],
         [edges.shape[1]-margin,0],
         [edges.shape[1]-margin,edges.shape[0]]])

    return src, dst, edges
