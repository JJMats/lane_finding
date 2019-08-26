# Helper Functions
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Draw ROI on image to determine points to choose, then warp utilizing those lines.
def draw_ROI_lines(image):
    img = np.copy(image)
    (height, width) = img.shape [:2]
    vertices = [(120, height), (580, 450), (width-590, 450), (width-120, height)]

    cv2.line(img, vertices[0], vertices[1],(255,0,0), 5)
    cv2.line(img, vertices[1], vertices[2],(255,0,0), 5)
    cv2.line(img, vertices[2], vertices[3],(255,0,0), 5)
    
    plt.imshow(img)


# Warp image - Perform Perspective Transform
def get_lane_perspective(image):
    img = np.copy(image)
    (height, width) = img.shape[:2]

    # Specify source points to transform
    top_left_corner = (588, 450)
    top_right_corner = (width-588, 450)
    bottom_right_corner = (width-120, height)
    bottom_left_corner = (120, height)
    src = np.float32([top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner])
    
    # Specify destination points to transform to
    dst_offset = width * 0.2
    dst = np.float32([[dst_offset,0], [width-dst_offset,0], [width-dst_offset, height], [dst_offset,height]])
    
    # Generate transformation matrices
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Warp image to return
    warped = cv2.warpPerspective(img, M, (width, height), flags = cv2.INTER_LINEAR)

    return warped, M, Minv


# Detect lane lines and fit polynomials
def hist(img):
    height, width = img.shape
    histogram = np.sum(img[height*2//3:, :], axis=0)
    return histogram


def find_lane_pixels(warped_image):
    # Get histogram to determine starting position of each lane line
    histogram = hist(warped_image)
    
    # Create an output image for drawing and visualization
    out_img = np.dstack((warped_image, warped_image, warped_image))*255

    # Split the image vertically into left and right halves
    midpoint = np.int(histogram.shape[0]//2)
    
    # Find the location of each lane line from the bottom of the image
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:])+midpoint

    # Set up sliding windows and hyperparameters
    nWindows = 10
    margin = 75
    min_pix = 50

    # Establish window height
    window_height = np.int(warped_image.shape[0]//nWindows)

    # Determine the x and y-positions of all pixels in the image
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set current position
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create lists for left and right lane indices
    left_lane_inds = []
    right_lane_inds = []

    # Fit a polynomial to the lane lines
    # Step through the windows
    for window in range(nWindows):
        # Identify the horizontal boundaries for the windows
        win_y_low = warped_image.shape[0] - (window + 1) * window_height
        win_y_high = warped_image.shape[0] - window * window_height
        win_x_left_low = leftx_current - margin
        win_x_left_high = leftx_current + margin
        win_x_right_low = rightx_current - margin
        win_x_right_high = rightx_current + margin

        # Draw the rectangular windows on the image
        cv2.rectangle(out_img, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (0,255,0), 2)

        # Identify the nonzero pixels inside of the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_x_left_low) & (nonzerox < win_x_left_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_right_low) & (nonzerox < win_x_right_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If the quantity of pixels found is greater than min_pix, re-center the next window
        # based upon their mean position.
        if len(good_left_inds) > min_pix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > min_pix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of lane line indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right positions of the pixels in the lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(warped_image):
    # Get lane pixels
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped_image)
    plt.imshow(out_img)
    
    # Fit a second order polynomial to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate coordinates for plotting
    ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Generate output image
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    plt.plot(left_fitx, ploty, color='yellow', lw=3)
    plt.plot(right_fitx, ploty, color='yellow', lw=3)
    
    return out_img, left_fit, right_fit, ploty


def fit_poly(img_shape, leftx, lefty, rightx, righty):
  
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fit = None
    right_fit = None
    left_fitx = None
    right_fitx = None 
    
    # Fit a second order polynomial to each lane line and generate coordinates for plotting
    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    
    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fit, left_fitx, right_fit, right_fitx, ploty


def search_around_poly(warped_image):
    margin = 100
    
    # Determine the x and y-positions of all pixels in the image
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped_image)
    
    # Fit new polynomials to the lane lines
    left_fit, left_fitx, right_fit, right_fitx, ploty = fit_poly(warped_image.shape, leftx, lefty, rightx, righty)
    
    # Determine the search area based upon the margin
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                     (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                      (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Create visualization
    out_img = np.dstack((warped_image, warped_image, warped_image))*255
    window_img = np.zeros_like(out_img)
    
    # Add color to lane line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255,0,0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0,0,255]
    
    # Generate a polygon to identify the search window area
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane lines onto the image
    cv2.fillPoly(window_img, np.int_([left_line_pts]),(0,255,0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]),(0,255,0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    
    return result


def search_around_poly2(warped_image, prev_left_fit, prev_right_fit):
    '''
        This function can be used to look around the previously found polynomial within
        the specified margin parameter to determine if another lane line can be found.
        
        This can help speed up the lane finding processing time for videos.
    '''
    margin = 100
    
    # Determine the x and y-positions of all pixels in the image
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
        
    # Determine the search area based upon the margin
    left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] - margin)) &
                     (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2] - margin)) &
                      (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin)))
    
    # Extract left and right positions of the pixels in the lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty