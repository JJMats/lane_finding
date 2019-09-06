import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import helpers
import line
import filters

# Use the interact widget to help with parameter tuning
from IPython.html.widgets import *

def find_lane(camera, image):    
    img = np.copy(image)
    
    #Undistort image
    undist_img = camera.undistort(img)
    
    # Apply various filters to obtain lane lines from image
    masked_img = filters.select_lines_in_colorspaces(undist_img)
    
    # Get lane perspective
    warped_image, M, Minv = helpers.get_lane_perspective(masked_img)
    
    if left_lane_line.detected and right_lane_line.detected:
        # Use the search around poly function here
        leftx, lefty, rightx, righty = helpers.search_around_poly2(warped_image, left_lane_line.get_last_fit(), right_lane_line.get_last_fit())
    else:            
        # Get lane line pixels via sliding window search
        leftx, lefty, rightx, righty, out_img = helpers.find_lane_pixels(warped_image)

    if len(leftx) > 0 and len(lefty) > 0 and len(rightx) > 0 and len(righty) > 0:    
        # Fit polynomial lines to lanes
        left_fit, left_fitx, right_fit, right_fitx, ploty = helpers.fit_poly(warped_image.shape, leftx, lefty, rightx, righty)

        # Get curve radius and lane offset from lane lines
        left_curve_rad, right_curve_rad, lane_center_offset = measure_curvature_pixels_and_lane_offset(warped_image.shape, ploty, left_fitx, right_fitx, left_fit, right_fit)


        # Add current fit to Line class for each lane line
        left_lane_line.add_current_fit(left_fit, left_fitx, lane_center_offset, left_curve_rad)
        right_lane_line.add_current_fit(right_fit, right_fitx, lane_center_offset, right_curve_rad)
    
    # Calculate average curve radius for the two lane lines
    curve_rad_avg = np.mean([left_lane_line.get_curve_radius(), right_lane_line.get_curve_radius()])
    
    # Get vehicle center offset from lane center
    lane_center_offset = left_lane_line.line_base_pos
    
    # Draw the best_fit (averaged) lane lines onto the image
    weighted_img = draw_lane(undist_img, warped_image, left_lane_line.get_best_fit(), right_lane_line.get_best_fit(), Minv)

    # Return the image with the radius of curvature and vehicle location information displayed
    return draw_curve_radius_info(weighted_img, curve_rad_avg, lane_center_offset)


# Calculate lane offset, then draw lane
def measure_curvature_pixels(ploty, leftx, rightx):
    '''
        Calculates the curvature of given polynomial functions in meters
    '''
    ym_per_pix = 30/720 # Meters per pixel in y-dimension
    xm_per_pix = 3.7/700 # Meters per pixel in x-dimension
    
    # Define the y-value at which the radius of curvature should be calculated. This will be the bottom of the image.
    y_eval = np.max(ploty)
    
    # Calculate the radius of curvature for each lane line
    left_curve_rad = None
    right_curve_rad = None
    
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit_m = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    if left_fit_m is not None:
        left_curve_rad = (((1 + (2*left_fit_m[0]*y_eval*ym_per_pix + left_fit_m[1])**2)**(3/2))/np.abs(2*left_fit_m[0]))
        
    if right_fit_m is not None:
        right_curve_rad = (((1 + (2*right_fit_m[0]*y_eval*ym_per_pix + right_fit_m[1])**2)**(3/2))/np.abs(2*right_fit_m[0]))
    
    return left_curve_rad, right_curve_rad


def measure_curvature_pixels_and_lane_offset(img_shape, ploty, leftx, rightx, lf, rf):
    '''
        Calculates the curvature of given polynomial functions in meters and calculates
         the distance between the centerline of the vehicle and the center of the lane.
        This assumes that the camera center is coincident with the centerline of the vehicle.
    '''
    height, width = img_shape[:2]
    y_eval = height
    left_location = (lf[0]*y_eval**2 + lf[1]*y_eval + lf[2])
    right_location = (rf[0]*y_eval**2 + rf[1]*y_eval + rf[2])
    
    ym_per_pix = 30/720 # Meters per pixel in y-dimension
    xm_per_pix = 3.7/(right_location-left_location) # Meters per pixel in x-dimension
    
    # Calculate lane center offset from image center
    lane_center_location = (left_location + right_location) / 2
    lane_offset = lane_center_location - width / 2    
    lane_offset_m = lane_offset * xm_per_pix # Lane offset in meters
    
    # Calculate the radius of curvature for each lane line
    left_curve_rad = None
    right_curve_rad = None
    
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit_m = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    if left_fit_m is not None:
        left_curve_rad = (((1 + (2*left_fit_m[0]*y_eval*ym_per_pix + left_fit_m[1])**2)**(3/2))/np.abs(2*left_fit_m[0]))
        
    if right_fit_m is not None:
        right_curve_rad = (((1 + (2*right_fit_m[0]*y_eval*ym_per_pix + right_fit_m[1])**2)**(3/2))/np.abs(2*right_fit_m[0]))
    
    return left_curve_rad, right_curve_rad, lane_offset_m


def draw_lane(prev_image, warped_image, left_fit, right_fit, minv):
    img = np.copy(prev_image)
    
    if left_fit is None or right_fit is None:
        return img
    
    # Create an image to draw lane lines on
    warped_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warped_zero, warped_zero, warped_zero))
    
    height, width = warped_image.shape[:2]
    ploty = np.linspace(0, height-1, height)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]    
    
    left_px = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_px = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pix = np.hstack((left_px, right_px))
    
    # Draw the lane back onto the warped image
    cv2.fillPoly(color_warp, np.int_([pix]), (0,255,0))
    cv2.polylines(color_warp, np.int32([left_px]), isClosed=False, color=(255,0,0), thickness=40)
    cv2.polylines(color_warp, np.int32([right_px]), isClosed=False, color=(0,0,255), thickness=40)
    
    # Unwarp the warped image
    unwarped_img = cv2.warpPerspective(color_warp, minv, (width, height))
    
    # Overlay the drawn lane onto the original image
    return cv2.addWeighted(img, 1, unwarped_img, 0.5, 0)


def draw_curve_radius_info(image, curve_rad, lane_center_dist):
    img = np.copy(image)
    height = img.shape[0]    
    font = cv2.FONT_HERSHEY_TRIPLEX
    
    radius_text = 'Radius of curvature: '
    # Generate radius of curvature text
    if curve_rad < 1000:        
        radius_text += '{:-4.2f}'.format(curve_rad) + 'm'
    else:
        radius_text += '{:-4.2f}'.format(curve_rad/1000) + 'km'
    
    # Generate lane center location text
    dist_from_center = '{:-4.3f}'.format(abs(lane_center_dist)) + 'm '
    loc_from_center = ''
    if lane_center_dist < 0:
        loc_from_center = dist_from_center + 'right of center'
    elif lane_center_dist > 0:
        loc_from_center = dist_from_center + 'left of center'
    else:
        loc_from_center = 'On center'
    
    location_text = 'Vehicle location: ' + loc_from_center
    
    # Add text to image    
    cv2.putText(img, radius_text, (50, 50), font, 1.5, (200, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(img, location_text, (50, 100), font, 1.5, (200, 255, 200), 2, cv2.LINE_AA)
    return img


left_lane_line = line.Line()
right_lane_line = line.Line()
#lane_img = find_lane(test_images_undst[3])
#plt.imshow(lane_img)
#plt.imsave("output_images/misc_images/lane_location_info.jpg", lane_img)