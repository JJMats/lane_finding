# Image filtering functions
import numpy as np
import matplotlib.pyplot as plt
import cv2
import helpers

def abs_sobel_thresh(image, orient='x', kernel_size=3, thresh=(0, 255)):
    img = np.copy(image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #Take the derivative in the axis specified by the orient parameter
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, kernel_size)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, kernel_size)
        
    # Get the absolute value of the derivative
    sobel_abs = np.absolute(sobel)
    
    # Scale absolute value to 8-bits
    sobel_norm = np.uint8(255*sobel_abs/np.max(sobel_abs))
    
    # Create a mask of 1's where the scaled derivative magnitude is between the threshold parameters
    sobel_bin = np.zeros_like(sobel_norm)
    sobel_bin[(sobel_norm >= thresh[0]) & (sobel_norm <= thresh[1])] = 1
    
    return sobel_bin


def mag_thresh(image, kernel_size=3, mag_thresh=(0,255)):
    img = np.copy(image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate the derivative in x and y-directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    # Calculate the magnitude of the derivative
    sobel_magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    
    # Normalize magnitude to 8-bit value
    sobel_magnitude_norm = np.uint8(255 * sobel_magnitude/np.max(sobel_magnitude))
    
    # Create binary mask of values within mag_threshold
    sobel_bin = np.zeros_like(sobel_magnitude_norm)
    sobel_bin[(sobel_magnitude_norm >= mag_thresh[0]) & (sobel_magnitude_norm <= mag_thresh[1])] = 1
    
    # Return the mask
    return sobel_bin


# Implement directional filtering
def dir_threshold(image, kernel_size=3, thresh=(0, np.pi/2)):
    img = np.copy(image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate the derivative in x and y-directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    # Calculate the absolute value of each derivative
    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)
    
    # Calculate the direction of the gradient
    direction = np.arctan2(sobely_abs, sobelx_abs)
    
    binary_output = np.zeros_like(gray)
    binary_output[(direction >= thresh[0])&(direction <= thresh[1])] = 1
    
    return binary_output


def select_lines_in_hls(image):
    img = np.copy(image)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # Mask for yellow lines
    lower_thresh = np.uint8([10, 0, 100])
    upper_thresh = np.uint8([40, 255, 255])
    yellow_line_mask = cv2.inRange(hls, lower_thresh, upper_thresh)
    
    # Mask for white lines
    lower_thresh = np.uint8([0,180,0])
    upper_thresh = np.uint8([255,255,255])
    white_line_mask = cv2.inRange(hls, lower_thresh, upper_thresh)
    
    # Combine color masks
    cmb_mask = cv2.bitwise_or(yellow_line_mask, white_line_mask)
    return cv2.bitwise_and(img, img, mask = cmb_mask)


def select_lines_in_colorspaces(image):
    img = np.copy(image)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    
    s = hls[:,:,2]
    s_lower_thresh = 180
    s_upper_thresh = 255
    s_binary = np.zeros_like(s)
    s_binary[(s >= s_lower_thresh) & (s <= s_upper_thresh)] = 1
    
    b = lab[:,:,2]
    b_lower_thresh = 160
    b_upper_thresh = 200
    b_binary = np.zeros_like(b)
    b_binary[(b >= b_lower_thresh) & (b <= b_upper_thresh)] = 1
    
    l = luv[:,:,0]
    l_lower_thresh = 210
    l_upper_thresh = 255
    l_binary = np.zeros_like(l)
    l_binary[(l >= l_lower_thresh) & (l <= l_upper_thresh)] = 1
    
    binary_comb = np.zeros_like(s)
    binary_comb[(l_binary == 1) | (b_binary == 1)] = 1
    
    return binary_comb


# Implement HLS (s-channel) filtering
def apply_saturation_mask(image):
    img = np.copy(image)
    
    s = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
    
    thresh = (100, 255)
    binary_output = np.zeros_like(s)
    binary_output[(s >= thresh[0]) & (s <= thresh[1])] = 1

    return binary_output


# Combine color and gradient filtering
def apply_saturation_and_gradient_masks(image):
    img = np.copy(image)
    
    kernel_size = 3
    s = apply_saturation_mask(image)
    dx = abs_sobel_thresh(img, 'x', kernel_size, (20, 100))
    dy = abs_sobel_thresh(img, 'y', kernel_size, (20, 100))
    mag = mag_thresh(img, kernel_size, (30, 100))
    direction = dir_threshold(img, kernel_size, (0.7, 1.3))
    comb = np.zeros_like(direction)    
    #comb[((dx == 1) & (dy == 1)) | ((mag == 1) & (direction == 1)) | (s == 1)] = 1
    comb[((dx == 1) & (dy == 1)) | (s == 1)] = 1
    
    return comb


# Tunable Filter Hyperparameters
def apply_saturation_mask2(image_id, kernel_size, thresh_low, thresh_high):
    img = np.copy(helpers.test_images_undst[image_id])
    warp, m, mInv = helpers.get_lane_perspective(img)
    hls_img = select_lines_in_hls(warp)
    
    blur = cv2.GaussianBlur(hls_img, (kernel_size, kernel_size), 0)
    
    s = cv2.cvtColor(blur, cv2.COLOR_RGB2HLS)[:,:,2]
    
    thresh = (thresh_low, thresh_high)
    binary_output = np.zeros_like(s)
    binary_output[(s >= thresh[0]) & (s <= thresh[1])] = 1

    plt.imshow(binary_output, cmap='gray')


def apply_sat_and_grad_masks(img_id, ksize, dxl, dxh, dyl, dyh, magl, magh, dirl, dirh):
    img = np.copy(helpers.test_images_undst[img_id])
    warp, m, mInv = helpers.get_lane_perspective(img)
    
    kernel_size = ksize
    s = apply_saturation_mask(warp)
    dx = abs_sobel_thresh(warp, 'x', kernel_size, (dxl, dxh))#20,100
    dy = abs_sobel_thresh(warp, 'y', kernel_size, (dyl, dyh)) #20,100
    mag = mag_thresh(warp, kernel_size, (magl, magh)) #30,100
    print(dirl, dirh)
    direction = dir_threshold(img, kernel_size, (dirl, dirh)) #0.7, 1.3
    comb = np.zeros_like(direction)    
    comb[((dx == 1) & (dy == 1)) | ((mag == 1) & (direction == 1)) | (s == 1)] = 1
    plt.imshow(comb)


def apply_colorspace_masks(img_id, satl, sath, bl, bh, ll, lh):
    img = np.copy(helpers.test_images_undst[img_id])
    warp, m, mInv = helpers.get_lane_perspective(img)

    hls = cv2.cvtColor(warp, cv2.COLOR_RGB2HLS)
    lab = cv2.cvtColor(warp, cv2.COLOR_RGB2Luv)
    luv = cv2.cvtColor(warp, cv2.COLOR_RGB2Lab)
    
    s = hls[:,:,2]
    s_binary = np.zeros_like(s)
    s_binary[(s >= satl) & (s <= sath)] = 1
    
    b = lab[:,:,2]
    b_binary = np.zeros_like(b)
    b_binary[(b >= bl) & (b <= bh)] = 1
    
    l = luv[:,:,0]
    l_binary = np.zeros_like(l)
    l_binary[(l >= ll) & (l <= lh)] = 1
    
    binary_comb = np.zeros_like(s)
    binary_comb[((l_binary == 1) | (b_binary == 1)) & (s_binary == 1)] = 1
    
    plt.imshow(binary_comb, cmap='gray')