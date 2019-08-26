import os, glob, cv2
import numpy as np
import matplotlib.image as mpimg

# Get camera calibration images
# TODO: Move this into a function that will calibrate the camera
#       and return the calibration matrix.
camera_cal_img_files = glob.glob("images/camera_cal/calibration*")

def cal_camera(img, obj_points, img_points):
    '''
    Take an image, object points, and image points, calibrate the camera,
    then correct and return an undistorted version of the image.

    Returns ret, mtx, dist, rvecs, tvecs
    '''

    return cv2.calibrateCamera(obj_points, img_points, img.shape[1:3], None, None)


def undistort_image(img, mtx, dist):
    '''
    Take an image and camera calibration values, undistort the image, and 
    return it.
    '''

    return cv2.undistort(img, mtx, dist, None, mtx)


def import_calibration_images(img_points, obj_points):
    # Prepare object points
    nx = 9
    ny = 6

    # Get known coordinates of the corners in the image
    objp = np.zeros((ny*nx, 3), np.float32)

    # Generate x, y-coordinates
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    for fname in camera_cal_img_files:
        image = mpimg.imread(fname)
        img = np.copy(image)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            img_points.append(corners)
            obj_points.append(objp)
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)


def undistort_image(image):
    '''
    Take an image as input, undistort with learned camera calibration
    values.
    '''
    obj_points = []
    img_points = []

    ret, mtx, dist, rvecs, tvecs = cal_camera(image, obj_points, img_points)
    img_undst = undistort_image(image, mtx, dist)

    return img_undst
