'''
Create an instance of a Camera that will store calibration matrix
information. Export it out to a pickle file, which can then be
reimported later when selecting a camera for use with the main
pipeline.
'''

import cv2, os, glob
import numpy as np
import matplotlib.image as mpimg

class Camera():
    def __init__(self):
        self.obj_points = []
        self.img_points = []
        self.mtx = None
        self.dist = None
        self.ret = None
        self.rvecs = None
        self.tvecs = None

        self.nx = 9
        self.ny = 6
        #self.objp = None

    
    def calibrate(self, image_dir):
        '''
        Take an image, calibrate the camera, and then return the 
        undistorted version of the provided image.
        Store the obj_points and img_points to the member variables.
        '''
        image_files = glob.glob(image_dir)
        self.generate_calibration_points(image_files)

        calibration_image = mpimg.imread(image_files[0])
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, calibration_image.shape[1:3], None, None)
        

    def undistort(self, image):
        '''
        Undistort an image from provided image and camera calibration
        values, then return it.
        '''
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    
    # TODO: Add Chessboard Corners detection function to implement calibration matrix
    def findChessboardCorners(self, image):
        return cv2.findChessboardCorners(image, (self.nx, self.ny), None)

    
    def drawChessboardCorners(self, image, corners, ret):
        return cv2.drawChessboardCorners(image, (self.nx, self.ny), corners, ret)

    
    def generate_calibration_points(self, image_files):
        objp = np.zeros((self.ny*self.nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)
        
        for fn in image_files:
            image = np.copy(mpimg.imread(fn))

            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            if ret == True:
                self.img_points.append(corners)
                self.obj_points.append(objp)