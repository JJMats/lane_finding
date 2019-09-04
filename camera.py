'''
Create an instance of a Camera that will store calibration matrix
information. Export it out to a pickle file, which can then be
reimported later when selecting a camera for use with the main
pipeline.
'''

import cv2

class Camera():
    def __init__(self):
        self.obj_points = []
        self.img_points = []
        self.mtx = None
        self.dist = None
        self.ret = None
        self.rvecs = None
        self.tvecs = None
        

    def calibrate(self, image):
        '''
        Take an image, calibrate the camera, and then return the 
        undistorted version of the provided image.
        Store the obj_points and img_points to the member variables.
        '''
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, image.shape[1:3], None, None)
        

    def undistort(self, image):
        '''
        Undistort an image from provided image and camera calibration
        values, then return it.
        '''
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    # TODO: Add Chessboard Corners detection function to implement calibration matrix
