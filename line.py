# Class to store information about lane line instances
import numpy as np

class Line():
    def __init__(self):
        # Set a boolean value to true of a line was detected in the last iteration
        self.detected = False
        
        # Create an array to store the last nFits of lane lines for averaging
        self.nFits = 5
        self.recent_fits = []
        self.recent_x_values = []        
        
        # Average polynomial coefficients over the last nFits iterations
        # This will return three elements [y**2, y, b]
        self.best_fit = None
        
        # Average the x values over the the last nFits iterations
        # This will return an average across each array for the corresponding y-value
        self.bestx = None
        
        # Polynomial coefficients for current fit
        self.current_fit = [np.array([False])]
        
        # Curvature of the turn radius in meters
        self.radius_of_curvature = 0
        
        # Distance in meters of vehicle center from the lane line
        self.line_base_pos = 0
        
        # Store the difference between fit coefficients between the current and last fit
        # This will be used to threshold bad lines
        self.diffs = np.array([0,0,0], dtype='float')
        
        # x-values for detected line pixels
        self.allx = None
        
        # y-values for detected line pixels
        self.ally = None
        
    def add_current_fit(self, fit, xvals, lane_offset, roc):
        if fit is not None:
            if self.best_fit is not None:
                if self.__validate_current_fit(fit):
                    self.__insert_fit(fit, xvals, lane_offset, roc)
                    if len(self.recent_fits) > self.nFits:
                        self.__drop_oldest_fit()
                else:
                    # This is a bad fit as the thresholds have been exceeded, do not keep
                    self.detected = False                    
            else:
                # The best fit has not been established. Update with this fit.
                if len(self.recent_fits) > 0:
                    self.drop_oldest_fit()
                    
                if len(self.recent_fits) == 0:
                    # Take this as the first fit
                    self.__insert_fit(fit, xvals, lane_offset, roc)
        else:
            self.detected = False

        
    def __calculate_best_fit(self):
        if len(self.recent_fits) > 0:
            self.best_fit = np.average(self.recent_fits, axis=0)
            self.bestx = np.average(self.recent_x_values, axis=0)
        else:
            self.best_fit = None
            self.bestx = None
        
    def __drop_oldest_fit(self):
        if len(self.recent_fits) > 0:
            self.recent_fits = self.recent_fits[1:]
            self.recent_x_values = self.recent_x_values[1:]
        
        # Recalculate best fit with fewer fits
        if len(self.recent_fits) > 0:
            self.__calculate_best_fit()
        
    def __insert_fit(self, fit, xvals, lane_offset, roc):
        self.detected = True
        self.line_base_pos = lane_offset
        self.radius_of_curvature = roc
        self.recent_x_values.append(xvals)
        self.recent_fits.append(fit)
        self.__calculate_best_fit()
        
    def __validate_current_fit(self, new_fit):
        # Thresholds must be set high enough that the vehicle can turn
        self.diffs = np.abs(new_fit - self.best_fit)
        if self.diffs[0] > 0.01 or self.diffs[1] > 1.0 or self.diffs[2] > 100.0:
            return False
        return True    
        
    def get_best_fit(self):
        self.__calculate_best_fit()
        return self.best_fit
    
    def get_last_fit(self):
        if len(self.recent_fits) > 0:
            return self.recent_fits[-1]
        else:
            return None
                
    def get_curve_radius(self):
        return self.radius_of_curvature
    
    def reset(self):
        '''
        Clear all instance variables
        '''
        self.detected = False
        self.nFits = 5
        self.recent_fits = []
        self.recent_x_values = []        
        self.best_fit = None
        self.bestx = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = 0
        self.line_base_pos = 0
        self.diffs = np.array([0,0,0], dtype='float')
        self.allx = None
        self.ally = None