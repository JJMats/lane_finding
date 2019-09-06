## Lane Finding

### Description

Lane finding app for the Jetson Nano Development Board.

This is a work in progress! The plan here is to extract the filters, helper functions, and classes from the Advanced Lane Finding Jupyter notebook into Python scripts. Then, I would like to assemble a pipeline to pull images from a live camera and overlay the detected lane image on top of them. This video can then be saved to disk for later review.

Additional thoughts include ROS integration, receiving vehicle CAN data for additional live overlays (throttle position, brake, vehicle speed, etc.). I also plan to rewrite this all in C++.
