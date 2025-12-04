"""
Tello Drone Constants

This module contains shared constants for the Tello drone system,
including camera calibration parameters and ArUco marker settings.
"""

import numpy as np
import cv2

# Approximate intrinsics

# Focal length in pixels (fx = fy for Tello)
FOCAL_LENGTH_PX = 921.170702

# Principal point (optical center) in pixels
PRINCIPAL_POINT_X = 459.904354  # cx
PRINCIPAL_POINT_Y = 351.238301  # cy

# Camera intrinsic matrix (3x3)
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH_PX, 0, PRINCIPAL_POINT_X],
    [0, FOCAL_LENGTH_PX, PRINCIPAL_POINT_Y],
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients [k1, k2, p1, p2, k3]
# Tello camera has minimal distortion, so these are typically near zero
DISTORTION_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# ==============================================================================
# ArUco Marker Settings
# ==============================================================================

# ArUco dictionary type (4x4 markers, 50 unique IDs)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# ArUco detector parameters (use defaults)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# Marker physical size in meters <- our printed markers are 15 cm
MARKER_SIZE_M = 0.15  # 15 cm

# ==============================================================================
# Camera Frame Settings
# ==============================================================================

# Frame dimensions (Tello camera resolution)
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

# Camera frame rate
CAMERA_FPS = 30
