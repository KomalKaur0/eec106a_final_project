import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from djitellopy import Tello
import cv2
import numpy as np


import tf2_ros
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray

import tello_constants

class TelloEnvironmentNode(Node):
    """
    ROS2 Node for creating and publishing the environment containing the drone
    """
    def __init__(self):
        """Initialize Node"""
        super().__init__('tello_environment_node')

        # Create publishers
        # Perhaps a global position publisher? It will use some combination of the drone's position and the tags
        # And also a publisher that shows the positions and orientations of the arucotags/already mapped areas? Probably a later task
        # Maybe that will be a dictionary of arucotags and their global positions. Perhaps this should be a service
        # I don't know what rviz needs to put the visuals of the room

        # drone pose publisher
        self.drone_pose_publisher = self.create_publisher(
            PoseStamped,
            '/tello/drone_pose',
            9
        )
        
        # aruco tags publisher
        self.aruco_pose_publisher = self.create_publisher(
            MarkerArray,
            '/world/aruco_poses',
            9
        )

        # TODO: tags/map/drone tf publishers
        # First tag becomes origin, try using mostly relative positions
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # TODO: Create subscribers
        # Subscribe to camera feed
        self.camera_subscriber = self.create_subscription(
            Image,
            '/tello/camera/image_raw',
            self.camera_callback,
            10
        )

        # TODO: Subscribe to raw drone position feed
        # Needs to be implemented elsewhere first
        
        # State stuff
        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X_1000)
        self.aruco_params = cv2.aruco.DetectorParameters

        self.tag_map = {}
        self.map_frame = None

    def camera_callback(self, msg: Image):
        """
        Handles interpreting camera data:
        1. update positions based on arucotags
        2. update arucotags themselves
        3. update environment data (Later)
        """
        # Get frame from image
        frame = self.bridge.imgmsg_to_cv2(msg)
        
        if frame is None:
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        # Pose estimation
        for i, marker_id in enumerate(ids.flatten()):
            success, rvec, tvec = cv2.solvePnP(...)

            if not success:
                self.get_logger.warn(f"failed aruco position solve for arucotag {marker_id}")
                continue

            # TODO: Invert
            # TODO: Publish transform
            # TODO: if first marker, set as map origin
            if self.map_frame == None:
                self.map_frame = f"tag_{marker_id}"



    def global_position():
        """Publishes global position"""
        # TODO:
        pass

    def update_arucotags(self):
        """Updates positions of aruco_tags when seen"""
        # Multi-tag
        # Single-tag (defers to multi tag?)
        pass

    def environment():
        """Publishes other environment data?"""
        # I just want to see what that camera has seen of the environment. 
        # Will need to update old stuff when it sees new?
        pass
    
def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)

    # Create node
    node = TelloEnvironmentNode()

    try:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        pass

if __name__ == '__main__':
    main()