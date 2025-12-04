import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from djitellopy import Tello
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


import tf2_ros
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import MarkerArray

from tello_controller import tello_constants as tc

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

        # tags/map/drone tf publishers
        # First tag becomes origin, try using mostly relative positions
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Create subscribers
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
        self.tag_map = {}
        self.map_frame = None

        # Camera intrinsics (from constants file)
        self.camera_matrix = tc.CAMERA_MATRIX
        self.dist_coeffs = tc.DISTORTION_COEFFS
        self.aruco_dict = tc.ARUCO_DICT
        self.aruco_params = tc.ARUCO_PARAMS
        self.marker_size_m = tc.MARKER_SIZE_M

        # 3D marker corner points (in marker's local frame)
        self.marker_points_3d = np.array([
            [-self.marker_size_m/2,  self.marker_size_m/2, 0],  # top-left
            [ self.marker_size_m/2,  self.marker_size_m/2, 0],  # top-right
            [ self.marker_size_m/2, -self.marker_size_m/2, 0],  # bottom-right
            [-self.marker_size_m/2, -self.marker_size_m/2, 0],  # bottom-left
        ], dtype=np.float32)

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

        # Check if any markers were detected
        if ids is None or len(ids) == 0:
            return  # No markers found

        # Pose estimation for each detected marker
        for i, marker_id in enumerate(ids.flatten()):
            frame_name = f"tag_{marker_id}"

            # Get 2D corners for this marker
            corners_2d = corners[i][0]  # Shape: (4, 2)

            # Solve PnP to get marker pose relative to camera
            success, rvec, tvec = cv2.solvePnP(
                self.marker_points_3d,  # 3D points in marker frame
                corners_2d,              # 2D points in image
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                self.get_logger().warn(f"Failed to solve pose for ArUco tag {marker_id}")
                continue


            # Invert transformation: T_camera_marker -> T_marker_camera
            R_camera_marker = cv2.Rodrigues(rvec)[0]  # Convert rvec to rotation matrix
            R_marker_camera = R_camera_marker.T        # Invert rotation (transpose)
            tvec_marker_camera = -R_marker_camera @ tvec  # Invert translation

            # Convert inverted rotation matrix to quaternion
            rot_marker_camera = Rotation.from_matrix(R_marker_camera)
            quat_drone = rot_marker_camera.as_quat()  # Returns [x, y, z, w]

            tvec_drone = tvec_marker_camera

            # Skip publishing if transformation not implemented yet
            if tvec_drone is None or quat_drone is None:
                continue

            # Publish transform: tag_N -> camera_link
            aruco_to_drone = TransformStamped()
            aruco_to_drone.header.frame_id = frame_name
            aruco_to_drone.header.stamp = self.get_clock().now().to_msg()
            aruco_to_drone.child_frame_id = "camera_link"

            aruco_to_drone.transform.translation.x = float(tvec_drone[0])
            aruco_to_drone.transform.translation.y = float(tvec_drone[1])
            aruco_to_drone.transform.translation.z = float(tvec_drone[2])

            # scipy returns [x, y, z, w] order
            aruco_to_drone.transform.rotation.x = float(quat_drone[0])
            aruco_to_drone.transform.rotation.y = float(quat_drone[1])
            aruco_to_drone.transform.rotation.z = float(quat_drone[2])
            aruco_to_drone.transform.rotation.w = float(quat_drone[3])

            self.tf_broadcaster.sendTransform(aruco_to_drone)

            # If first marker detected, set it as the map origin
            if self.map_frame is None:
                self.map_frame = f"tag_{marker_id}"

                # Publish map -> tag_N transform (identity)
                map_frame_origin = TransformStamped()
                map_frame_origin.header.frame_id = "map"
                map_frame_origin.header.stamp = self.get_clock().now().to_msg()
                map_frame_origin.child_frame_id = frame_name

                map_frame_origin.transform.translation.x = 0.0
                map_frame_origin.transform.translation.y = 0.0
                map_frame_origin.transform.translation.z = 0.0

                # Identity quaternion
                map_frame_origin.transform.rotation.x = 0.0
                map_frame_origin.transform.rotation.y = 0.0
                map_frame_origin.transform.rotation.z = 0.0
                map_frame_origin.transform.rotation.w = 1.0

                self.tf_broadcaster.sendTransform(map_frame_origin)
                self.get_logger().info(f"Set map origin to tag_{marker_id}")





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
        node.get_logger().info("Tello Environment Node started. Waiting for camera frames...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()