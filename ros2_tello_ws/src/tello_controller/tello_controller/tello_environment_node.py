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
from tello_interfaces.msg import TelloTelemetry

from tello_controller import tello_constants as tc

class TelloEnvironmentNode(Node):
    """
    ROS2 Node for creating and publishing the environment containing the drone
    """
    def __init__(self):
        """
        Initialize Node
        """
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

        # Subscribe to drone telemetry
        self.telemetry_subscriber = self.create_subscription(
            TelloTelemetry,
            '/tello/telemetry',
            self.telemetry_callback,
            10
        )

        # State stuff
        self.bridge = CvBridge()
        self.tag_map = {}  # Persistent map of all seen tags
        self.map_frame = None  # First tag becomes map origin

        # Latest telemetry data for sensor fusion
        self.latest_telemetry = None

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

        # Timer to publish static map transforms
        self.map_timer = self.create_timer(0.1, self.publish_static_map)  # 10 Hz

        self.get_logger().info("Tello Environment Node initialized")

    def camera_callback(self, msg: Image):
        """
        Handles interpreting camera data:
        1. Detect ArUco tags
        2. Compute camera pose relative to each tag
        3. Register new tags or update existing ones
        4. Handle multi-tag registration
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

        # Store poses of all currently visible tags
        current_observations = {}  # {marker_id: (tvec_camera, quat_camera, R_marker_camera)}

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

            # Compute reprojection error to assess detection quality
            projected_corners, _ = cv2.projectPoints(
                self.marker_points_3d, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            reprojection_error = np.mean(np.linalg.norm(
                corners_2d - projected_corners.reshape(-1, 2), axis=1
            ))

            # Log quality metrics for problematic tags
            if reprojection_error > 2.0:
                distance = np.linalg.norm(tvec)
                self.get_logger().info(
                    f"Tag {marker_id}: reproj_error={reprojection_error:.2f}px, "
                    f"distance={distance:.3f}m"
                )

            # Invert transformation: T_camera_marker -> T_marker_camera
            R_camera_marker = cv2.Rodrigues(rvec)[0]  # Convert rvec to rotation matrix
            R_marker_camera = R_camera_marker.T        # Invert rotation (transpose)
            tvec_marker_camera = -R_marker_camera @ tvec  # Invert translation

            # Convert inverted rotation matrix to quaternion
            rot_marker_camera = Rotation.from_matrix(R_marker_camera)
            quat_camera = rot_marker_camera.as_quat()  # Returns [x, y, z, w]

            # Store this observation for multi-tag registration
            # Flatten tvec to ensure it's 1D array shape (3,) not (3,1)
            current_observations[marker_id] = {
                'tvec': tvec_marker_camera.flatten(),
                'quat': quat_camera.copy(),
                'R': R_marker_camera.copy()
            }

        # After processing all visible tags, update the persistent map
        self.update_tag_map(current_observations)

        # Publish ArUco markers for rtab-map
        if len(current_observations) > 0:
            self.publish_aruco_markers(current_observations)

        # Publish camera pose in map frame (avoids conflicts from multiple visible tags)
        if len(current_observations) > 0 and self.map_frame is not None:
            self._publish_camera_pose(current_observations)

    def telemetry_callback(self, msg: TelloTelemetry):
        """Store latest telemetry for sensor fusion with AR tag odometry."""
        if not msg.telemetry_valid:
            self.get_logger().warn('Received invalid telemetry, ignoring')
            return

        self.latest_telemetry = msg

        # Log periodically (every 90 messages = 3 sec at 30Hz)
        if not hasattr(self, '_telemetry_count'):
            self._telemetry_count = 0

        self._telemetry_count += 1
        if self._telemetry_count % 90 == 0:
            self.get_logger().info(
                f'Telemetry: height={msg.height:.1f}cm, '
                f'battery={msg.battery}%, '
                f'attitude=({msg.roll:.1f}, {msg.pitch:.1f}, {msg.yaw:.1f})°'
            )


    def update_tag_map(self, observations):
        """
        Update the persistent tag map with new observations.
        Handles both single-tag and multi-tag registration.

        Args:
            observations: dict of {marker_id: {'tvec': ..., 'quat': ..., 'R': ...}}
        """
        if len(observations) == 0:
            return

        # If this is the first tag ever seen, set it as map origin
        if self.map_frame is None:
            first_id = list(observations.keys())[0]
            self.map_frame = f"tag_{first_id}"

            # Add origin tag at identity pose
            self.tag_map[first_id] = {
                'position': np.array([0.0, 0.0, 0.0]),  # Origin
                'orientation': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
                'R': np.eye(3),  # Identity rotation matrix
                'observations': 1,
                'last_seen': self.get_clock().now(),
                'is_origin': True
            }
            self.get_logger().info(f"Set map origin to tag_{first_id}")

        # Update all observed tags
        for marker_id, obs in observations.items():
            if marker_id not in self.tag_map:
                # New tag detected!
                if marker_id == int(self.map_frame.split('_')[1]):
                    # This is the origin tag, already added
                    continue

                # Register new tag using multi-tag registration
                self._register_new_tag(marker_id, obs, observations)
            else:
                # Update existing tag
                self.tag_map[marker_id]['observations'] += 1
                self.tag_map[marker_id]['last_seen'] = self.get_clock().now()
                # TODO: Could implement pose refinement here (e.g., averaging)

    def _register_new_tag(self, new_id, new_obs, all_observations):
        """
        Register a new tag by computing its position relative to a known tag.

        Args:
            new_id: ID of the new tag
            new_obs: Observation data for the new tag
            all_observations: All tags visible in current frame
        """
        # Find a known tag in the current observations
        known_tag_id = None
        for obs_id in all_observations.keys():
            if obs_id in self.tag_map:
                known_tag_id = obs_id
                break

        if known_tag_id is None:
            self.get_logger().warn(f"Cannot register tag_{new_id}: no known tags visible")
            return

        # Multi-tag registration:
        # We know: camera pose relative to known_tag (T_known_camera)
        # We know: camera pose relative to new_tag (T_new_camera)
        # We want: new_tag pose relative to known_tag (T_known_new)
        # Math: T_known_new = T_known_camera @ inverse(T_new_camera)

        known_obs = all_observations[known_tag_id]

        # T_known_camera (camera in known tag frame)
        R_known_camera = known_obs['R']
        t_known_camera = known_obs['tvec']

        # T_new_camera (camera in new tag frame)
        R_new_camera = new_obs['R']
        t_new_camera = new_obs['tvec']

        # Invert T_new_camera to get T_camera_new
        R_camera_new = R_new_camera.T
        t_camera_new = (-R_camera_new @ t_new_camera).flatten()

        # Compose: T_known_new = T_known_camera @ T_camera_new
        R_known_new = R_known_camera @ R_camera_new
        t_known_new = (R_known_camera @ t_camera_new + t_known_camera).flatten()

        # Convert rotation to quaternion
        rot = Rotation.from_matrix(R_known_new)
        quat_known_new = rot.as_quat()  # [x, y, z, w]

        # Now transform from known_tag frame to map frame
        known_tag_data = self.tag_map[known_tag_id]

        # T_map_new = T_map_known @ T_known_new
        R_map_known = known_tag_data['R']
        t_map_known = known_tag_data['position']

        R_map_new = R_map_known @ R_known_new
        t_map_new = (R_map_known @ t_known_new + t_map_known).flatten()

        # Convert final rotation to quaternion
        rot_final = Rotation.from_matrix(R_map_new)
        quat_map_new = rot_final.as_quat()

        # Store in map
        self.tag_map[new_id] = {
            'position': t_map_new,
            'orientation': quat_map_new,
            'R': R_map_new,
            'observations': 1,
            'last_seen': self.get_clock().now(),
            'is_origin': False
        }

        self.get_logger().info(
            f"Registered new tag_{new_id} at position "
            f"[{t_map_new[0]:.3f}, {t_map_new[1]:.3f}, {t_map_new[2]:.3f}] "
            f"relative to map origin"
        )

    def _publish_camera_pose(self, observations):
        """
        Publish camera pose in map frame using visible tag observations.

        Strategy: Average all visible known tags for robustness.

        Args:
            observations: dict of {marker_id: {'tvec': ..., 'quat': ..., 'R': ...}}
        """
        # Collect camera pose estimates from all known visible tags
        camera_poses = []  # List of (t_map_camera, R_map_camera) from each tag

        for marker_id, obs in observations.items():
            if marker_id not in self.tag_map:
                continue  # Skip unknown tags

            # Get camera pose relative to this tag (fresh observation)
            t_tag_camera = obs['tvec']
            R_tag_camera = obs['R']

            # Get tag pose in map frame (from persistent map)
            tag_data = self.tag_map[marker_id]
            t_map_tag = tag_data['position']
            R_map_tag = tag_data['R']

            # Transform camera pose to map frame: T_map_camera = T_map_tag @ T_tag_camera
            R_map_camera = R_map_tag @ R_tag_camera
            t_map_camera = (R_map_tag @ t_tag_camera + t_map_tag).flatten()

            camera_poses.append((t_map_camera, R_map_camera))

        if len(camera_poses) == 0:
            return  # No known tags visible

        # For now, use the first estimate (could average multiple later)
        t_map_camera, R_map_camera = camera_poses[0]

        # Debug: Log position every 30 frames (~1 second at 30fps)
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
            self._last_pos = None

        self._frame_count += 1
        if self._frame_count % 30 == 0:
            if self._last_pos is not None:
                delta = np.linalg.norm(t_map_camera - self._last_pos)
                self.get_logger().info(
                    f"Camera pos: [{t_map_camera[0]:.3f}, {t_map_camera[1]:.3f}, {t_map_camera[2]:.3f}] "
                    f"delta: {delta:.4f}m"
                )
            self._last_pos = t_map_camera.copy()

        # Convert rotation to quaternion
        rot = Rotation.from_matrix(R_map_camera)
        quat_map_camera = rot.as_quat()  # [x, y, z, w]

        # Publish map -> camera_link transform
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = "camera_link"

        transform.transform.translation.x = float(t_map_camera[0])
        transform.transform.translation.y = float(t_map_camera[1])
        transform.transform.translation.z = float(t_map_camera[2])

        transform.transform.rotation.x = float(quat_map_camera[0])
        transform.transform.rotation.y = float(quat_map_camera[1])
        transform.transform.rotation.z = float(quat_map_camera[2])
        transform.transform.rotation.w = float(quat_map_camera[3])

        self.tf_broadcaster.sendTransform(transform)

    def publish_aruco_markers(self, observations):
        """
        Publish detected ArUco markers for rtab-map landmark integration.

        Args:
            observations: dict of {marker_id: {'tvec': ..., 'quat': ..., 'R': ...}}
        """
        from visualization_msgs.msg import Marker

        marker_array = MarkerArray()

        for marker_id, obs in observations.items():
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'tello_camera'
            marker.ns = 'aruco'
            marker.id = int(marker_id)  # Convert numpy int to Python int
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position from tag detection
            marker.pose.position.x = float(obs['tvec'][0])
            marker.pose.position.y = float(obs['tvec'][1])
            marker.pose.position.z = float(obs['tvec'][2])

            # Orientation from quaternion
            marker.pose.orientation.x = float(obs['quat'][0])
            marker.pose.orientation.y = float(obs['quat'][1])
            marker.pose.orientation.z = float(obs['quat'][2])
            marker.pose.orientation.w = float(obs['quat'][3])

            # Size matches physical tag (15cm)
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.01

            # Color - orange for visibility
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.aruco_pose_publisher.publish(marker_array)

    def publish_static_map(self):
        """
        Publish transforms for all known tags in the persistent map.
        """
        if self.map_frame is None or len(self.tag_map) == 0:
            return

        current_time = self.get_clock().now().to_msg()

        # Publish map -> tag_N transform for all known tags
        for marker_id, tag_data in self.tag_map.items():
            transform = TransformStamped()
            transform.header.stamp = current_time
            transform.header.frame_id = "map"
            transform.child_frame_id = f"tag_{marker_id}"

            # Position
            transform.transform.translation.x = float(tag_data['position'][0])
            transform.transform.translation.y = float(tag_data['position'][1])
            transform.transform.translation.z = float(tag_data['position'][2])

            # Orientation (quaternion [x, y, z, w])
            transform.transform.rotation.x = float(tag_data['orientation'][0])
            transform.transform.rotation.y = float(tag_data['orientation'][1])
            transform.transform.rotation.z = float(tag_data['orientation'][2])
            transform.transform.rotation.w = float(tag_data['orientation'][3])

            self.tf_broadcaster.sendTransform(transform)
    
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