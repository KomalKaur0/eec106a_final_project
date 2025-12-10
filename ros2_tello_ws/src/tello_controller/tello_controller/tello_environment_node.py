#!/usr/bin/env python3
"""
Enhanced Tello Environment Node with Odometry and Complete TF Tree

Publishes:
- /tello/drone_pose (geometry_msgs/PoseStamped) - Drone pose in map frame
- /world/aruco_poses (visualization_msgs/MarkerArray) - All known AR tags
- /odom (nav_msgs/Odometry) - Odometry for RTAB-Map
- TF tree: map -> odom -> base_link -> camera_link
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import tf2_ros
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from tello_interfaces.msg import TelloTelemetry

from tello_controller import tello_constants as tc


class TelloEnvironmentNode(Node):
    """
    ROS2 Node for creating and publishing the environment containing the drone.
    
    Combines:
    - Visual odometry from AR tags
    - IMU data from Tello telemetry
    - Publishes complete TF tree for RTAB-Map
    """
    
    def __init__(self):
        super().__init__('tello_environment_node')

        # Create publishers
        self.drone_pose_publisher = self.create_publisher(
            PoseStamped,
            '/tello/drone_pose',
            10
        )
        
        self.aruco_pose_publisher = self.create_publisher(
            MarkerArray,
            '/world/aruco_poses',
            10
        )
        
        # Odometry publisher for RTAB-Map
        self.odom_publisher = self.create_publisher(
            Odometry,
            '/odom',
            10
        )

        # TF broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Create subscribers
        self.camera_subscriber = self.create_subscription(
            Image,
            '/tello/camera/image_raw',
            self.camera_callback,
            10
        )

        self.telemetry_subscriber = self.create_subscription(
            TelloTelemetry,
            '/tello/telemetry',
            self.telemetry_callback,
            10
        )

        # State
        self.bridge = CvBridge()
        self.tag_map = {}  # Persistent map of all seen tags
        self.map_frame = None  # First tag becomes map origin

        # Latest telemetry data for sensor fusion
        self.latest_telemetry = None

        # Camera intrinsics
        self.camera_matrix = tc.CAMERA_MATRIX
        self.dist_coeffs = tc.DISTORTION_COEFFS
        self.aruco_dict = tc.ARUCO_DICT
        self.aruco_params = tc.ARUCO_PARAMS
        self.marker_size_m = tc.MARKER_SIZE_M

        # 3D marker corner points (in marker's local frame)
        self.marker_points_3d = np.array([
            [-self.marker_size_m/2,  self.marker_size_m/2, 0],
            [ self.marker_size_m/2,  self.marker_size_m/2, 0],
            [ self.marker_size_m/2, -self.marker_size_m/2, 0],
            [-self.marker_size_m/2, -self.marker_size_m/2, 0],
        ], dtype=np.float32)

        # Odometry state
        self.last_odom_time = None
        self.odom_seq = 0

        # Camera pose for odometry
        self.last_camera_pose = None  # (position, orientation, timestamp)

        # Publish static transforms
        self.publish_static_transforms()

        # Timer to publish map transforms and odometry
        self.map_timer = self.create_timer(0.033, self.publish_dynamic_data)  # 30 Hz

        self.get_logger().info("Tello Environment Node initialized")
        self.get_logger().info("Ready for RTAB-Map integration")

    def publish_static_transforms(self):
        """
        Publish static TF transforms that don't change.
        
        TF Tree: map -> odom -> base_link -> camera_link
        
        Static: base_link -> camera_link (camera mounted on drone)
        """
        transforms = []
        
        # base_link -> camera_link
        # Tello camera is mounted facing forward, slightly downward
        # Adjust these values based on your physical setup
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'
        
        # Camera is ~5cm forward, 0cm sideways, -2cm down from center
        t.transform.translation.x = 0.05
        t.transform.translation.y = 0.0
        t.transform.translation.z = -0.02
        
        # Camera points forward with slight downward tilt (~15 degrees)
        # This is a typical Tello camera orientation
        rot = Rotation.from_euler('xyz', [0, 15, 0], degrees=True)
        quat = rot.as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        transforms.append(t)
        
        self.static_tf_broadcaster.sendTransform(transforms)
        self.get_logger().info("Published static TF: base_link -> camera_link")

    def camera_callback(self, msg: Image):
        """
        Process camera frames to detect AR tags and update pose.
        """
        frame = self.bridge.imgmsg_to_cv2(msg)
        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if ids is None or len(ids) == 0:
            return

        current_observations = {}

        # Process each detected marker
        for i, marker_id in enumerate(ids.flatten()):
            frame_name = f"tag_{marker_id}"
            corners_2d = corners[i][0]

            success, rvec, tvec = cv2.solvePnP(
                self.marker_points_3d,
                corners_2d,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                continue

            # Compute quality metrics
            projected_corners, _ = cv2.projectPoints(
                self.marker_points_3d, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            reprojection_error = np.mean(np.linalg.norm(
                corners_2d - projected_corners.reshape(-1, 2), axis=1
            ))

            if reprojection_error > 5.0:  # Skip low quality detections
                continue

            # Transform: camera pose in marker frame
            R_camera_marker = cv2.Rodrigues(rvec)[0]
            R_marker_camera = R_camera_marker.T
            tvec_marker_camera = -R_marker_camera @ tvec

            rot_marker_camera = Rotation.from_matrix(R_marker_camera)
            quat_camera = rot_marker_camera.as_quat()

            current_observations[marker_id] = {
                'tvec': tvec_marker_camera.flatten(),
                'quat': quat_camera.copy(),
                'R': R_marker_camera.copy(),
                'timestamp': msg.header.stamp
            }

        # Update persistent map
        self.update_tag_map(current_observations)

        # Publish camera pose and odometry
        if len(current_observations) > 0 and self.map_frame is not None:
            self._publish_camera_pose_and_odom(current_observations, msg.header.stamp)

    def telemetry_callback(self, msg: TelloTelemetry):
        """Store latest telemetry for sensor fusion."""
        if not msg.telemetry_valid:
            return
        self.latest_telemetry = msg

    def update_tag_map(self, observations):
        """
        Update the persistent tag map with new observations.
        """
        if len(observations) == 0:
            return

        # Initialize map with first tag
        if self.map_frame is None:
            first_id = list(observations.keys())[0]
            self.map_frame = f"tag_{first_id}"

            self.tag_map[first_id] = {
                'position': np.array([0.0, 0.0, 0.0]),
                'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
                'R': np.eye(3),
                'observations': 1,
                'last_seen': self.get_clock().now(),
                'is_origin': True
            }
            self.get_logger().info(f"Map origin set to tag_{first_id}")

        # Update existing tags or register new ones
        for marker_id, obs in observations.items():
            if marker_id not in self.tag_map:
                self._register_new_tag(marker_id, obs, observations)
            else:
                self.tag_map[marker_id]['observations'] += 1
                self.tag_map[marker_id]['last_seen'] = self.get_clock().now()

    def _register_new_tag(self, new_id, new_obs, all_observations):
        """
        Register a new tag using multi-tag registration.
        """
        # Find a known tag in current observations
        known_tag_id = None
        for obs_id in all_observations.keys():
            if obs_id in self.tag_map:
                known_tag_id = obs_id
                break

        if known_tag_id is None:
            return

        known_obs = all_observations[known_tag_id]

        # Multi-tag registration: T_known_new = T_known_camera @ T_camera_new
        R_known_camera = known_obs['R']
        t_known_camera = known_obs['tvec']

        R_new_camera = new_obs['R']
        t_new_camera = new_obs['tvec']

        R_camera_new = R_new_camera.T
        t_camera_new = (-R_camera_new @ t_new_camera).flatten()

        R_known_new = R_known_camera @ R_camera_new
        t_known_new = (R_known_camera @ t_camera_new + t_known_camera).flatten()

        # Transform to map frame
        known_tag_data = self.tag_map[known_tag_id]
        R_map_known = known_tag_data['R']
        t_map_known = known_tag_data['position']

        R_map_new = R_map_known @ R_known_new
        t_map_new = (R_map_known @ t_known_new + t_map_known).flatten()

        rot_final = Rotation.from_matrix(R_map_new)
        quat_map_new = rot_final.as_quat()

        self.tag_map[new_id] = {
            'position': t_map_new,
            'orientation': quat_map_new,
            'R': R_map_new,
            'observations': 1,
            'last_seen': self.get_clock().now(),
            'is_origin': False
        }

        self.get_logger().info(
            f"Registered tag_{new_id} at "
            f"[{t_map_new[0]:.3f}, {t_map_new[1]:.3f}, {t_map_new[2]:.3f}]"
        )

    def _publish_camera_pose_and_odom(self, observations, timestamp):
        """
        Publish camera pose in map frame and compute odometry.
        """
        # Get camera pose from first visible known tag
        camera_poses = []

        for marker_id, obs in observations.items():
            if marker_id not in self.tag_map:
                continue

            t_tag_camera = obs['tvec']
            R_tag_camera = obs['R']

            tag_data = self.tag_map[marker_id]
            t_map_tag = tag_data['position']
            R_map_tag = tag_data['R']

            R_map_camera = R_map_tag @ R_tag_camera
            t_map_camera = (R_map_tag @ t_tag_camera + t_map_tag).flatten()

            camera_poses.append((t_map_camera, R_map_camera))

        if len(camera_poses) == 0:
            return

        t_map_camera, R_map_camera = camera_poses[0]
        rot = Rotation.from_matrix(R_map_camera)
        quat_map_camera = rot.as_quat()

        # Store for odometry computation
        current_pose = (t_map_camera.copy(), quat_map_camera.copy(), timestamp)

        # Compute velocity if we have a previous pose
        velocity = np.array([0.0, 0.0, 0.0])
        angular_velocity = np.array([0.0, 0.0, 0.0])

        if self.last_camera_pose is not None:
            last_pos, last_quat, last_time = self.last_camera_pose
            
            # Compute time delta
            dt_sec = (timestamp.sec - last_time.sec) + \
                     (timestamp.nanosec - last_time.nanosec) * 1e-9
            
            if dt_sec > 0.001:  # Avoid division by very small numbers
                # Linear velocity
                velocity = (t_map_camera - last_pos) / dt_sec
                
                # Angular velocity (simplified)
                last_rot = Rotation.from_quat(last_quat)
                current_rot = Rotation.from_quat(quat_map_camera)
                delta_rot = current_rot * last_rot.inv()
                angular_velocity = delta_rot.as_rotvec() / dt_sec

        self.last_camera_pose = current_pose

        # Publish odometry (camera_link as child frame)
        self._publish_odometry(
            t_map_camera, quat_map_camera,
            velocity, angular_velocity,
            timestamp
        )

        # Publish TF: map -> camera_link
        self._publish_camera_tf(t_map_camera, quat_map_camera, timestamp)

    def _publish_odometry(self, position, orientation, velocity, angular_velocity, timestamp):
        """
        Publish odometry message for RTAB-Map.
        """
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Position
        odom_msg.pose.pose.position.x = float(position[0])
        odom_msg.pose.pose.position.y = float(position[1])
        odom_msg.pose.pose.position.z = float(position[2])

        # Orientation
        odom_msg.pose.pose.orientation.x = float(orientation[0])
        odom_msg.pose.pose.orientation.y = float(orientation[1])
        odom_msg.pose.pose.orientation.z = float(orientation[2])
        odom_msg.pose.pose.orientation.w = float(orientation[3])

        # Velocity
        odom_msg.twist.twist.linear.x = float(velocity[0])
        odom_msg.twist.twist.linear.y = float(velocity[1])
        odom_msg.twist.twist.linear.z = float(velocity[2])

        odom_msg.twist.twist.angular.x = float(angular_velocity[0])
        odom_msg.twist.twist.angular.y = float(angular_velocity[1])
        odom_msg.twist.twist.angular.z = float(angular_velocity[2])

        # Covariance (conservative estimates)
        # Position covariance (m^2)
        odom_msg.pose.covariance[0] = 0.01  # x
        odom_msg.pose.covariance[7] = 0.01  # y
        odom_msg.pose.covariance[14] = 0.01  # z
        # Orientation covariance (rad^2)
        odom_msg.pose.covariance[21] = 0.1  # roll
        odom_msg.pose.covariance[28] = 0.1  # pitch
        odom_msg.pose.covariance[35] = 0.1  # yaw

        self.odom_publisher.publish(odom_msg)

    def _publish_camera_tf(self, position, orientation, timestamp):
        """
        Publish TF transform: map -> camera_link
        """
        transform = TransformStamped()
        transform.header.stamp = timestamp
        transform.header.frame_id = "map"
        transform.child_frame_id = "camera_link"

        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])

        transform.transform.rotation.x = float(orientation[0])
        transform.transform.rotation.y = float(orientation[1])
        transform.transform.rotation.z = float(orientation[2])
        transform.transform.rotation.w = float(orientation[3])

        self.tf_broadcaster.sendTransform(transform)

    def publish_dynamic_data(self):
        """
        Publish map transforms and marker visualizations.
        Called at 30 Hz.
        """
        if self.map_frame is None or len(self.tag_map) == 0:
            return

        current_time = self.get_clock().now().to_msg()

        # Publish map -> odom (identity for now, could implement drift correction)
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = current_time
        map_to_odom.header.frame_id = "map"
        map_to_odom.child_frame_id = "odom"
        map_to_odom.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(map_to_odom)

        # Publish odom -> base_link if we have camera pose
        if self.last_camera_pose is not None:
            pos, quat, _ = self.last_camera_pose
            
            # Since camera_link is static relative to base_link,
            # we can compute base_link pose from camera pose
            odom_to_base = TransformStamped()
            odom_to_base.header.stamp = current_time
            odom_to_base.header.frame_id = "odom"
            odom_to_base.child_frame_id = "base_link"
            
            # For now, assume base_link = camera_link (will be corrected by static TF)
            odom_to_base.transform.translation.x = float(pos[0])
            odom_to_base.transform.translation.y = float(pos[1])
            odom_to_base.transform.translation.z = float(pos[2])
            
            odom_to_base.transform.rotation.x = float(quat[0])
            odom_to_base.transform.rotation.y = float(quat[1])
            odom_to_base.transform.rotation.z = float(quat[2])
            odom_to_base.transform.rotation.w = float(quat[3])
            
            self.tf_broadcaster.sendTransform(odom_to_base)

        # Publish marker array for visualization
        marker_array = MarkerArray()
        
        for marker_id, tag_data in self.tag_map.items():
            marker = Marker()
            marker.header.stamp = current_time
            marker.header.frame_id = "map"
            marker.ns = "aruco_tags"
            marker.id = int(marker_id)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = float(tag_data['position'][0])
            marker.pose.position.y = float(tag_data['position'][1])
            marker.pose.position.z = float(tag_data['position'][2])
            
            # Orientation
            marker.pose.orientation.x = float(tag_data['orientation'][0])
            marker.pose.orientation.y = float(tag_data['orientation'][1])
            marker.pose.orientation.z = float(tag_data['orientation'][2])
            marker.pose.orientation.w = float(tag_data['orientation'][3])
            
            # Scale (15cm marker)
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.01
            
            # Color (green for origin, blue for others)
            if tag_data.get('is_origin', False):
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 0.5
                marker.color.b = 1.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        self.aruco_pose_publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = TelloEnvironmentNode()

    try:
        node.get_logger().info("Tello Environment Node running...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()