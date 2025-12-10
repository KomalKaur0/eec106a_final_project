#!/usr/bin/env python3
"""
Simplified Tello Environment Node - No RTAB-Map Dependencies

Publishes:
- /tello/drone_pose (geometry_msgs/PoseStamped) - Drone pose in map frame
- /world/aruco_poses (visualization_msgs/MarkerArray) - All known AR tags
- TF tree: map -> base_link -> camera_link
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
from visualization_msgs.msg import MarkerArray, Marker

from tello_controller import tello_constants as tc


class TelloEnvironmentNode(Node):
    """
    Simplified ROS2 Node for AR tag-based localization without RTAB-Map.
    
    Creates a persistent map of AR tags and tracks drone pose relative to them.
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

        # TF broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Create subscriber
        self.camera_subscriber = self.create_subscription(
            Image,
            '/tello/camera/image_raw',
            self.camera_callback,
            10
        )

        # State
        self.bridge = CvBridge()
        self.tag_map = {}  # Persistent map of all seen tags
        self.map_frame = None  # First tag becomes map origin
        self.current_drone_pose = None  # (position, orientation, timestamp)

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

        # Publish static transforms
        self.publish_static_transforms()

        # Timer to publish visualizations
        self.viz_timer = self.create_timer(0.1, self.publish_visualizations)  # 10 Hz

        self.get_logger().info("Tello Environment Node initialized (No RTAB-Map)")
        self.get_logger().info("Publishing: drone pose, AR tag map, TF tree")

    def publish_static_transforms(self):
        """
        Publish static TF transform: base_link -> camera_link
        
        TF Tree: map -> base_link -> camera_link
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'
        
        # Camera is ~5cm forward, 0cm sideways, -2cm down from drone center
        t.transform.translation.x = 0.05
        t.transform.translation.y = 0.0
        t.transform.translation.z = -0.02
        
        # Camera points forward with slight downward tilt (~15 degrees)
        rot = Rotation.from_euler('xyz', [0, 15, 0], degrees=True)
        quat = rot.as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.static_tf_broadcaster.sendTransform(t)
        self.get_logger().info("Published static TF: base_link -> camera_link")

    def camera_callback(self, msg: Image):
        """
        Process camera frames to detect AR tags and update drone pose.
        """
        frame = self.bridge.imgmsg_to_cv2(msg)
        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if ids is None or len(ids) == 0:
            return

        current_observations = {}

        # Process each detected marker
        for i, marker_id in enumerate(ids.flatten()):
            corners_2d = corners[i][0]

            # Solve PnP to get camera pose relative to marker
            success, rvec, tvec = cv2.solvePnP(
                self.marker_points_3d,
                corners_2d,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                continue

            # Compute reprojection error for quality check
            projected_corners, _ = cv2.projectPoints(
                self.marker_points_3d, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            reprojection_error = np.mean(np.linalg.norm(
                corners_2d - projected_corners.reshape(-1, 2), axis=1
            ))

            if reprojection_error > 5.0:  # Skip low quality detections
                continue

            # Transform to get marker pose in camera frame
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

        # Update persistent map and compute drone pose
        if len(current_observations) > 0:
            self.update_tag_map(current_observations)
            self.update_drone_pose(current_observations, msg.header.stamp)

    def update_tag_map(self, observations):
        """
        Update the persistent map of AR tags.
        First seen tag becomes the map origin.
        """
        if len(observations) == 0:
            return

        # Initialize map with first tag if needed
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
        Register a new tag by triangulating its position from known tags.
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

        # Invert to get camera->new transform
        R_camera_new = R_new_camera.T
        t_camera_new = (-R_camera_new @ t_new_camera).flatten()

        # Compose: known->new = known->camera @ camera->new
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

    def update_drone_pose(self, observations, timestamp):
        """
        Compute and publish drone pose in map frame from visible tags.
        """
        # Compute camera pose from each visible known tag
        camera_poses = []

        for marker_id, obs in observations.items():
            if marker_id not in self.tag_map:
                continue

            # Camera pose in marker frame
            t_tag_camera = obs['tvec']
            R_tag_camera = obs['R']

            # Tag pose in map frame
            tag_data = self.tag_map[marker_id]
            t_map_tag = tag_data['position']
            R_map_tag = tag_data['R']

            # Compose: map->camera = map->tag @ tag->camera
            R_map_camera = R_map_tag @ R_tag_camera
            t_map_camera = (R_map_tag @ t_tag_camera + t_map_tag).flatten()

            camera_poses.append((t_map_camera, R_map_camera))

        if len(camera_poses) == 0:
            return

        # Average multiple observations if available
        if len(camera_poses) == 1:
            t_map_camera, R_map_camera = camera_poses[0]
        else:
            # Simple averaging (could use more sophisticated fusion)
            positions = np.array([p[0] for p in camera_poses])
            t_map_camera = np.mean(positions, axis=0)
            R_map_camera = camera_poses[0][1]  # Use first rotation (averaging rotations is complex)

        # Convert to quaternion
        rot = Rotation.from_matrix(R_map_camera)
        quat_map_camera = rot.as_quat()

        # Camera pose is now in map frame
        # For drone pose, we need to account for base_link->camera_link transform
        # For simplicity, we'll publish camera pose as drone pose
        # (The static TF handles the offset)
        
        self.current_drone_pose = (t_map_camera.copy(), quat_map_camera.copy(), timestamp)

        # Publish drone pose message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = 'map'
        
        pose_msg.pose.position.x = float(t_map_camera[0])
        pose_msg.pose.position.y = float(t_map_camera[1])
        pose_msg.pose.position.z = float(t_map_camera[2])
        
        pose_msg.pose.orientation.x = float(quat_map_camera[0])
        pose_msg.pose.orientation.y = float(quat_map_camera[1])
        pose_msg.pose.orientation.z = float(quat_map_camera[2])
        pose_msg.pose.orientation.w = float(quat_map_camera[3])
        
        self.drone_pose_publisher.publish(pose_msg)

        # Publish TF: map -> base_link (using camera pose for now)
        self._publish_drone_tf(t_map_camera, quat_map_camera, timestamp)

    def _publish_drone_tf(self, position, orientation, timestamp):
        """
        Publish TF transform: map -> base_link
        """
        transform = TransformStamped()
        transform.header.stamp = timestamp
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_link"

        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])

        transform.transform.rotation.x = float(orientation[0])
        transform.transform.rotation.y = float(orientation[1])
        transform.transform.rotation.z = float(orientation[2])
        transform.transform.rotation.w = float(orientation[3])

        self.tf_broadcaster.sendTransform(transform)

    def publish_visualizations(self):
        """
        Publish marker visualizations for RViz.
        """
        if self.map_frame is None or len(self.tag_map) == 0:
            return

        current_time = self.get_clock().now().to_msg()

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
            
            # Add text label
            marker_array.markers.append(marker)
            
            # Add text marker for ID
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "aruco_labels"
            text_marker.id = int(marker_id) + 1000  # Offset to avoid ID collision
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose = marker.pose
            text_marker.pose.position.z += 0.1  # Offset text above marker
            text_marker.scale.z = 0.05  # Text size
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"ID: {marker_id}"
            marker_array.markers.append(text_marker)
        
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