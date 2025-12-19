#!/usr/bin/env python3
"""
Interactive ArUco Tag Navigator for Tello Drone

This node provides an interactive system for navigating the Tello drone to user-selected
ArUco tags. It subscribes to the published ArUco poses from the environment node and uses
the TF tree to navigate to selected tags.

Workflow:
1. Shows available detected tags
2. User selects target tag ID
3. Drone takes off
4. Flies to the tag using TF-based navigation
5. Lands
6. Waits for next tag selection (loop)
"""

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tello_interfaces.srv import TelloCommand
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from tello_controller import tello_constants as tc


class TelloArucoNavigatorNode(Node):
    """
    Interactive navigation node that flies the Tello to user-selected ArUco tags.
    """

    def __init__(self):
        super().__init__('tello_aruco_navigator_node')

        # Parameters
        self.declare_parameter('min_battery', 20)
        self.declare_parameter('approach_distance_cm', 50)  # Stop this far from tag
        self.declare_parameter('max_forward_cm', 150)  # Max distance per move command

        self.min_battery = self.get_parameter('min_battery').value
        self.approach_distance = self.get_parameter('approach_distance_cm').value
        self.max_forward = self.get_parameter('max_forward_cm').value

        # Flight state tracking
        self.in_flight = False

        # Cached set of available tags (updated via MarkerArray subscription)
        self.available_tags = set()

        # TF for getting tag positions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Camera-based detection (for fresh observations during navigation)
        self.bridge = CvBridge()
        self.latest_camera_frame = None
        self.camera_matrix = tc.CAMERA_MATRIX
        self.dist_coeffs = tc.DISTORTION_COEFFS
        self.aruco_dict = tc.ARUCO_DICT
        self.aruco_params = tc.ARUCO_PARAMS
        self.marker_size_m = tc.MARKER_SIZE_M
        self.focal_length_px = tc.FOCAL_LENGTH_PX

        # Service client for flight commands
        self.command_client = self.create_client(TelloCommand, '/tello/command')

        # Subscribe to published ArUco markers from environment node
        self.aruco_sub = self.create_subscription(
            MarkerArray,
            '/world/aruco_poses',
            self.aruco_callback,
            10
        )

        # Subscribe to camera for fresh detections during navigation
        self.camera_sub = self.create_subscription(
            Image,
            '/tello/camera/image_raw',
            self.camera_callback,
            10
        )

        # Wait for service to be available
        self.get_logger().info('Waiting for /tello/command service...')
        if not self.command_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Command service not available yet. Make sure camera node is running.')

        self.get_logger().info('Tello ArUco Navigator initialized')
        self.get_logger().info(f'Min battery: {self.min_battery}%')
        self.get_logger().info(f'Approach distance: {self.approach_distance}cm')

    def aruco_callback(self, msg: MarkerArray):
        """
        Update the set of available tags based on published markers.
        Environment node publishes all known tags at 2 Hz.
        """
        if len(msg.markers) > 0:
            new_tags = {marker.id for marker in msg.markers}
            if new_tags != self.available_tags:
                self.available_tags = new_tags
                self.get_logger().debug(f'Available tags updated: {sorted(self.available_tags)}')

    def camera_callback(self, msg: Image):
        """Cache latest camera frame for navigation."""
        try:
            self.latest_camera_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f'Failed to convert camera image: {e}')

    def detect_any_tag_in_camera(self, timeout_sec: float = 5.0):
        """
        Detect ANY ArUco tag in camera frame (not target-specific).
        Returns the first/closest tag found.

        Args:
            timeout_sec: Maximum time to wait for detection

        Returns:
            (tag_id, distance_cm, angle_deg, lateral_offset_cm) or None if not detected
        """
        import time
        start_time = time.time()

        while time.time() - start_time < timeout_sec:
            # Spin to get fresh camera frame
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_camera_frame is None:
                continue

            frame = self.latest_camera_frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

            if ids is None or len(ids) == 0:
                continue

            ids = ids.flatten()

            # Return the FIRST tag found (could improve to return closest)
            marker_id = int(ids[0])
            marker_corners = corners[0][0]  # Shape (4, 2)

            # Calculate center pixel position
            center_x = np.mean(marker_corners[:, 0])
            center_y = np.mean(marker_corners[:, 1])

            # Calculate tag width in pixels
            tag_width_px = np.linalg.norm(marker_corners[0] - marker_corners[1])

            # Image center
            h, w = frame.shape[:2]
            img_center_x = w / 2.0

            # Calculate horizontal offset from image center
            dx_px = center_x - img_center_x

            # Pinhole camera model (from working mission node)
            # Distance: Z = (real_size * focal_length) / pixel_size
            tag_size_cm = self.marker_size_m * 100  # Convert to cm
            distance_cm = (tag_size_cm * self.focal_length_px) / tag_width_px

            # Angle: theta = atan2(dx_px, focal_length)
            angle_rad = math.atan2(dx_px, self.focal_length_px)
            angle_deg = math.degrees(angle_rad)

            # Lateral offset: X = Z * sin(theta)
            lateral_offset_cm = distance_cm * math.sin(angle_rad)

            self.get_logger().info(
                f'Detected ANY tag {marker_id}: '
                f'distance={distance_cm:.1f}cm, angle={angle_deg:.1f}°, '
                f'lateral_offset={lateral_offset_cm:.1f}cm'
            )

            return marker_id, distance_cm, angle_deg, lateral_offset_cm

        self.get_logger().warn(f'No tags detected after {timeout_sec}s')
        return None

    def detect_tag_in_camera(self, target_tag_id: int, timeout_sec: float = 5.0):
        """
        Detect target tag in camera frame using direct ArUco detection.
        Spins until tag is detected or timeout.

        Args:
            target_tag_id: ID of tag to detect
            timeout_sec: Maximum time to wait for detection

        Returns:
            (distance_cm, angle_deg, lateral_offset_cm) or None if not detected
        """
        import time
        start_time = time.time()

        while time.time() - start_time < timeout_sec:
            # Spin to get fresh camera frame
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_camera_frame is None:
                continue

            frame = self.latest_camera_frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

            if ids is None or len(ids) == 0:
                continue

            ids = ids.flatten()

            # Find target tag
            target_idx = None
            for i, marker_id in enumerate(ids):
                if marker_id == target_tag_id:
                    target_idx = i
                    break

            if target_idx is None:
                continue

            # Get marker corners
            marker_corners = corners[target_idx][0]  # Shape (4, 2)

            # Calculate center pixel position
            center_x = np.mean(marker_corners[:, 0])
            center_y = np.mean(marker_corners[:, 1])

            # Calculate tag width in pixels
            tag_width_px = np.linalg.norm(marker_corners[0] - marker_corners[1])

            # Image center
            h, w = frame.shape[:2]
            img_center_x = w / 2.0

            # Calculate horizontal offset from image center
            dx_px = center_x - img_center_x

            # Pinhole camera model (from working mission node)
            # Distance: Z = (real_size * focal_length) / pixel_size
            tag_size_cm = self.marker_size_m * 100  # Convert to cm
            distance_cm = (tag_size_cm * self.focal_length_px) / tag_width_px

            # Angle: theta = atan2(dx_px, focal_length)
            angle_rad = math.atan2(dx_px, self.focal_length_px)
            angle_deg = math.degrees(angle_rad)

            # Lateral offset: X = Z * sin(theta)
            lateral_offset_cm = distance_cm * math.sin(angle_rad)

            self.get_logger().info(
                f'Detected tag {target_tag_id}: '
                f'distance={distance_cm:.1f}cm, angle={angle_deg:.1f}°, '
                f'lateral_offset={lateral_offset_cm:.1f}cm'
            )

            return distance_cm, angle_deg, lateral_offset_cm

        self.get_logger().warn(f'Tag {target_tag_id} not detected after {timeout_sec}s')
        return None

    def calculate_target_angle_and_distance_from_visible_tag(
        self,
        target_tag_id: int,
        visible_tag_id: int,
        visible_tag_angle_deg: float
    ):
        """
        Calculate angle and distance to target tag using a visible tag + TF.

        This allows navigation to a tag that's NOT in the camera view, by using
        a tag that IS visible plus the known map.

        Args:
            target_tag_id: ID of tag we want to navigate to
            visible_tag_id: ID of tag currently visible in camera
            visible_tag_angle_deg: Angle to visible tag from camera center

        Returns:
            (angle_to_target_deg, distance_to_target_cm) or None if TF lookup fails
        """
        try:
            # Get both tags' positions in map frame
            visible_tag_tf = self.tf_buffer.lookup_transform(
                'map',
                f'tag_{visible_tag_id}',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            target_tag_tf = self.tf_buffer.lookup_transform(
                'map',
                f'tag_{target_tag_id}',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            # Get camera position in map frame
            camera_tf = self.tf_buffer.lookup_transform(
                'map',
                'camera_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            # Extract positions
            cam_x = camera_tf.transform.translation.x
            cam_y = camera_tf.transform.translation.y

            target_x = target_tag_tf.transform.translation.x
            target_y = target_tag_tf.transform.translation.y

            # Calculate vector from camera to target in map frame
            dx_map = target_x - cam_x
            dy_map = target_y - cam_y

            # Calculate distance
            distance_m = math.sqrt(dx_map**2 + dy_map**2)
            distance_cm = distance_m * 100

            # Calculate angle in map frame
            target_angle_map = math.atan2(dy_map, dx_map)

            # Get camera's current yaw in map frame
            q = camera_tf.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            camera_yaw = math.atan2(siny_cosp, cosy_cosp)

            # Calculate angle relative to camera (camera-frame angle)
            # This is the angle the drone needs to rotate
            angle_relative_to_camera_rad = target_angle_map - camera_yaw
            angle_relative_to_camera_deg = math.degrees(angle_relative_to_camera_rad)

            # Normalize to [-180, 180]
            while angle_relative_to_camera_deg > 180:
                angle_relative_to_camera_deg -= 360
            while angle_relative_to_camera_deg < -180:
                angle_relative_to_camera_deg += 360

            # Apply same coordinate frame correction
            # angle_relative_to_camera_deg = -angle_relative_to_camera_deg

            self.get_logger().info(
                f'Calculated target (tag {target_tag_id}) from visible tag {visible_tag_id}: '
                f'angle={angle_relative_to_camera_deg:.1f}°, distance={distance_cm:.1f}cm'
            )

            return angle_relative_to_camera_deg, distance_cm

        except Exception as e:
            self.get_logger().error(f'Failed to calculate target from visible tag: {e}')
            return None

    def get_available_tags(self):
        """
        Get available ArUco tags from cached subscription data with retry logic.

        Returns:
            Set of tag IDs that are available
        """
        import time

        # Try multiple times to ensure MarkerArray message arrives
        for attempt in range(3):
            rclpy.spin_once(self, timeout_sec=0.5)

            if len(self.available_tags) > 0:
                self.get_logger().info(f'Found {len(self.available_tags)} tags: {sorted(list(self.available_tags))}')
                return self.available_tags.copy()

            if attempt < 2:  # Don't sleep on the last attempt
                time.sleep(0.5)

        # Return empty set if no tags found after retries
        self.get_logger().warn('No tags found after retries')
        return self.available_tags.copy()

    def send_command(self, command: str, value: int = 0, stabilize_after: float = 0.0) -> bool:
        """
        Send a command to the Tello via the service.

        Args:
            command: Command name (e.g., 'takeoff', 'move_forward')
            value: Optional value (distance in cm, angle in degrees)
            stabilize_after: Seconds to wait after command for drone to stabilize (default: 0.0)

        Returns:
            True if command succeeded, False otherwise
        """
        if not self.command_client.service_is_ready():
            self.get_logger().error('Command service not available')
            return False

        request = TelloCommand.Request()
        request.command = command
        request.value = value

        future = self.command_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Command {command} succeeded: {response.message}')

                # Add stabilization delay if requested
                if stabilize_after > 0:
                    import time
                    self.get_logger().info(f'Stabilizing for {stabilize_after:.1f}s...')
                    start = time.time()
                    while time.time() - start < stabilize_after:
                        rclpy.spin_once(self, timeout_sec=0.1)
                        time.sleep(0.1)

                return True
            else:
                self.get_logger().error(f'Command {command} failed: {response.message}')
                return False
        else:
            self.get_logger().error(f'Command {command} timed out')
            return False

    def get_tag_transform(self, tag_id: int):
        """
        Look up the full transform of a tag in the map frame using TF.

        Returns:
            TransformStamped message, or None if not available
        """
        try:
            # Look up transform from map to tag
            transform = self.tf_buffer.lookup_transform(
                'map',
                f'tag_{tag_id}',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return transform

        except Exception as e:
            self.get_logger().warn(f'Could not get transform for tag {tag_id}: {e}')
            return None

    def get_tag_position(self, tag_id: int):
        """
        Look up the position of a tag in the map frame using TF.

        Returns:
            (x, y, z) in meters relative to map frame, or None if not available
        """
        transform = self.get_tag_transform(tag_id)
        if transform is None:
            return None

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        z = transform.transform.translation.z

        return (x, y, z)

    def get_camera_position(self):
        """
        Look up current camera position in map frame.

        Returns:
            (x, y, z) in meters relative to map frame, or None if not available
        """
        try:
            # Look up transform from map to camera
            transform = self.tf_buffer.lookup_transform(
                'map',
                'camera_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            return (x, y, z)

        except Exception as e:
            self.get_logger().warn(f'Could not get camera position: {e}')
            return None

    def is_localized(self) -> bool:
        """
        Check if the drone is localized (camera_link exists in TF tree).

        Returns:
            True if localized, False otherwise
        """
        try:
            self.tf_buffer.lookup_transform(
                'map',
                'camera_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return True
        except Exception:
            return False

    def ensure_localized(self, max_rotation: int = 360) -> bool:
        """
        Ensure drone is localized. If not, rotate until ANY tag is visible.

        Args:
            max_rotation: Maximum degrees to rotate while searching

        Returns:
            True if localized, False if failed after full rotation
        """
        if self.is_localized():
            self.get_logger().info('Already localized')
            return True

        print("\nNot localized - rotating to find a tag...")
        self.get_logger().warn('Lost localization, searching for tags...')

        import time
        total_rotation = 0
        rotation_increment = 30

        while total_rotation < max_rotation:
            # Try to detect ANY tag in camera
            detection = self.detect_any_tag_in_camera(timeout_sec=2.0)

            if detection is not None:
                tag_id, _, _, _ = detection
                print(f"✓ Found tag {tag_id}")
                self.get_logger().info(f'Detected tag {tag_id}, waiting for TF to update...')

                # Wait for TF to update with this observation
                time.sleep(0.5)
                for _ in range(3):
                    rclpy.spin_once(self, timeout_sec=0.2)

                # Check if we're now localized
                if self.is_localized():
                    print("Localized!")
                    self.get_logger().info('Successfully re-localized')
                    return True

            # Rotate to search for tags
            print(f"Rotating to search... ({total_rotation}° / {max_rotation}°)")
            if not self.send_command('rotate_clockwise', rotation_increment):
                return False

            total_rotation += rotation_increment
            time.sleep(0.5)

        self.get_logger().error('Failed to localize after full rotation')
        return False

    def search_and_localize(self, stay_airborne: bool = True) -> bool:
        """
        Search routine: fly up and spin to find known tags.
        Assumes drone is already in the air if stay_airborne=True.

        Args:
            stay_airborne: If True, assumes already flying. If False, takes off first.

        Returns:
            True if successfully localized, False otherwise
        """
        try:
            if not stay_airborne:
                # Step 1: Takeoff
                print("\nTaking off...")
                self.get_logger().info('Taking off for search...')
                if not self.send_command('takeoff'):
                    return False
                self.in_flight = True
                print("Takeoff complete")

                # Step 2: Fly up for better view
                print("\nFlying up 50cm for better visibility...")
                self.get_logger().info('Moving up for search...')
                if not self.send_command('move_up', 50):
                    self.emergency_land()
                    return False
                print("✓ Altitude gained")

            # Step 3: Rotate and search for tags
            print("\nSearching for ArUco tags (will rotate 360°)...")
            self.get_logger().info('Starting rotation search...')

            total_rotation = 0
            rotation_increment = 30  # Rotate 30° at a time
            max_rotation = 360

            while total_rotation < max_rotation:
                # Give time for camera to process new view
                import time
                time.sleep(1.0)  # Wait for camera processing

                # Spin to get fresh camera frames
                for _ in range(5):
                    rclpy.spin_once(self, timeout_sec=0.2)

                # Check if we can see any tags in camera (more reliable than TF check)
                detection_found = False
                if self.latest_camera_frame is not None:
                    gray = cv2.cvtColor(self.latest_camera_frame, cv2.COLOR_RGB2GRAY)
                    corners, ids, _ = cv2.aruco.detectMarkers(
                        gray, self.aruco_dict, parameters=self.aruco_params
                    )
                    if ids is not None and len(ids) > 0:
                        detection_found = True
                        detected_ids = ids.flatten().tolist()
                        self.get_logger().info(f'Camera detected tags: {detected_ids}')

                # If we detected tags, wait for TF to update and check localization
                if detection_found:
                    time.sleep(0.5)  # Brief delay for TF to update
                    for _ in range(3):
                        rclpy.spin_once(self, timeout_sec=0.2)

                    # Now check if we're localized via TF
                    if self.is_localized():
                        available_tags = self.get_available_tags()
                        if len(available_tags) > 0:
                            print(f"\nLocalized! Found {len(available_tags)} tag(s): {sorted(available_tags)}")
                            self.get_logger().info(f'Localized successfully with tags: {sorted(available_tags)}')
                            return True

                # Rotate a bit more
                print(f"Rotating... ({total_rotation}° / {max_rotation}°)")
                if not self.send_command('rotate_clockwise', rotation_increment):
                    self.emergency_land()
                    return False

                total_rotation += rotation_increment

            # Completed full rotation without localizing
            print("\nSearch complete but no known tags found")
            self.get_logger().warn('Completed 360° search without finding known tags')

            if not stay_airborne:
                # Land since we couldn't localize
                print("Landing...")
                self.send_command('land')
                self.in_flight = False

            return False

        except Exception as e:
            self.get_logger().error(f'Search failed: {e}')
            self.emergency_land()
            return False

    def show_available_tags(self):
        """Display available tags to the user."""
        # Spin briefly to update TF buffer
        rclpy.spin_once(self, timeout_sec=0.1)

        available_tags = self.get_available_tags()

        if len(available_tags) == 0:
            print("\nNo ArUco tags detected yet!")
            print("   Make sure the camera and environment nodes are running.")
            print("   Point the camera at an ArUco tag to detect it.\n")
            return False

        print("\n" + "="*50)
        print("Available ArUco Tags:")
        print("="*50)

        for tag_id in sorted(available_tags):
            pos = self.get_tag_position(tag_id)
            if pos:
                x, y, z = pos
                print(f"  Tag {tag_id}: ({x:.2f}m, {y:.2f}m, {z:.2f}m)")
            else:
                print(f"  Tag {tag_id}: (position not available)")

        print("="*50 + "\n")
        return True

    def get_user_target(self) -> int:
        """
        Prompt user to select a target tag.

        Returns:
            Selected tag ID, or -1 to quit
        """
        while True:
            # Refresh available tags
            if not self.show_available_tags():
                input("Press ENTER to refresh and try again (or Ctrl+C to quit)...")
                continue

            try:
                response = input("Enter target tag ID (or 'q' to quit): ").strip()

                if response.lower() == 'q':
                    return -1

                tag_id = int(response)

                if tag_id in self.get_available_tags():
                    return tag_id
                else:
                    print(f"Tag {tag_id} not detected. Please choose from available tags.\n")

            except ValueError:
                print("Invalid input. Please enter a number or 'q'.\n")
            except KeyboardInterrupt:
                return -1

    def navigate_to_tag(self, tag_id: int) -> bool:
        """
        Navigate to tag using hybrid approach:
        1. Use TF to verify tag exists in map and calculate initial bearing
        2. Rotate to face tag (using TF for course alignment)
        3. Use fresh camera detection for final approach (fixes stale data bug)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Get initial bearing from TF (for rotation planning)
            camera_transform = self.tf_buffer.lookup_transform(
                'map',
                'camera_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            tag_transform = self.tf_buffer.lookup_transform(
                'map',
                f'tag_{tag_id}',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # Calculate initial rotation needed (map frame)
            drone_x = camera_transform.transform.translation.x
            drone_y = camera_transform.transform.translation.y

            q = camera_transform.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            drone_yaw_rad = math.atan2(siny_cosp, cosy_cosp)
            drone_yaw_deg = math.degrees(drone_yaw_rad)

            tag_x = tag_transform.transform.translation.x
            tag_y = tag_transform.transform.translation.y

            dx_map = tag_x - drone_x
            dy_map = tag_y - drone_y

            target_angle_rad = math.atan2(dy_map, dx_map)
            target_angle_deg = math.degrees(target_angle_rad)

            # DEBUG: Log all coordinate frame data
            self.get_logger().info('='*60)
            self.get_logger().info('COORDINATE FRAME DEBUG')
            self.get_logger().info(f'Drone position (map frame): ({drone_x:.3f}, {drone_y:.3f})')
            self.get_logger().info(f'Target tag position (map frame): ({tag_x:.3f}, {tag_y:.3f})')
            self.get_logger().info(f'Vector to target: dx={dx_map:.3f}, dy={dy_map:.3f}')
            self.get_logger().info(f'Drone quaternion: x={q.x:.3f}, y={q.y:.3f}, z={q.z:.3f}, w={q.w:.3f}')
            self.get_logger().info(f'Drone yaw (from quat): {drone_yaw_deg:.1f}°')
            self.get_logger().info(f'Target angle (atan2): {target_angle_deg:.1f}°')
            self.get_logger().info('='*60)

            rotation_needed_deg = target_angle_deg - drone_yaw_deg

            # Normalize to [-180, 180]
            while rotation_needed_deg > 180:
                rotation_needed_deg -= 360
            while rotation_needed_deg < -180:
                rotation_needed_deg += 360

            # Coordinate frame correction: Negate rotation for correct direction
            # rotation_needed_deg = -rotation_needed_deg

            self.get_logger().info(
                f'Initial bearing: drone_yaw={drone_yaw_deg:.1f}°, '
                f'target_angle={target_angle_deg:.1f}°, rotation={rotation_needed_deg:.1f}° (negated)'
            )

            # Step 2: Rotate to face tag (coarse alignment)
            if abs(rotation_needed_deg) > 5:
                print(f"\nRotating {rotation_needed_deg:.1f}° to face tag...")

                if rotation_needed_deg > 0:
                    cmd = 'rotate_clockwise'
                else:
                    cmd = 'rotate_counter_clockwise'

                if not self.send_command(cmd, int(abs(rotation_needed_deg))):
                    self.emergency_land()
                    return False
                print("✓ Rotation complete")

            # Step 3: Detect ANY tag in camera (KEY FIX - not just target!)
            print("\nDetecting ANY tag in camera for refinement...")
            any_tag_detection = self.detect_any_tag_in_camera(timeout_sec=5.0)

            if any_tag_detection is not None:
                visible_tag_id, visible_dist, visible_angle, _ = any_tag_detection

                if visible_tag_id == tag_id:
                    # Lucky! We can see the target tag directly
                    print(f"Can see target tag {tag_id} directly")
                    distance_cm = visible_dist
                    angle_deg = visible_angle
                else:
                    # Can see a different tag - use it to calculate target position
                    print(f"Using visible tag {visible_tag_id} to navigate to tag {tag_id}")

                    result = self.calculate_target_angle_and_distance_from_visible_tag(
                        tag_id, visible_tag_id, visible_angle
                    )

                    if result is None:
                        self.get_logger().error('Failed to calculate target from visible tag')
                        self.emergency_land()
                        return False

                    angle_deg, distance_cm = result

                # Step 4: Fine rotation if needed
                if abs(angle_deg) > 5:
                    print(f"\nFine-tuning rotation ({angle_deg:.1f}°)...")

                    if angle_deg > 0:
                        cmd = 'rotate_clockwise'
                    else:
                        cmd = 'rotate_counter_clockwise'

                    if not self.send_command(cmd, int(abs(angle_deg))):
                        self.emergency_land()
                        return False
                    print("✓ Fine rotation complete")

            else:
                # No tags visible - fall back to TF-only navigation
                print("\nNo tags visible in camera - using TF-only navigation")

                try:
                    # Get distance from TF
                    camera_tf = self.tf_buffer.lookup_transform(
                        'map', 'camera_link', rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.5)
                    )
                    target_tf = self.tf_buffer.lookup_transform(
                        'map', f'tag_{tag_id}', rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.5)
                    )

                    cam_x = camera_tf.transform.translation.x
                    cam_y = camera_tf.transform.translation.y
                    tgt_x = target_tf.transform.translation.x
                    tgt_y = target_tf.transform.translation.y

                    dx = tgt_x - cam_x
                    dy = tgt_y - cam_y
                    distance_cm = math.sqrt(dx**2 + dy**2) * 100

                except Exception as e:
                    self.get_logger().error(f'TF-only navigation failed: {e}')
                    self.emergency_land()
                    return False

            # Step 5: Move forward
            # Apply safety buffer
            forward_distance = max(0, distance_cm - self.approach_distance)
            forward_distance = min(forward_distance, self.max_forward)

            if forward_distance > 20:
                print(f"\nMoving forward {forward_distance:.0f}cm...")
                if not self.send_command('move_forward', int(forward_distance)):
                    self.emergency_land()
                    return False
                print("✓ Forward movement complete")
            else:
                print("\n✓ Already at target distance")

            # Step 6: Land
            print("\nLanding...")
            if not self.send_command('land'):
                self.emergency_land()
                return False

            self.in_flight = False
            print("Landed successfully!")
            return True

        except Exception as e:
            self.get_logger().error(f'Navigation failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.emergency_land()
            return False

    def emergency_land(self):
        """Attempt emergency landing."""
        if self.in_flight:
            self.get_logger().warn('Attempting emergency landing...')
            if self.send_command('land'):
                self.in_flight = False
            else:
                self.get_logger().error('Emergency landing failed')

    def run_interactive_loop(self):
        """Main interactive loop for tag navigation."""
        print("\n" + "="*50)
        print("Tello ArUco Navigator")
        print("="*50)
        print("This tool allows you to navigate the Tello drone to ArUco tags.")
        print("Make sure the camera and environment nodes are running!")
        print("="*50 + "\n")

        # Wait for TF buffer to populate (increased from 3 to 5 iterations)
        print("Waiting for TF tree and tag data to populate...")
        self.get_logger().info('Waiting for TF tree to populate...')

        # Spin for a few seconds to let TF messages and MarkerArray arrive
        import time
        for i in range(5):
            rclpy.spin_once(self, timeout_sec=1.0)
            time.sleep(0.5)
            # Log progress
            if i == 2:
                tags_found = len(self.available_tags)
                if tags_found > 0:
                    self.get_logger().info(f'Found {tags_found} tags so far...')

        self.get_logger().info(f'TF buffer ready. Total tags: {len(self.available_tags)}')

        while True:
            try:
                # Get target tag from user (while on ground)
                print("\n" + "="*50)
                print("SELECT TARGET TAG")
                print("="*50)
                tag_id = self.get_user_target()

                if tag_id == -1:
                    self.get_logger().info('User quit')
                    break

                # Take off
                print("\n" + "="*50)
                print("TAKEOFF")
                print("="*50)
                print(f"Taking off to navigate to tag {tag_id}...")

                if not self.send_command('takeoff'):
                    print("Takeoff failed")
                    continue

                self.in_flight = True
                print("Airborne\n")

                # CRITICAL: Wait for IMU to stabilize after takeoff
                # The Tello's IMU needs 3-4 seconds to stabilize before accepting movement commands
                # Without this delay, we get "error No valid imu" responses
                print("Waiting for IMU to stabilize...")
                import time
                stabilization_time = 3.5  # seconds
                start_wait = time.time()
                while time.time() - start_wait < stabilization_time:
                    # Keep ROS spinning while we wait
                    rclpy.spin_once(self, timeout_sec=0.1)
                    time.sleep(0.1)
                print("IMU stabilized\n")

                # Fly up and rotate until we see a tag (to localize position)
                print("Flying up 30cm for better view...")
                if not self.send_command('move_up', 30):
                    self.emergency_land()
                    continue
                print("Altitude gained\n")

                # Rotate until we see a tag and become localized
                print("Rotating to find ArUco tags for localization...")

                # Spin TF buffer to get latest transforms
                for _ in range(5):
                    rclpy.spin_once(self, timeout_sec=0.2)

                max_search_rotation = 360
                rotation_increment = 30
                total_rotation = 0

                while total_rotation < max_search_rotation and not self.is_localized():
                    # Check if we can see any tags now
                    available_tags = self.get_available_tags()
                    if len(available_tags) > 0:
                        print(f"Found {len(available_tags)} tag(s): {sorted(available_tags)}")
                        self.get_logger().info(f'Localized with tags: {sorted(available_tags)}')
                        break

                    # Rotate a bit more to search
                    print(f"  Rotating to search... ({total_rotation}° / {max_search_rotation}°)")
                    if not self.send_command('rotate_clockwise', rotation_increment):
                        self.emergency_land()
                        continue

                    total_rotation += rotation_increment

                    # Give TF buffer time to update
                    import time
                    time.sleep(0.5)
                    for _ in range(3):
                        rclpy.spin_once(self, timeout_sec=0.2)

                # Final check if we're localized
                if not self.is_localized():
                    print("\nCould not find any tags after full rotation")
                    self.emergency_land()

                    response = input("\nTry again? (y/n): ").strip().lower()
                    if response != 'y':
                        break
                    continue

                print("Localized!\n")

                # Navigate to selected tag
                print("\n" + "="*50)
                print(f"NAVIGATING TO TAG {tag_id}")
                print("="*50)

                success = self.navigate_to_tag(tag_id)

                if success:
                    print("\nNavigation complete and landed!")
                    # Drone is now on the ground
                    self.in_flight = False

                    # Ask if user wants to fly to another tag
                    response = input("\nNavigate to another tag? (y/n): ").strip().lower()
                    if response == 'y':
                        continue  # Loop back to tag selection
                    else:
                        self.get_logger().info('User chose to quit')
                        break  # Exit program
                else:
                    # Navigation failed
                    print("\nNavigation failed")
                    # Already emergency landed if needed
                    continue

            except KeyboardInterrupt:
                self.get_logger().info('Interrupted by user')
                self.emergency_land()
                break

    def cleanup(self):
        """Clean up resources."""
        if self.in_flight:
            self.emergency_land()
        self.get_logger().info('Navigator shutdown')


def main(args=None):
    rclpy.init(args=args)

    node = TelloArucoNavigatorNode()

    try:
        # Wait for service to be ready
        node.get_logger().info('Waiting for command service...')
        if not node.command_client.wait_for_service(timeout_sec=10.0):
            node.get_logger().error('Command service not available. Make sure camera node is running.')
            return

        # Start interactive loop
        node.run_interactive_loop()

    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt')

    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
