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
from visualization_msgs.msg import MarkerArray
from tf2_ros import Buffer, TransformListener
from tello_interfaces.srv import TelloCommand
import math


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

        # ArUco tag tracking
        self.available_tags = set()  # Set of detected tag IDs

        # TF for getting tag positions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Service client for flight commands
        self.command_client = self.create_client(TelloCommand, '/tello/command')

        # Wait for service to be available
        self.get_logger().info('Waiting for /tello/command service...')
        if not self.command_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Command service not available yet. Make sure camera node is running.')

        # Subscribe to published ArUco markers
        self.aruco_sub = self.create_subscription(
            MarkerArray,
            '/world/aruco_poses',
            self.aruco_callback,
            10
        )

        self.get_logger().info('Tello ArUco Navigator initialized')
        self.get_logger().info(f'Min battery: {self.min_battery}%')
        self.get_logger().info(f'Approach distance: {self.approach_distance}cm')

    def aruco_callback(self, msg: MarkerArray):
        """
        Update the set of available tags based on published markers.
        """
        if len(msg.markers) > 0:
            new_tags = {marker.id for marker in msg.markers}
            if new_tags != self.available_tags:
                self.available_tags = new_tags
                # Only log changes to avoid spam
                if len(self.available_tags) > 0:
                    self.get_logger().info(
                        f'Available tags: {sorted(self.available_tags)}',
                        throttle_duration_sec=2.0
                    )

    def send_command(self, command: str, value: int = 0) -> bool:
        """
        Send a command to the Tello via the service.

        Args:
            command: Command name (e.g., 'takeoff', 'move_forward')
            value: Optional value (distance in cm, angle in degrees)

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

    def is_tag_horizontal(self, tag_id: int, threshold_deg: float = 30.0) -> bool:
        """
        Check if a tag is horizontal (lying flat, e.g., on the ground).

        A tag is considered horizontal if its Z-axis (normal vector) points
        roughly upward in the world frame.

        Args:
            tag_id: The tag ID to check
            threshold_deg: Maximum angle from vertical (default 30°)

        Returns:
            True if tag is horizontal, False otherwise
        """
        transform = self.get_tag_transform(tag_id)
        if transform is None:
            return False

        # Get quaternion
        q = transform.transform.rotation
        qx, qy = q.x, q.y

        # Convert quaternion to rotation matrix (we only need the Z component)
        # Z-axis of tag frame in world coordinates (3rd row, 3rd column of rotation matrix)
        tag_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)

        # World Z-axis is (0, 0, 1)
        # Dot product tells us alignment: close to 1 = pointing up, close to -1 = pointing down
        dot_product = tag_z_z  # Since world Z is (0,0,1)

        # Calculate angle from vertical
        angle_rad = math.acos(max(-1.0, min(1.0, abs(dot_product))))
        angle_deg = math.degrees(angle_rad)

        is_horizontal = angle_deg < threshold_deg

        self.get_logger().info(
            f'Tag {tag_id} orientation: {angle_deg:.1f}° from vertical '
            f'{"HORIZONTAL" if is_horizontal else "VERTICAL"}'
        )

        return is_horizontal

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

    def show_available_tags(self):
        """Display available tags to the user."""
        if len(self.available_tags) == 0:
            print("\nNo ArUco tags detected yet!")
            print("   Make sure the camera and environment nodes are running.")
            print("   Point the camera at an ArUco tag to detect it.\n")
            return False

        print("\n" + "="*50)
        print("Available ArUco Tags:")
        print("="*50)

        for tag_id in sorted(self.available_tags):
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

                if tag_id in self.available_tags:
                    return tag_id
                else:
                    print(f"Tag {tag_id} not detected. Please choose from available tags.\n")

            except ValueError:
                print("Invalid input. Please enter a number or 'q'.\n")
            except KeyboardInterrupt:
                return -1

    def navigate_to_tag(self, tag_id: int) -> bool:
        """
        Navigate the drone to the specified tag.

        Strategy for VERTICAL tags (on walls):
        1. Rotate to face tag
        2. Move forward (stopping before tag)
        3. Land

        Strategy for HORIZONTAL tags (on ground):
        1. Move above the tag (XY positioning)
        2. Descend and land directly on tag

        Returns:
            True if successful, False otherwise
        """
        # Check if tag is horizontal
        is_horizontal = self.is_tag_horizontal(tag_id)

        # Get tag position
        tag_pos = self.get_tag_position(tag_id)
        if tag_pos is None:
            self.get_logger().error(f'Cannot navigate: Tag {tag_id} position not available')
            return False

        tag_x, tag_y, tag_z = tag_pos
        self.get_logger().info(f'Target tag {tag_id} at ({tag_x:.2f}, {tag_y:.2f}, {tag_z:.2f})m')

        # Get current camera position
        cam_pos = self.get_camera_position()
        if cam_pos is None:
            self.get_logger().warn('Camera position not available, using (0,0,0)')
            cam_x, cam_y, cam_z = 0.0, 0.0, 0.0
        else:
            cam_x, cam_y, cam_z = cam_pos
            self.get_logger().info(f'Camera at ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f})m')

        # Calculate vector to tag
        dx = tag_x - cam_x
        dy = tag_y - cam_y
        dz = tag_z - cam_z

        if is_horizontal:
            # Horizontal tag: land on top
            return self._navigate_to_horizontal_tag(dx, dy, dz)
        else:
            # Vertical tag: approach from front
            return self._navigate_to_vertical_tag(dx, dy)

    def _navigate_to_vertical_tag(self, dx: float, dy: float) -> bool:
        """
        Navigate to a vertical tag (on a wall).
        Approach from the front and stop before hitting it.
        """
        # Calculate horizontal distance and angle
        horizontal_dist = math.sqrt(dx**2 + dy**2)
        angle_to_tag = math.atan2(dx, dy)  # Note: atan2(x, y) because drone forward is +Y
        angle_deg = math.degrees(angle_to_tag)

        # Convert to cm for Tello commands
        horizontal_dist_cm = horizontal_dist * 100

        self.get_logger().info(
            f'Vertical tag navigation: rotate {angle_deg:.1f}°, '
            f'forward {horizontal_dist_cm:.1f}cm'
        )

        # Confirm with user
        print(f"\n📋 Navigation Plan (VERTICAL TAG):")
        print(f"   Rotate: {angle_deg:.1f}°")
        print(f"   Move forward: {horizontal_dist_cm:.1f}cm")
        print(f"   (Will stop {self.approach_distance}cm before tag)")

        response = input("\nExecute? (y/n): ").strip().lower()
        if response != 'y':
            self.get_logger().info('Navigation cancelled by user')
            return False

        try:
            # Step 1: Takeoff
            self.get_logger().info('Taking off...')
            if not self.send_command('takeoff'):
                return False
            self.in_flight = True
            self.get_logger().info('✓ Takeoff complete')

            # Step 2: Rotate to face tag
            if abs(angle_deg) > 5:  # Only rotate if angle is significant
                if angle_deg > 0:
                    self.get_logger().info(f'Rotating clockwise {angle_deg:.1f}°')
                    if not self.send_command('rotate_clockwise', int(abs(angle_deg))):
                        self.emergency_land()
                        return False
                else:
                    self.get_logger().info(f'Rotating counter-clockwise {abs(angle_deg):.1f}°')
                    if not self.send_command('rotate_counter_clockwise', int(abs(angle_deg))):
                        self.emergency_land()
                        return False
                self.get_logger().info('✓ Rotation complete')

            # Step 3: Move forward toward tag (leaving approach distance buffer)
            forward_dist = max(0, horizontal_dist_cm - self.approach_distance)
            forward_dist = min(forward_dist, self.max_forward)  # Safety limit

            if forward_dist > 20:  # Tello minimum is 20cm
                self.get_logger().info(f'Moving forward {forward_dist:.1f}cm')
                if not self.send_command('move_forward', int(forward_dist)):
                    self.emergency_land()
                    return False
                self.get_logger().info('✓ Movement complete')
            else:
                self.get_logger().info('Already close to tag, skipping forward movement')

            # Step 4: Land
            self.get_logger().info('Landing...')
            if not self.send_command('land'):
                return False
            self.in_flight = False
            self.get_logger().info('✓ Landed successfully')

            print("\n" + "="*50)
            print("MISSION COMPLETE!")
            print("="*50 + "\n")

            return True

        except Exception as e:
            self.get_logger().error(f'Navigation failed: {e}')
            self.emergency_land()
            return False

    def _navigate_to_horizontal_tag(self, dx: float, dy: float, dz: float) -> bool:
        """
        Navigate to a horizontal tag (on the ground).
        Position above it and land directly on top.
        """
        # Calculate horizontal distance and angle
        horizontal_dist = math.sqrt(dx**2 + dy**2)
        angle_to_tag = math.atan2(dx, dy)
        angle_deg = math.degrees(angle_to_tag)

        # Convert to cm for Tello commands
        horizontal_dist_cm = horizontal_dist * 100
        vertical_dist_cm = dz * 100

        self.get_logger().info(
            f'Horizontal tag navigation: rotate {angle_deg:.1f}°, '
            f'forward {horizontal_dist_cm:.1f}cm, descend {vertical_dist_cm:.1f}cm'
        )

        # Confirm with user
        print(f"\n📋 Navigation Plan (HORIZONTAL TAG - WILL LAND ON TOP):")
        print(f"   Rotate: {angle_deg:.1f}°")
        print(f"   Move forward: {horizontal_dist_cm:.1f}cm (to position above tag)")
        print(f"   Land directly on tag")

        response = input("\nExecute? (y/n): ").strip().lower()
        if response != 'y':
            self.get_logger().info('Navigation cancelled by user')
            return False

        try:
            # Step 1: Takeoff
            self.get_logger().info('Taking off...')
            if not self.send_command('takeoff'):
                return False
            self.in_flight = True
            self.get_logger().info('✓ Takeoff complete')

            # Step 2: Rotate to face tag
            if abs(angle_deg) > 5:
                if angle_deg > 0:
                    self.get_logger().info(f'Rotating clockwise {angle_deg:.1f}°')
                    if not self.send_command('rotate_clockwise', int(abs(angle_deg))):
                        self.emergency_land()
                        return False
                else:
                    self.get_logger().info(f'Rotating counter-clockwise {abs(angle_deg):.1f}°')
                    if not self.send_command('rotate_counter_clockwise', int(abs(angle_deg))):
                        self.emergency_land()
                        return False
                self.get_logger().info('✓ Rotation complete')

            # Step 3: Move forward to position above tag (no buffer needed)
            forward_dist = min(horizontal_dist_cm, self.max_forward)

            if forward_dist > 20:
                self.get_logger().info(f'Moving forward {forward_dist:.1f}cm to position above tag')
                if not self.send_command('move_forward', int(forward_dist)):
                    self.emergency_land()
                    return False
                self.get_logger().info('✓ Positioned above tag')
            else:
                self.get_logger().info('Already above tag')

            # Step 4: Land directly on tag
            self.get_logger().info('Landing on tag...')
            if not self.send_command('land'):
                return False
            self.in_flight = False
            self.get_logger().info('✓ Landed on tag!')

            print("\n" + "="*50)
            print("✅ LANDED ON TAG!")
            print("="*50 + "\n")

            return True

        except Exception as e:
            self.get_logger().error(f'Navigation failed: {e}')
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
        print("🚁 Tello ArUco Navigator")
        print("="*50)
        print("This tool allows you to navigate the Tello drone to ArUco tags.")
        print("Make sure the camera and environment nodes are running!")
        print("="*50 + "\n")

        # Tag detection happens in the background via subscription
        self.get_logger().info('Monitoring for tag detection...')

        while True:
            try:
                # Get target tag from user
                tag_id = self.get_user_target()

                if tag_id == -1:
                    self.get_logger().info('User quit')
                    break

                # Navigate to tag
                self.navigate_to_tag(tag_id)

                # Ask if user wants to continue
                print()
                response = input("Navigate to another tag? (y/n): ").strip().lower()
                if response != 'y':
                    self.get_logger().info('Exiting navigation loop')
                    break

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
