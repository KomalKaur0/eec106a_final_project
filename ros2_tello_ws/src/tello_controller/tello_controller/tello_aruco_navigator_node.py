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
from tf2_ros import Buffer, TransformListener
from tello_interfaces.srv import TelloCommand
import math
import yaml


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

        # TF for getting tag positions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Service client for flight commands
        self.command_client = self.create_client(TelloCommand, '/tello/command')

        # Wait for service to be available
        self.get_logger().info('Waiting for /tello/command service...')
        if not self.command_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Command service not available yet. Make sure camera node is running.')

        self.get_logger().info('Tello ArUco Navigator initialized')
        self.get_logger().info(f'Min battery: {self.min_battery}%')
        self.get_logger().info(f'Approach distance: {self.approach_distance}cm')

    def get_available_tags(self):
        """
        Query TF tree for all available ArUco tags.

        Returns:
            Set of tag IDs that are available in the TF tree
        """
        available_tags = set()

        # Get all frames from TF tree
        try:
            all_frames = self.tf_buffer.all_frames_as_yaml()

            # Parse for tag frames (format: "tag_N")
            if all_frames:
                frames_dict = yaml.safe_load(all_frames)

                if frames_dict:
                    all_frame_names = list(frames_dict.keys())
                    self.get_logger().debug(f'All frames in TF tree: {all_frame_names}')

                    for frame_name in all_frame_names:
                        if frame_name.startswith('tag_'):
                            try:
                                # Extract tag ID from frame name (e.g., "tag_5" -> 5)
                                tag_id_str = frame_name[4:]  # Remove "tag_" prefix
                                tag_id = int(tag_id_str)
                                available_tags.add(tag_id)
                                self.get_logger().debug(f'Found tag {tag_id} from frame "{frame_name}"')
                            except (ValueError, IndexError) as e:
                                self.get_logger().warn(f'Failed to parse tag ID from frame "{frame_name}": {e}')
                else:
                    self.get_logger().warn('TF tree YAML parsed to None')
            else:
                self.get_logger().warn('TF tree YAML is empty')

        except Exception as e:
            self.get_logger().error(f'Failed to query TF tree: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

        self.get_logger().info(f'Total tags found: {len(available_tags)} - {sorted(list(available_tags))}')
        return available_tags

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
                print("\n📍 Taking off...")
                self.get_logger().info('Taking off for search...')
                if not self.send_command('takeoff'):
                    return False
                self.in_flight = True
                print("✓ Takeoff complete")

                # Step 2: Fly up for better view
                print("\n📍 Flying up 50cm for better visibility...")
                self.get_logger().info('Moving up for search...')
                if not self.send_command('move_up', 50):
                    self.emergency_land()
                    return False
                print("✓ Altitude gained")

            # Step 3: Rotate and search for tags
            print("\n📍 Searching for ArUco tags (will rotate 360°)...")
            self.get_logger().info('Starting rotation search...')

            total_rotation = 0
            rotation_increment = 30  # Rotate 30° at a time
            max_rotation = 360

            while total_rotation < max_rotation:
                # Spin multiple times to ensure TF buffer updates
                for _ in range(3):
                    rclpy.spin_once(self, timeout_sec=0.2)

                # Check if we're localized
                if self.is_localized():
                    available_tags = self.get_available_tags()
                    if len(available_tags) > 0:
                        print(f"\n✓ Localized! Found {len(available_tags)} tag(s): {sorted(available_tags)}")
                        self.get_logger().info(f'Localized successfully with tags: {sorted(available_tags)}')
                        return True

                # Rotate a bit more
                print(f"  Rotating... ({total_rotation}° / {max_rotation}°)")
                if not self.send_command('rotate_clockwise', rotation_increment):
                    self.emergency_land()
                    return False

                total_rotation += rotation_increment

                # Give time for camera to process new view
                import time
                time.sleep(1.0)  # Longer wait for camera processing

            # Completed full rotation without localizing
            print("\n❌ Search complete but no known tags found")
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

        # Get transform from camera to tag (relative position)
        try:
            transform = self.tf_buffer.lookup_transform(
                'camera_link',  # From camera
                f'tag_{tag_id}',  # To tag
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # Extract relative position
            dx = transform.transform.translation.x
            dy = transform.transform.translation.y
            dz = transform.transform.translation.z

            self.get_logger().info(
                f'Tag {tag_id} relative to camera: '
                f'dx={dx:.2f}m, dy={dy:.2f}m, dz={dz:.2f}m'
            )

        except Exception as e:
            self.get_logger().error(f'Cannot get transform from camera to tag {tag_id}: {e}')
            return False

        if is_horizontal:
            # Horizontal tag: land on top
            return self._navigate_to_horizontal_tag(dx, dy, dz)
        else:
            # Vertical tag: approach from front
            return self._navigate_to_vertical_tag(dx, dy, dz)

    def _navigate_to_vertical_tag(self, dx: float, dy: float, dz: float) -> bool:
        """
        Navigate to a vertical tag (on a wall).
        Adjusts height, then approaches from the front.

        Args:
            dx, dy, dz: Relative position of tag from camera (meters)
        """
        # Calculate horizontal distance and angle
        horizontal_dist = math.sqrt(dx**2 + dy**2)
        angle_to_tag = math.atan2(dx, dy)  # Note: atan2(x, y) because drone forward is +Y
        angle_deg = math.degrees(angle_to_tag)

        # Convert to cm for Tello commands
        horizontal_dist_cm = horizontal_dist * 100
        vertical_dist_cm = dz * 100  # Positive = tag is above, negative = tag is below

        self.get_logger().info(
            f'Vertical tag navigation: height_adjust={vertical_dist_cm:.1f}cm, '
            f'rotate={angle_deg:.1f}°, forward={horizontal_dist_cm:.1f}cm'
        )

        # Confirm with user
        print(f"\n📋 Navigation Plan (VERTICAL TAG):")
        if abs(vertical_dist_cm) > 20:
            if vertical_dist_cm > 0:
                print(f"   Move up: {abs(vertical_dist_cm):.1f}cm (to match tag height)")
            else:
                print(f"   Move down: {abs(vertical_dist_cm):.1f}cm (to match tag height)")
        print(f"   Rotate: {angle_deg:.1f}°")
        print(f"   Move forward: {horizontal_dist_cm:.1f}cm")
        print(f"   (Will stop {self.approach_distance}cm before tag)")

        response = input("\nExecute? (y/n): ").strip().lower()
        if response != 'y':
            self.get_logger().info('Navigation cancelled by user')
            return False

        try:
            # Step 1: Adjust height to match tag
            if abs(vertical_dist_cm) > 20:  # Only adjust if significant difference
                if vertical_dist_cm > 0:
                    # Tag is above, fly up
                    move_up_cm = min(abs(vertical_dist_cm), 500)  # Safety limit
                    print(f"\n📍 Adjusting altitude (up {move_up_cm:.0f}cm)...")
                    if not self.send_command('move_up', int(move_up_cm)):
                        self.emergency_land()
                        return False
                else:
                    # Tag is below, fly down
                    move_down_cm = min(abs(vertical_dist_cm), 500)  # Safety limit
                    print(f"\n📍 Adjusting altitude (down {move_down_cm:.0f}cm)...")
                    if not self.send_command('move_down', int(move_down_cm)):
                        self.emergency_land()
                        return False
                print("✓ Height adjusted")

            # Step 2: Rotate to face tag
            if abs(angle_deg) > 5:  # Only rotate if angle is significant
                print(f"\n📍 Rotating to face tag ({abs(angle_deg):.1f}°)...")
                if angle_deg > 0:
                    if not self.send_command('rotate_clockwise', int(abs(angle_deg))):
                        self.emergency_land()
                        return False
                else:
                    if not self.send_command('rotate_counter_clockwise', int(abs(angle_deg))):
                        self.emergency_land()
                        return False
                print("✓ Rotation complete")

            # Step 3: Move forward toward tag (leaving approach distance buffer)
            forward_dist = max(0, horizontal_dist_cm - self.approach_distance)
            forward_dist = min(forward_dist, self.max_forward)  # Safety limit

            if forward_dist > 20:  # Tello minimum is 20cm
                print(f"\n📍 Moving forward ({forward_dist:.0f}cm)...")
                if not self.send_command('move_forward', int(forward_dist)):
                    self.emergency_land()
                    return False
                print("✓ Movement complete")
            else:
                self.get_logger().info('Already close to tag, skipping forward movement')

            # Navigation complete - stay in the air
            print("\n✓ Navigation complete (hovering)")

            return True

        except Exception as e:
            self.get_logger().error(f'Navigation failed: {e}')
            self.emergency_land()
            return False

    def _navigate_to_horizontal_tag(self, dx: float, dy: float, dz: float) -> bool:
        """
        Navigate to a horizontal tag (on the ground or a surface).
        Position above it and land directly on top.

        Args:
            dx, dy, dz: Relative position of tag from camera (meters)
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
            f'forward {horizontal_dist_cm:.1f}cm, vertical_adjust {vertical_dist_cm:.1f}cm'
        )

        # Confirm with user
        print(f"\n📋 Navigation Plan (HORIZONTAL TAG - WILL LAND ON TOP):")
        if abs(vertical_dist_cm) > 20:
            if vertical_dist_cm > 0:
                print(f"   Move up: {abs(vertical_dist_cm):.1f}cm")
            else:
                print(f"   Move down: {abs(vertical_dist_cm):.1f}cm")
        print(f"   Rotate: {angle_deg:.1f}°")
        print(f"   Move forward: {horizontal_dist_cm:.1f}cm (to position above tag)")
        print(f"   Land directly on tag")

        response = input("\nExecute? (y/n): ").strip().lower()
        if response != 'y':
            self.get_logger().info('Navigation cancelled by user')
            return False

        try:
            # Step 1: Adjust height if needed (to be level with or slightly above tag)
            if abs(vertical_dist_cm) > 20:
                if vertical_dist_cm > 0:
                    move_up_cm = min(abs(vertical_dist_cm), 500)
                    print(f"\n📍 Adjusting altitude (up {move_up_cm:.0f}cm)...")
                    if not self.send_command('move_up', int(move_up_cm)):
                        self.emergency_land()
                        return False
                else:
                    move_down_cm = min(abs(vertical_dist_cm), 500)
                    print(f"\n📍 Adjusting altitude (down {move_down_cm:.0f}cm)...")
                    if not self.send_command('move_down', int(move_down_cm)):
                        self.emergency_land()
                        return False
                print("✓ Height adjusted")

            # Step 2: Rotate to face tag
            if abs(angle_deg) > 5:
                print(f"\n📍 Rotating to face tag ({abs(angle_deg):.1f}°)...")
                if angle_deg > 0:
                    if not self.send_command('rotate_clockwise', int(abs(angle_deg))):
                        self.emergency_land()
                        return False
                else:
                    if not self.send_command('rotate_counter_clockwise', int(abs(angle_deg))):
                        self.emergency_land()
                        return False
                print("✓ Rotation complete")

            # Step 3: Move forward to position above tag (no buffer needed)
            forward_dist = min(horizontal_dist_cm, self.max_forward)

            if forward_dist > 20:
                print(f"\n📍 Moving forward ({forward_dist:.0f}cm) to position above tag...")
                if not self.send_command('move_forward', int(forward_dist)):
                    self.emergency_land()
                    return False
                print("✓ Positioned above tag")
            else:
                self.get_logger().info('Already above tag')

            # Step 4: Position above tag (stay hovering)
            print("\n✓ Positioned above tag (hovering)")

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

        # Wait for TF buffer to populate
        print("Waiting for TF tree to populate...")
        self.get_logger().info('Waiting for TF tree to populate...')

        # Spin for a few seconds to let TF messages arrive
        import time
        for _ in range(3):
            rclpy.spin_once(self, timeout_sec=1.0)
            time.sleep(0.5)

        self.get_logger().info('TF buffer ready')

        while True:
            try:
                # Get target tag from user (while on ground)
                print("\n" + "="*50)
                print("📋 SELECT TARGET TAG")
                print("="*50)
                tag_id = self.get_user_target()

                if tag_id == -1:
                    self.get_logger().info('User quit')
                    break

                # Take off
                print("\n" + "="*50)
                print("🚁 TAKEOFF")
                print("="*50)
                print(f"Taking off to navigate to tag {tag_id}...")

                if not self.send_command('takeoff'):
                    print("❌ Takeoff failed")
                    continue

                self.in_flight = True
                print("✓ Airborne\n")

                # Fly up for better visibility
                print("Flying up 50cm for better view...")
                if not self.send_command('move_up', 50):
                    self.emergency_land()
                    continue
                print("✓ Altitude gained\n")

                # Check if we're localized, if not, search
                if not self.is_localized():
                    print("⚠️  Not localized yet, searching for known tags...")

                    if not self.search_and_localize(stay_airborne=True):
                        print("\n❌ Failed to localize")
                        self.emergency_land()

                        response = input("\nTry again? (y/n): ").strip().lower()
                        if response != 'y':
                            break
                        continue

                # Navigate to selected tag
                print("\n" + "="*50)
                print(f"🎯 NAVIGATING TO TAG {tag_id}")
                print("="*50)

                success = self.navigate_to_tag(tag_id)

                if success:
                    print("\n✅ Navigation complete!")
                else:
                    print("\n❌ Navigation failed")
                    if self.in_flight:
                        self.emergency_land()
                    continue

                # Drone is still in the air - ask what to do next
                print("\n" + "="*50)
                print("📋 NEXT ACTION")
                print("="*50)
                print("  1. Navigate to another tag (stay airborne)")
                print("  2. Land and quit")
                print("="*50)

                response = input("\nChoose (1/2): ").strip()

                if response == '1':
                    # Stay in the air, loop continues
                    continue
                elif response == '2':
                    # Land and exit
                    print("\n📍 Landing...")
                    if self.in_flight:
                        self.send_command('land')
                        self.in_flight = False
                    print("✓ Landed")
                    self.get_logger().info('User requested landing and quit')
                    break
                else:
                    print("Invalid choice, landing for safety...")
                    if self.in_flight:
                        self.send_command('land')
                        self.in_flight = False
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
