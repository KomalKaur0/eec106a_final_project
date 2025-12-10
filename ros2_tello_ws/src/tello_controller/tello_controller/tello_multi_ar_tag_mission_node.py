#!/usr/bin/env python3

import time
import math

import rclpy
from rclpy.node import Node
from djitellopy import Tello
import numpy as np
import matplotlib.pyplot as plt

from geometry_msgs.msg import PoseStamped, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from scipy.spatial.transform import Rotation


class TelloMultiArTagMissionNode(Node):
    """
    ROS2 Node for multi-ArUco-tag-guided Tello mission using global map frame.
    
    This node:
    1. Subscribes to the environment node's tag map and drone pose
    2. Plans paths in the global map frame (not camera-relative)
    3. Commands the drone to visit selected tags
    4. Works with RTAB-Map for 3D environment mapping
    
    Prerequisites:
    - tello_camera_node must be running
    - tello_environment_node must be running (builds the map)
    """

    def __init__(self):
        super().__init__("tello_multi_ar_tag_mission_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("marker_ids", [0, 1, 2, 3])
        self.declare_parameter("approach_distance_m", 0.5)  # Stop 50cm from tag
        self.declare_parameter("visit_all_tags", False)
        self.declare_parameter("max_flight_speed_cm", 100)  # Conservative speed
        self.declare_parameter("position_tolerance_m", 0.15)  # 15cm position tolerance

        self.marker_ids = self.get_parameter("marker_ids").value
        self.approach_distance_m = float(self.get_parameter("approach_distance_m").value)
        self.visit_all_tags = bool(self.get_parameter("visit_all_tags").value)
        self.max_speed_cm = int(self.get_parameter("max_flight_speed_cm").value)
        self.position_tolerance_m = float(self.get_parameter("position_tolerance_m").value)

        self.get_logger().info(
            f"Tello Multi-AR Tag Mission Node (Global Frame) initialized\n"
            f"  Marker IDs: {self.marker_ids}\n"
            f"  Approach distance: {self.approach_distance_m} m\n"
            f"  Visit all tags: {self.visit_all_tags}\n"
            f"  Max speed: {self.max_speed_cm} cm/s"
        )

        # Tello connection
        self.tello = None
        self.connected = False

        # Track visited tags
        self.visited_tags = set()

        # State from environment node
        self.known_tags = {}  # {marker_id: {'position': [x,y,z], 'orientation': [x,y,z,w]}}
        self.current_drone_pose = None  # PoseStamped in map frame
        self.map_initialized = False

        # Create subscribers
        self.tag_map_subscriber = self.create_subscription(
            MarkerArray,
            '/world/aruco_poses',
            self.tag_map_callback,
            10
        )

        self.drone_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/tello/drone_pose',
            self.drone_pose_callback,
            10
        )

        self.get_logger().info("Waiting for environment node data...")

    # ------------------------------------------------
    # Subscribers for environment data
    # ------------------------------------------------

    def tag_map_callback(self, msg: MarkerArray):
        """
        Receive the global map of all known AR tags from environment node.
        """
        for marker in msg.markers:
            # Extract marker ID from namespace or id
            marker_id = marker.id
            
            if marker_id not in self.marker_ids:
                continue  # Only track configured markers
            
            self.known_tags[marker_id] = {
                'position': np.array([
                    marker.pose.position.x,
                    marker.pose.position.y,
                    marker.pose.position.z
                ]),
                'orientation': np.array([
                    marker.pose.orientation.x,
                    marker.pose.orientation.y,
                    marker.pose.orientation.z,
                    marker.pose.orientation.w
                ])
            }
        
        if len(self.known_tags) > 0 and not self.map_initialized:
            self.map_initialized = True
            self.get_logger().info(
                f"Map initialized with {len(self.known_tags)} tags: "
                f"{list(self.known_tags.keys())}"
            )

    def drone_pose_callback(self, msg: PoseStamped):
        """
        Receive current drone pose in global map frame from environment node.
        """
        self.current_drone_pose = msg

    # ------------------------------------------------
    # Tello connection
    # ------------------------------------------------

    def connect_tello(self) -> bool:
        """Connect to Tello drone."""
        try:
            self.get_logger().info("Connecting to Tello...")
            self.tello = Tello()
            self.tello.connect()

            battery = self.tello.get_battery()
            self.get_logger().info(f"Connected. Battery: {battery}%")
            
            if battery < 20:
                self.get_logger().error("Battery too low (<20%). Aborting flight.")
                return False

            self.connected = True
            return True

        except Exception as e:
            self.get_logger().error(f"Failed to connect to Tello: {e}")
            return False

    # ------------------------------------------------
    # Wait for environment initialization
    # ------------------------------------------------

    def wait_for_map_initialization(self, timeout_sec=30.0):
        """
        Wait for the environment node to detect at least one AR tag.
        This ensures we have a map frame before starting navigation.
        """
        self.get_logger().info("Waiting for map initialization (need at least 1 tag)...")
        start_time = time.time()
        
        rate = self.create_rate(10)  # 10 Hz
        
        while time.time() - start_time < timeout_sec:
            if self.map_initialized and self.current_drone_pose is not None:
                self.get_logger().info("✓ Map initialized and drone pose available")
                return True
            
            rclpy.spin_once(self, timeout_sec=0.1)
            rate.sleep()
        
        self.get_logger().error("Timeout waiting for map initialization")
        return False

    # ------------------------------------------------
    # Path planning in global frame
    # ------------------------------------------------

    def plan_path_to_tag_global(self, tag_id):
        """
        Plan path from current drone pose to target tag in global map frame.
        
        Returns:
            dict with keys:
                - 'distance_m': horizontal distance to tag
                - 'angle_rad': required heading change
                - 'angle_deg': required heading change in degrees
                - 'dz_m': vertical distance to tag
                - 'target_pos': target position [x, y, z]
        """
        if tag_id not in self.known_tags:
            self.get_logger().warn(f"Tag {tag_id} not in known tags")
            return None
        
        if self.current_drone_pose is None:
            self.get_logger().warn("Current drone pose not available")
            return None
        
        # Current drone position in map frame
        drone_pos = np.array([
            self.current_drone_pose.pose.position.x,
            self.current_drone_pose.pose.position.y,
            self.current_drone_pose.pose.position.z
        ])
        
        # Current drone orientation (yaw)
        drone_quat = self.current_drone_pose.pose.orientation
        drone_rot = Rotation.from_quat([
            drone_quat.x, drone_quat.y, drone_quat.z, drone_quat.w
        ])
        drone_euler = drone_rot.as_euler('xyz', degrees=False)
        current_yaw = drone_euler[2]  # Yaw in radians
        
        # Target tag position
        tag_pos = self.known_tags[tag_id]['position']
        
        # Compute target position (approach_distance away from tag)
        # Calculate direction from tag to drone, then back off
        dx_full = tag_pos[0] - drone_pos[0]
        dy_full = tag_pos[1] - drone_pos[1]
        dist_full = math.sqrt(dx_full**2 + dy_full**2)
        
        if dist_full > self.approach_distance_m:
            # Target is approach_distance away from tag
            approach_ratio = (dist_full - self.approach_distance_m) / dist_full
            target_pos = drone_pos + np.array([dx_full, dy_full, 0]) * approach_ratio
            target_pos[2] = tag_pos[2]  # Match tag height
        else:
            # Already within approach distance
            target_pos = tag_pos.copy()
        
        # Compute required movement
        dx = target_pos[0] - drone_pos[0]
        dy = target_pos[1] - drone_pos[1]
        dz = target_pos[2] - drone_pos[2]
        
        distance_horizontal = math.sqrt(dx**2 + dy**2)
        
        # Required heading in map frame
        target_yaw = math.atan2(dy, dx)
        
        # Heading change needed
        angle_change = self._normalize_angle(target_yaw - current_yaw)
        
        return {
            'distance_m': distance_horizontal,
            'angle_rad': angle_change,
            'angle_deg': math.degrees(angle_change),
            'dz_m': dz,
            'target_pos': target_pos,
            'current_pos': drone_pos
        }

    def _normalize_angle(self, angle_rad):
        """Normalize angle to [-pi, pi]"""
        while angle_rad > math.pi:
            angle_rad -= 2 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2 * math.pi
        return angle_rad

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    def show_multi_path_graph(self, tag_paths):
        """
        Show 2D top-down view of planned paths to all tags in map frame.
        
        tag_paths: list of dicts with path planning results
        """
        plt.figure(figsize=(10, 10))
        
        colors = ['g', 'b', 'r', 'm', 'c', 'y']
        
        # Plot each tag and path
        for idx, path_info in enumerate(tag_paths):
            if path_info['path'] is None:
                continue
                
            color = colors[idx % len(colors)]
            tag_id = path_info['id']
            path = path_info['path']
            
            current = path['current_pos']
            target = path['target_pos']
            
            visited_marker = '*' if tag_id in self.visited_tags else 'o'
            
            # Draw path line
            plt.plot([current[0], target[0]], 
                    [current[1], target[1]], 
                    f"{color}-{visited_marker}",
                    linewidth=2,
                    label=f"Tag {tag_id}")
            
            # Label target
            plt.text(target[0], target[1], f" Tag {tag_id}", 
                    fontsize=10, color=color)
        
        # Plot drone current position
        if self.current_drone_pose:
            drone_x = self.current_drone_pose.pose.position.x
            drone_y = self.current_drone_pose.pose.position.y
            plt.scatter([drone_x], [drone_y], 
                       c='black', s=200, marker='D', 
                       label="Drone", zorder=10)
            
            # Draw drone heading
            drone_quat = self.current_drone_pose.pose.orientation
            drone_rot = Rotation.from_quat([
                drone_quat.x, drone_quat.y, drone_quat.z, drone_quat.w
            ])
            yaw = drone_rot.as_euler('xyz')[2]
            arrow_len = 0.2
            dx_arrow = arrow_len * math.cos(yaw)
            dy_arrow = arrow_len * math.sin(yaw)
            plt.arrow(drone_x, drone_y, dx_arrow, dy_arrow,
                     head_width=0.1, head_length=0.05, 
                     fc='black', ec='black')
        
        plt.title("Global Map - Planned Paths to Tags", fontsize=14)
        plt.xlabel("X (meters)", fontsize=12)
        plt.ylabel("Y (meters)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.legend()
        
        plt.show(block=False)
        plt.pause(0.1)

    # ------------------------------------------------
    # Tag selection
    # ------------------------------------------------

    def select_target_tag(self, tag_paths):
        """
        Let user select which tag to visit.
        
        Returns:
            Selected tag_path dict or None
        """
        print("\n" + "="*60)
        print("AVAILABLE TAGS IN MAP:")
        for idx, path_info in enumerate(tag_paths):
            tag_id = path_info['id']
            path = path_info['path']
            
            if path is None:
                print(f"  [{idx}] Tag {tag_id}: PATH UNAVAILABLE")
                continue
            
            visited = " (VISITED)" if tag_id in self.visited_tags else ""
            print(f"  [{idx}] Tag {tag_id}: "
                  f"Distance={path['distance_m']:.2f}m, "
                  f"Heading change={path['angle_deg']:.1f}°, "
                  f"Height change={path['dz_m']:.2f}m{visited}")
        print("="*60)
        
        while True:
            choice = input("\nEnter tag number to visit (or 'q' to land): ").strip()
            
            if choice.lower() == 'q':
                return None
            
            try:
                idx = int(choice)
                if 0 <= idx < len(tag_paths):
                    if tag_paths[idx]['path'] is not None:
                        return tag_paths[idx]
                    else:
                        print("That tag has no valid path")
                else:
                    print(f"Invalid choice. Enter 0-{len(tag_paths)-1}")
            except ValueError:
                print("Invalid input. Enter a number or 'q'")

    # ------------------------------------------------
    # Navigation execution
    # ------------------------------------------------

    def execute_path_to_tag(self, path_info):
        """
        Execute navigation to target tag using global path planning.
        
        Uses closed-loop control: repeatedly check position and adjust.
        """
        tag_id = path_info['id']
        path = path_info['path']
        
        self.get_logger().info(f"Navigating to Tag {tag_id}...")
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Recompute path with latest position
            rclpy.spin_once(self, timeout_sec=0.1)
            current_path = self.plan_path_to_tag_global(tag_id)
            
            if current_path is None:
                self.get_logger().error("Lost tracking, aborting navigation")
                return False
            
            distance = current_path['distance_m']
            angle_deg = current_path['angle_deg']
            dz_m = current_path['dz_m']
            
            self.get_logger().info(
                f"Iteration {iteration}: dist={distance:.2f}m, "
                f"angle={angle_deg:.1f}°, dz={dz_m:.2f}m"
            )
            
            # Check if arrived
            if distance < self.position_tolerance_m and abs(dz_m) < 0.1:
                self.get_logger().info(f"✓ Arrived at Tag {tag_id}")
                self.visited_tags.add(tag_id)
                return True
            
            # Step 1: Rotate to face target
            if abs(angle_deg) > 10.0:
                rotate_deg = int(np.clip(angle_deg, -90, 90))
                self.get_logger().info(f"Rotating {rotate_deg}°")
                
                try:
                    if rotate_deg > 0:
                        self.tello.rotate_clockwise(abs(rotate_deg))
                    else:
                        self.tello.rotate_counter_clockwise(abs(rotate_deg))
                    time.sleep(1.5)
                except Exception as e:
                    self.get_logger().error(f"Rotation failed: {e}")
                
                continue  # Re-evaluate after rotation
            
            # Step 2: Adjust height if needed
            if abs(dz_m) > 0.15:
                dz_cm = int(np.clip(dz_m * 100, -100, 100))
                self.get_logger().info(f"Adjusting height {dz_cm}cm")
                
                try:
                    if dz_cm > 20:
                        self.tello.move_up(min(abs(dz_cm), 100))
                    elif dz_cm < -20:
                        self.tello.move_down(min(abs(dz_cm), 100))
                    time.sleep(1.2)
                except Exception as e:
                    self.get_logger().error(f"Height adjustment failed: {e}")
                
                continue
            
            # Step 3: Move forward toward target
            if distance > self.position_tolerance_m:
                # Conservative forward movement
                forward_cm = int(np.clip(distance * 100, 20, 150))
                self.get_logger().info(f"Moving forward {forward_cm}cm")
                
                try:
                    self.tello.move_forward(forward_cm)
                    time.sleep(1.5)
                except Exception as e:
                    self.get_logger().error(f"Forward movement failed: {e}")
                    return False
        
        self.get_logger().warn(f"Max iterations reached for Tag {tag_id}")
        return False

    # ------------------------------------------------
    # Main mission
    # ------------------------------------------------

    def execute_mission(self):
        """
        Full mission: takeoff, navigate to tags using global map, land.
        """
        if not self.connected:
            self.get_logger().error("Not connected to Tello")
            return False
        
        # Wait for map initialization
        if not self.wait_for_map_initialization():
            self.get_logger().error("Map not initialized, cannot start mission")
            return False
        
        try:
            # Takeoff
            self.get_logger().info("Taking off...")
            self.tello.takeoff()
            time.sleep(3.0)
            self.get_logger().info("✓ Takeoff complete")
            
            # Give environment node time to get initial pose
            self.get_logger().info("Stabilizing and getting initial pose...")
            for _ in range(30):  # 3 seconds at 10Hz
                rclpy.spin_once(self, timeout_sec=0.1)
                time.sleep(0.1)
            
            # Main navigation loop
            while True:
                # Get current state
                rclpy.spin_once(self, timeout_sec=0.1)
                
                # Plan paths to all known tags
                tag_paths = []
                for tag_id in self.known_tags.keys():
                    path = self.plan_path_to_tag_global(tag_id)
                    tag_paths.append({
                        'id': tag_id,
                        'path': path
                    })
                
                if len(tag_paths) == 0:
                    self.get_logger().warn("No tags available")
                    break
                
                # Show visualization
                self.show_multi_path_graph(tag_paths)
                
                # Select target
                if self.visit_all_tags:
                    # Auto-visit unvisited tags
                    unvisited = [p for p in tag_paths 
                               if p['id'] not in self.visited_tags 
                               and p['path'] is not None]
                    if len(unvisited) == 0:
                        self.get_logger().info("✓ All tags visited!")
                        break
                    selected = unvisited[0]
                    print(f"\n[AUTO] Visiting Tag {selected['id']}")
                else:
                    # Manual selection
                    selected = self.select_target_tag(tag_paths)
                    if selected is None:
                        self.get_logger().info("User chose to land")
                        break
                
                # Navigate to selected tag
                success = self.execute_path_to_tag(selected)
                
                if not success:
                    self.get_logger().warn("Navigation failed")
                    retry = input("Retry? (y/n): ").strip().lower()
                    if retry != 'y':
                        break
                
                # Check if should continue
                if not self.visit_all_tags:
                    cont = input("\nVisit another tag? (y/n): ").strip().lower()
                    if cont != 'y':
                        break
            
            # Land
            self.get_logger().info("Landing...")
            self.tello.land()
            time.sleep(3.0)
            self.get_logger().info("✓ Mission complete")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Mission failed: {e}")
            self.get_logger().warn("Attempting emergency landing...")
            try:
                if self.tello:
                    self.tello.land()
                    time.sleep(3.0)
            except:
                pass
            return False

    # ------------------------------------------------
    # Cleanup
    # ------------------------------------------------

    def cleanup(self):
        """Clean up resources."""
        self.get_logger().info("Cleaning up...")
        if self.tello:
            try:
                self.tello.end()
            except:
                pass
        plt.close('all')
        self.get_logger().info("Cleanup done")


def main(args=None):
    rclpy.init(args=args)
    node = TelloMultiArTagMissionNode()
    
    try:
        if node.connect_tello():
            time.sleep(1.0)
            node.execute_mission()
        else:
            node.get_logger().error("Could not connect to Tello")
    except KeyboardInterrupt:
        try:
            if node.tello:
                node.get_logger().info("KeyboardInterrupt! Emergency landing...")
                node.tello.land()
        except Exception as e:
            node.get_logger().error(f"Emergency land failed: {e}")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()