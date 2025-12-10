#!/usr/bin/env python3
"""
Multi AR Tag Mission Node for Tello

Autonomous mission that navigates through multiple AR tags in sequence.
Uses ROS2 topics for localization from tello_environment_node.

Mission Flow:
1. Takeoff
2. Navigate to tag 0
3. Navigate to tag 1
4. Navigate to tag 2
5. Land

Subscribes to:
- /tello/drone_pose - Current drone position
- /world/aruco_poses - Known AR tag positions
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from djitellopy import Tello
import time
import numpy as np
from scipy.spatial.transform import Rotation


class TelloMultiTagMissionNode(Node):
    """
    ROS2 Node for multi-waypoint AR tag navigation
    """
    
    def __init__(self):
        super().__init__('tello_multi_tag_mission_node')
        
        # Parameters
        self.declare_parameter('tag_sequence', [0, 1, 2])
        self.declare_parameter('approach_distance', 0.5)  # meters - stop this far from tag
        self.declare_parameter('position_tolerance', 0.15)  # meters - "reached" threshold
        self.declare_parameter('hover_time', 2.0)  # seconds to hover at each waypoint
        
        self.tag_sequence = self.get_parameter('tag_sequence').value
        self.approach_distance = self.get_parameter('approach_distance').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.hover_time = self.get_parameter('hover_time').value
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/tello/drone_pose',
            self.pose_callback,
            10
        )
        
        self.markers_sub = self.create_subscription(
            MarkerArray,
            '/world/aruco_poses',
            self.markers_callback,
            10
        )
        
        # State
        self.current_pose = None
        self.tag_positions = {}  # {tag_id: (x, y, z)}
        self.tello = None
        self.connected = False
        
        self.get_logger().info('Multi-Tag Mission Node initialized')
        self.get_logger().info(f'Tag sequence: {self.tag_sequence}')
        
    def pose_callback(self, msg: PoseStamped):
        """Update current drone pose"""
        self.current_pose = msg
        
    def markers_callback(self, msg: MarkerArray):
        """Update known tag positions"""
        for marker in msg.markers:
            if marker.ns == "aruco_tags":  # Only process tag markers, not labels
                tag_id = marker.id
                pos = marker.pose.position
                self.tag_positions[tag_id] = (pos.x, pos.y, pos.z)
    
    def connect_tello(self) -> bool:
        """Connect to Tello drone"""
        try:
            self.get_logger().info('Connecting to Tello...')
            self.tello = Tello()
            self.tello.connect()
            
            battery = self.tello.get_battery()
            self.get_logger().info(f'Connected! Battery: {battery}%')
            
            if battery < 20:
                self.get_logger().error('Battery too low (<20%). Aborting.')
                return False
            
            self.connected = True
            return True
            
        except Exception as e:
            self.get_logger().error(f'Failed to connect: {e}')
            return False
    
    def get_distance_to_target(self, target_pos):
        """Calculate distance from current pose to target position"""
        if self.current_pose is None:
            return float('inf')
        
        current = self.current_pose.pose.position
        dx = target_pos[0] - current.x
        dy = target_pos[1] - current.y
        dz = target_pos[2] - current.z
        
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def get_relative_position(self, target_pos):
        """
        Get target position relative to drone's current orientation
        Returns: (forward, right, up) in meters relative to drone
        """
        if self.current_pose is None:
            return None
        
        current = self.current_pose.pose.position
        current_quat = self.current_pose.pose.orientation
        
        # Vector from drone to target in map frame
        dx = target_pos[0] - current.x
        dy = target_pos[1] - current.y
        dz = target_pos[2] - current.z
        
        # Convert quaternion to rotation matrix
        rot = Rotation.from_quat([
            current_quat.x,
            current_quat.y, 
            current_quat.z,
            current_quat.w
        ])
        
        # Transform to drone body frame
        # Map frame vector
        vec_map = np.array([dx, dy, dz])
        
        # Rotate to body frame (inverse rotation)
        vec_body = rot.inv().apply(vec_map)
        
        # Return as (forward, right, up)
        return vec_body[0], vec_body[1], vec_body[2]
    
    def navigate_to_tag(self, tag_id: int) -> bool:
        """
        Navigate to a specific AR tag
        Returns True if successful
        """
        self.get_logger().info(f'Navigating to tag {tag_id}...')
        
        # Wait for tag to be discovered
        timeout = 30.0
        start_time = time.time()
        
        while tag_id not in self.tag_positions:
            if time.time() - start_time > timeout:
                self.get_logger().error(f'Tag {tag_id} not found within {timeout}s')
                return False
            
            self.get_logger().info(f'Waiting for tag {tag_id} to be discovered...')
            time.sleep(1.0)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        target_pos = self.tag_positions[tag_id]
        self.get_logger().info(
            f'Tag {tag_id} at position: '
            f'[{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]'
        )
        
        # Navigate in steps until close enough
        max_iterations = 50
        iteration = 0
        
        while iteration < max_iterations:
            rclpy.spin_once(self, timeout_sec=0.1)
            
            if self.current_pose is None:
                self.get_logger().warn('No pose update, waiting...')
                time.sleep(0.5)
                continue
            
            distance = self.get_distance_to_target(target_pos)
            
            if distance < self.position_tolerance:
                self.get_logger().info(
                    f'Reached tag {tag_id} (distance: {distance:.2f}m)'
                )
                return True
            
            # Get relative position
            rel_pos = self.get_relative_position(target_pos)
            if rel_pos is None:
                self.get_logger().warn('Could not compute relative position')
                time.sleep(0.5)
                continue
            
            forward, right, up = rel_pos
            
            self.get_logger().info(
                f'Distance to tag {tag_id}: {distance:.2f}m | '
                f'Relative: F={forward:.2f}, R={right:.2f}, U={up:.2f}'
            )
            
            # Compute movement commands (conservative steps)
            # Stop before reaching tag (approach_distance)
            if distance > self.approach_distance + 0.2:
                # Move in steps
                move_forward = min(max(int(forward * 50), -100), 100)  # cm, clamped
                move_right = min(max(int(right * 50), -100), 100)
                move_up = min(max(int(up * 50), -100), 100)
                
                # Execute moves
                try:
                    if abs(move_forward) > 20:
                        if move_forward > 0:
                            self.get_logger().info(f'Moving forward {move_forward}cm')
                            self.tello.move_forward(move_forward)
                        else:
                            self.get_logger().info(f'Moving back {-move_forward}cm')
                            self.tello.move_back(-move_forward)
                        time.sleep(1.5)
                    
                    if abs(move_right) > 20:
                        if move_right > 0:
                            self.get_logger().info(f'Moving right {move_right}cm')
                            self.tello.move_right(move_right)
                        else:
                            self.get_logger().info(f'Moving left {-move_right}cm')
                            self.tello.move_left(-move_right)
                        time.sleep(1.5)
                    
                    if abs(move_up) > 20:
                        if move_up > 0:
                            self.get_logger().info(f'Moving up {move_up}cm')
                            self.tello.move_up(move_up)
                        else:
                            self.get_logger().info(f'Moving down {-move_up}cm')
                            self.tello.move_down(-move_up)
                        time.sleep(1.5)
                    
                except Exception as e:
                    self.get_logger().error(f'Movement failed: {e}')
                    return False
                
            else:
                # Close enough, stop
                self.get_logger().info(f'Within approach distance of tag {tag_id}')
                return True
            
            iteration += 1
        
        self.get_logger().warn(f'Max iterations reached for tag {tag_id}')
        return False
    
    def execute_mission(self):
        """Execute the multi-waypoint mission"""
        if not self.connected:
            self.get_logger().error('Not connected to Tello')
            return False
        
        try:
            # Takeoff
            self.get_logger().info('Taking off...')
            self.tello.takeoff()
            time.sleep(3.0)
            self.get_logger().info('Takeoff complete')
            
            # Wait for initial localization
            self.get_logger().info('Waiting for initial localization...')
            timeout = 10.0
            start = time.time()
            while self.current_pose is None and (time.time() - start) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
                time.sleep(0.1)
            
            if self.current_pose is None:
                self.get_logger().error('No pose received, aborting mission')
                self.tello.land()
                return False
            
            self.get_logger().info('Localization acquired, starting waypoint navigation')
            
            # Navigate through each tag in sequence
            for i, tag_id in enumerate(self.tag_sequence):
                self.get_logger().info(
                    f'Waypoint {i+1}/{len(self.tag_sequence)}: Tag {tag_id}'
                )
                
                success = self.navigate_to_tag(tag_id)
                
                if not success:
                    self.get_logger().error(f'Failed to reach tag {tag_id}, aborting')
                    self.tello.land()
                    return False
                
                # Hover at waypoint
                self.get_logger().info(f'Hovering at tag {tag_id} for {self.hover_time}s')
                time.sleep(self.hover_time)
            
            # Mission complete, land
            self.get_logger().info('All waypoints reached! Landing...')
            self.tello.land()
            time.sleep(3.0)
            
            self.get_logger().info('='*50)
            self.get_logger().info('MISSION COMPLETE!')
            self.get_logger().info('='*50)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Mission failed: {e}')
            self.get_logger().warn('Attempting emergency landing...')
            try:
                self.tello.land()
            except:
                pass
            return False
    
    def cleanup(self):
        """Cleanup connections"""
        if self.tello:
            try:
                self.tello.end()
            except:
                pass
        self.get_logger().info('Cleanup complete')


def main(args=None):
    rclpy.init(args=args)
    node = TelloMultiTagMissionNode()
    
    try:
        if node.connect_tello():
            # Give time for ROS topics to populate
            time.sleep(2.0)
            
            # Execute mission
            node.execute_mission()
        else:
            node.get_logger().error('Could not connect to Tello')
    
    except KeyboardInterrupt:
        node.get_logger().info('Mission interrupted by user')
        try:
            if node.tello:
                node.get_logger().info('Emergency landing...')
                node.tello.land()
        except:
            pass
    
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()