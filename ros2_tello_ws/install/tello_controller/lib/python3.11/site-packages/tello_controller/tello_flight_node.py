#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from djitellopy import Tello
import time


class TelloFlightNode(Node):
    """
    ROS2 Node for controlling DJI Tello drone
    Mission: Takeoff -> Up 50cm -> Forward 50cm -> Land
    """
    
    def __init__(self):
        super().__init__('tello_flight_node')
        
        # Declare parameters
        self.declare_parameter('up_distance', 20)
        self.declare_parameter('forward_distance', 20)
        
        # Get parameters
        self.up_distance = self.get_parameter('up_distance').value
        self.forward_distance = self.get_parameter('forward_distance').value
        
        # Initialize Tello
        self.tello = None
        self.connected = False
        
        self.get_logger().info('Tello Flight Node Initialized')
        self.get_logger().info(f'Mission: Up {self.up_distance}cm, Forward {self.forward_distance}cm')
        
    def connect_tello(self):
        """Connect to Tello drone"""
        try:
            self.get_logger().info('Connecting to Tello...')
            self.tello = Tello()
            self.tello.connect()
            
            # Get battery info
            battery = self.tello.get_battery()
            self.get_logger().info(f'Connected! Battery: {battery}%')
            
            if battery < 10:
                self.get_logger().error('Battery too low! Cannot fly.')
                return False
            
            self.connected = True
            return True
            
        except Exception as e:
            self.get_logger().error(f'Failed to connect: {str(e)}')
            return False
    
    def execute_mission(self):
        """Execute the complete flight mission"""
        if not self.connected:
            self.get_logger().error('Not connected to Tello!')
            return False
        
        try:
            # Step 1: Takeoff
            self.get_logger().info('Step 1: Taking off...')
            self.tello.takeoff()
            time.sleep(3)
            self.get_logger().info('✓ Takeoff complete')
            
            # Step 2: Move up
            self.get_logger().info(f'Step 2: Moving up {self.up_distance}cm...')
            self.tello.move_up(self.up_distance)
            time.sleep(2)
            height = self.tello.get_height()
            self.get_logger().info(f'✓ Moved up - Current height: {height}cm')
            
            # Step 3: Move forward
            self.get_logger().info(f'Step 3: Moving forward {self.forward_distance}cm...')
            self.tello.move_forward(self.forward_distance)
            time.sleep(2)
            self.get_logger().info('✓ Moved forward')
            
            # Step 4: Land
            self.get_logger().info('Step 4: Landing...')
            self.tello.land()
            time.sleep(3)
            self.get_logger().info('✓ Landed successfully')
            
            # Mission complete
            self.get_logger().info('='*50)
            self.get_logger().info('MISSION COMPLETE!')
            self.get_logger().info('='*50)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Mission failed: {str(e)}')
            self.get_logger().warn('Attempting emergency landing...')
            try:
                self.tello.land()
            except:
                pass
            return False
    
    def cleanup(self):
        """Clean up connection"""
        if self.tello:
            try:
                self.tello.end()
                self.get_logger().info('Connection closed')
            except:
                pass


def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create node
    node = TelloFlightNode()
    
    try:
        # Connect to drone
        if node.connect_tello():
            # Wait a moment
            time.sleep(1)
            
            # Execute mission
            node.execute_mission()
        else:
            node.get_logger().error('Could not connect to Tello. Exiting.')
    
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    
    finally:
        # Cleanup
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
