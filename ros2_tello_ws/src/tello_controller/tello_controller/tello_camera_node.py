#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from djitellopy import Tello
from tello_interfaces.msg import TelloTelemetry
import cv2
import numpy as np


class TelloCameraNode(Node):
    """
    ROS2 Node for streaming Tello camera feed
    Publishes to /tello/camera/image_raw topic
    """
    
    def __init__(self):
        super().__init__('tello_camera_node')
        
        # Create publisher for camera images
        self.image_pub = self.create_publisher(
            Image,
            '/tello/camera/image_raw',
            10
        )

        # Create publisher for telemetry data
        self.telemetry_pub = self.create_publisher(
            TelloTelemetry,
            '/tello/telemetry',
            10
        )

        # Create CV Bridge for converting OpenCV images to ROS messages
        self.bridge = CvBridge()
        
        # Tello object
        self.tello = None
        self.frame_read = None
        
        # Timer for publishing frames
        self.timer = None
        
        # Declare parameters
        self.declare_parameter('publish_rate', 30.0)  # Hz
        self.declare_parameter('show_window', True)
        
        self.publish_rate = self.get_parameter('publish_rate').value
        self.show_window = self.get_parameter('show_window').value
        
        self.get_logger().info('Tello Camera Node Initialized')
        self.get_logger().info(f'Publish rate: {self.publish_rate} Hz')
        
    def connect_tello(self):
        """Connect to Tello and start video stream"""
        try:
            self.get_logger().info('Connecting to Tello...')
            self.tello = Tello()
            self.tello.connect()
            
            battery = self.tello.get_battery()
            self.get_logger().info(f'Connected! Battery: {battery}%')
            
            # Start video stream
            self.get_logger().info('Starting video stream...')
            self.tello.streamon()
            self.frame_read = self.tello.get_frame_read()
            
            self.get_logger().info('✓ Video stream started')
            
            # Create timer for publishing frames
            timer_period = 1.0 / self.publish_rate
            self.timer = self.create_timer(timer_period, self.publish_frame)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Failed to connect: {str(e)}')
            return False
    
    def publish_frame(self):
        """Read frame and publish to ROS topic"""
        if self.frame_read is None or self.frame_read.stopped:
            self.get_logger().warn('Frame reader not available')
            return
        
        try:
            # Get frame from Tello
            frame = self.frame_read.frame
            
            if frame is None:
                return
            
            # Convert BGR to RGB (ROS standard is RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to ROS Image message
            image_msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding='rgb8')
            
            # Add timestamp
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = 'tello_camera'
            
            # Publish
            self.image_pub.publish(image_msg)

            # Publish telemetry alongside camera frame
            if self.tello:
                try:
                    telemetry_msg = self.query_telemetry()
                    self.telemetry_pub.publish(telemetry_msg)
                except Exception as e:
                    self.get_logger().warn(f'Failed to publish telemetry: {str(e)}')

            # Show window if enabled
            if self.show_window:
                cv2.imshow('Tello Camera', frame_rgb)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error publishing frame: {str(e)}')
    
    def cleanup(self):
        """Clean up resources"""
        if self.timer:
            self.timer.cancel()

        if self.tello:
            try:
                self.tello.streamoff()
                self.tello.end()
                self.get_logger().info('Video stream stopped')
            except:
                pass

        cv2.destroyAllWindows()

    def query_telemetry(self) -> TelloTelemetry:
        """Query all telemetry from Tello SDK and package into message."""
        msg = TelloTelemetry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'tello_base_link'
        msg.telemetry_valid = True

        try:
            # Position (cm)
            msg.height = float(self.tello.get_height())
            msg.distance_tof = float(self.tello.get_distance_tof())
            msg.barometer = float(self.tello.get_barometer())

            # Velocity (cm/s)
            msg.speed_x = float(self.tello.get_speed_x())
            msg.speed_y = float(self.tello.get_speed_y())
            msg.speed_z = float(self.tello.get_speed_z())

            # Acceleration (0.001g)
            msg.acceleration_x = float(self.tello.get_acceleration_x())
            msg.acceleration_y = float(self.tello.get_acceleration_y())
            msg.acceleration_z = float(self.tello.get_acceleration_z())

            # Attitude (degrees)
            msg.pitch = float(self.tello.get_pitch())
            msg.roll = float(self.tello.get_roll())
            msg.yaw = float(self.tello.get_yaw())

            # Status
            msg.battery = int(self.tello.get_battery())
            msg.flight_time = int(self.tello.get_flight_time())
            msg.temperature = float(self.tello.get_temperature())
            msg.temperature_min = float(self.tello.get_lowest_temperature())
            msg.temperature_max = float(self.tello.get_highest_temperature())

        except Exception as e:
            self.get_logger().warn(f'Telemetry query failed: {str(e)}')
            msg.telemetry_valid = False

        return msg


def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create node
    node = TelloCameraNode()
    
    try:
        # Connect to drone and start streaming
        if node.connect_tello():
            node.get_logger().info('Camera streaming started. Press Ctrl+C to stop.')
            # Spin to keep publishing
            rclpy.spin(node)
        else:
            node.get_logger().error('Could not connect to Tello')
    
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    
    finally:
        # Cleanup
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()