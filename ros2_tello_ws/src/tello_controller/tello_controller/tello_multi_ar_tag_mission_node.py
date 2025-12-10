#!/usr/bin/env python3
"""
Enhanced Tello Camera Node with Camera Info Publisher

Publishes:
- /tello/camera/image_raw (sensor_msgs/Image)
- /tello/camera/camera_info (sensor_msgs/CameraInfo)
- /tello/telemetry (tello_interfaces/TelloTelemetry)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from djitellopy import Tello
from tello_interfaces.msg import TelloTelemetry
import cv2
import numpy as np

from tello_controller import tello_constants as tc


class TelloCameraNode(Node):
    """
    ROS2 Node for streaming Tello camera feed with calibration info
    """
    
    def __init__(self):
        super().__init__('tello_camera_node')
        
        # Create publishers
        self.image_pub = self.create_publisher(
            Image,
            '/tello/camera/image_raw',
            10
        )

        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            '/tello/camera/camera_info',
            10
        )

        self.telemetry_pub = self.create_publisher(
            TelloTelemetry,
            '/tello/telemetry',
            10
        )

        # CV Bridge
        self.bridge = CvBridge()
        
        # Tello object
        self.tello = None
        self.frame_read = None
        
        # Timer for publishing frames
        self.timer = None
        
        # Declare parameters
        self.declare_parameter('publish_rate', 30.0)  # Hz
        self.declare_parameter('show_window', False)  # Changed to False for headless
        
        self.publish_rate = self.get_parameter('publish_rate').value
        self.show_window = self.get_parameter('show_window').value
        
        # Build camera info message (constant for all frames)
        self.camera_info_msg = self._build_camera_info()
        
        self.get_logger().info('Tello Camera Node Initialized')
        self.get_logger().info(f'Publish rate: {self.publish_rate} Hz')
        
    def _build_camera_info(self) -> CameraInfo:
        """
        Build CameraInfo message from calibration constants.
        """
        msg = CameraInfo()
        
        # Image dimensions
        msg.height = tc.FRAME_HEIGHT
        msg.width = tc.FRAME_WIDTH
        
        # Distortion model
        msg.distortion_model = "plumb_bob"
        msg.d = tc.DISTORTION_COEFFS.tolist()
        
        # Intrinsic camera matrix (3x3)
        msg.k = tc.CAMERA_MATRIX.flatten().tolist()
        
        # Rectification matrix (identity for monocular)
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        
        # Projection matrix (3x4)
        # P = [fx  0  cx  0]
        #     [ 0 fy  cy  0]
        #     [ 0  0   1  0]
        msg.p = [tc.FOCAL_LENGTH_PX, 0.0, tc.PRINCIPAL_POINT_X, 0.0,
                 0.0, tc.FOCAL_LENGTH_PX, tc.PRINCIPAL_POINT_Y, 0.0,
                 0.0, 0.0, 1.0, 0.0]
        
        # Binning (no binning)
        msg.binning_x = 0
        msg.binning_y = 0
        
        # ROI (region of interest) - use full image
        msg.roi.do_rectify = False
        
        return msg
        
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
        """Read frame and publish to ROS topics"""
        if self.frame_read is None or self.frame_read.stopped:
            self.get_logger().warn('Frame reader not available')
            return
        
        try:
            # Get frame from Tello
            frame = self.frame_read.frame
            
            if frame is None:
                return
            
            # Get timestamp
            current_time = self.get_clock().now().to_msg()
            
            # Convert BGR to RGB (ROS standard is RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Publish image
            image_msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding='rgb8')
            image_msg.header.stamp = current_time
            image_msg.header.frame_id = 'camera_link'
            self.image_pub.publish(image_msg)
            
            # Publish camera info (synchronized with image)
            self.camera_info_msg.header.stamp = current_time
            self.camera_info_msg.header.frame_id = 'camera_link'
            self.camera_info_pub.publish(self.camera_info_msg)

            # Publish telemetry
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
        msg.header.frame_id = 'base_link'
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
    rclpy.init(args=args)
    node = TelloCameraNode()
    
    try:
        if node.connect_tello():
            node.get_logger().info('Camera streaming started. Press Ctrl+C to stop.')
            rclpy.spin(node)
        else:
            node.get_logger().error('Could not connect to Tello')
    
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()