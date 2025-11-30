from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Camera node
        Node(
            package='tello_controller',
            executable='tello_camera',
            name='tello_camera_node',
            output='screen',
            parameters=[
                {'publish_rate': 30.0},
                {'show_window': True}
            ]
        ),
        # You can add the flight node here too if you want
        # But typically you'd run camera continuously and flight separately
    ])
