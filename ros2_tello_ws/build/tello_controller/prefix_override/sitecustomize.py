import sys
if sys.prefix == '/opt/miniconda3/envs/ros2':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/Users/xiaoyuecindyhuang/ros2_tello_ws/install/tello_controller'
