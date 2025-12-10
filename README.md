# eecs106a_final_project
final project repo for ee106a FA25. team 9 project members: komal, max, cindy, lydia, nikki

## launch sequence

### Terminal 1: Camera stream
ros2 run tello_controller tello_camera_node

### Terminal 2: Environment mapping (builds tag map)
ros2 run tello_controller tello_environment_node

### Terminal 3: Mission execution
ros2 run tello_controller tello_multi_ar_tag_mission_node

### Terminal 4: RTAB-Map
ros2 launch rtabmap_ros rtabmap.launch.py \
    rgb_topic:=/tello/camera/image_raw \
    camera_info_topic:=/tello/camera/camera_info \
    frame_id:=camera_link \
    approx_sync:=true

## file structure
tello_controller/
├── tello_controller/
│   ├── __init__.py
│   ├── tello_constants.py
│   ├── tello_camera_node.py (UPDATED)
│   ├── tello_environment_node.py (UPDATED)
│   ├── tello_multi_ar_tag_mission_node.py (UPDATED)
│   └── (other nodes...)
├── launch/
│   └── tello_rtabmap.launch.py (NEW)
├── rviz/
│   └── tello_rtabmap.rviz (NEW)
├── setup.py
└── package.xml

## TF tree
map (AR tag origin)
 └─ odom
     └─ base_link (drone body)
         └─ camera_link (Tello camera)

## Handoff
Tello Camera
    ↓ /tello/camera/image_raw (sensor_msgs/Image)
    ↓ /tello/camera/camera_info (sensor_msgs/CameraInfo)
    
Environment Node
    ↓ Detects AR tags → Builds map
    ↓ /world/aruco_poses (visualization_msgs/MarkerArray)
    ↓ /odom (nav_msgs/Odometry)
    ↓ TF: map → odom → base_link → camera_link
    
RTAB-Map
    ↓ Consumes images + odometry + TF
    ↓ Builds 3D point cloud map
    ↓ /rtabmap/cloud_map (sensor_msgs/PointCloud2)