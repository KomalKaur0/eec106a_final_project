# eecs106a_final_project
final project repo for ee106a FA25. team 9 project members: komal, max, cindy, lydia, nikki

## launch sequence

### First, launch the environment node
ros2 launch tello_controller tello_launch_simple.py

### Then in another terminal, run the mission
ros2 run tello_controller tello_multi_ar_tag_mission_node

### Or with custom sequence
ros2 run tello_controller tello_multi_ar_tag_mission_node \
  --ros-args \
  -p tag_sequence:=[0,2,1,3]

## TF tree
map

 └── base_link

      └── camera_link

           └── tag_{marker_id}

### coordinate convention
map (world frame):

X: Forward (from first tag's perspective)

Y: Left

Z: Up

base link:

X: Forward (drone's nose direction)

Y: Left (drone's left side)

Z: Up (drone's top)

camera_link:

X: Right (in image)

Y: Down (in image)

Z: Forward (into scene)
