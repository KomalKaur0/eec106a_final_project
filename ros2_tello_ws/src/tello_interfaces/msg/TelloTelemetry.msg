# Header with timestamp and frame_id
std_msgs/Header header

# Position (cm from SDK)
float32 height              # Barometer/TOF fusion height
float32 distance_tof        # Time-of-Flight sensor
float32 barometer           # Barometric altitude

# Velocity (cm/s from SDK)
float32 speed_x
float32 speed_y
float32 speed_z

# Acceleration (0.001g units from SDK)
float32 acceleration_x
float32 acceleration_y
float32 acceleration_z

# Attitude (degrees from SDK)
float32 pitch
float32 roll
float32 yaw

# Status
int8 battery                # 0-100%
int32 flight_time           # Seconds
float32 temperature
float32 temperature_min
float32 temperature_max

# Quality flag
bool telemetry_valid        # False if queries failed
