#!/usr/bin/env python3
"""
Quadcopter position controller with PID for Gazebo Harmonic.
Usage: python3 goto_position.py <x> <y> <z>
Example: python3 goto_position.py 2 3 5
"""

import sys
import time
import math
import subprocess
import re

class PIDController:
    def __init__(self, kp, ki, kd, max_output=None, max_integral=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.max_integral = max_integral if max_integral is not None else 5.0
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        
    def update(self, error, current_time):
        """Update PID controller with new error."""
        if self.prev_time is None:
            self.prev_time = current_time
            self.prev_error = error
            return self.kp * error
        
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 0.01
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup - only accumulate if error is small
        if abs(error) < 2.0:  # Only integrate when close to target
            self.integral += error * dt
            # Clamp integral to prevent windup
            self.integral = max(min(self.integral, self.max_integral), -self.max_integral)
        else:
            # Reset integral if far from target
            self.integral = 0.0
            
        i_term = self.ki * self.integral
        
        # Derivative term with low-pass filter
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.max_output is not None:
            output = max(min(output, self.max_output), -self.max_output)
        
        # Update state
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def reset(self):
        """Reset the PID controller."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None


class QuadcopterController:
    def __init__(self):
        self.cmd_vel_topic = "/quadcopter/cmd_vel"
        self.enable_topic = "/quadcopter/enable"
        
        # Current pose
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.5
        
        # Conservative PID tuning
        self.pid_x = PIDController(kp=0.25, ki=0.01, kd=0.8, max_output=1.5, max_integral=3.0)
        self.pid_y = PIDController(kp=0.25, ki=0.01, kd=0.8, max_output=1.5, max_integral=3.0)
        self.pid_z = PIDController(kp=0.3, ki=0.02, kd=0.9, max_output=1.5, max_integral=3.0)
        
        print("Waiting for Gazebo to start...")
        time.sleep(2)
        
        # Enable the controller
        self.enable_controller()
        
    def enable_controller(self):
        """Enable the multicopter controller."""
        cmd = [
            "gz", "topic", "-t", self.enable_topic,
            "-m", "gz.msgs.Boolean",
            "-p", "data: true"
        ]
        subprocess.run(cmd, capture_output=True)
        print("Controller enabled")
        print(f"PID Gains - X: Kp={self.pid_x.kp}, Ki={self.pid_x.ki}, Kd={self.pid_x.kd}")
        print(f"PID Gains - Y: Kp={self.pid_y.kp}, Ki={self.pid_y.ki}, Kd={self.pid_y.kd}")
        print(f"PID Gains - Z: Kp={self.pid_z.kp}, Ki={self.pid_z.ki}, Kd={self.pid_z.kd}")
        
    def get_pose(self):
        """Get current pose from Gazebo using gz model command."""
        try:
            cmd = ["gz", "model", "-m", "quadcopter", "-p"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            
            output = result.stdout
            # Look for pattern: [x y z]
            match = re.search(r'\[([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\]', output)
            if match:
                self.current_x = float(match.group(1))
                self.current_y = float(match.group(2))
                self.current_z = float(match.group(3))
                        
        except Exception as e:
            print(f"Warning: Could not read pose: {e}")
            pass
    
    def goto_position(self, target_x, target_y, target_z, tolerance=0.8):
        """
        Command the quadcopter to go to a specific position using PID control.
        
        Args:
            tolerance: Distance in meters to consider "at target" (default: 0.8m)
        """
        print(f"\nFlying to position: ({target_x}, {target_y}, {target_z})")
        print(f"Tolerance: {tolerance}m")
        
        # Reset PID controllers
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        
        max_iterations = 800
        iteration = 0
        start_time = time.time()
        
        # Track if we're stable at target
        stable_count = 0
        required_stable = 10  # Need to be stable for 10 iterations (0.5 seconds)
        
        while iteration < max_iterations:
            current_time = time.time() - start_time
            
            # Get current position
            self.get_pose()
            
            # Calculate errors
            error_x = target_x - self.current_x
            error_y = target_y - self.current_y
            error_z = target_z - self.current_z
            
            # Calculate distance to target
            distance = math.sqrt(error_x**2 + error_y**2 + error_z**2)
            
            # Check if we've reached the target and are stable
            if distance < tolerance:
                stable_count += 1
                if stable_count >= required_stable:
                    print(f"\n✓ Reached target! Final position: ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f})")
                    print(f"   Final error: {distance:.2f}m")
                    self.send_velocity(0, 0, 0, 0)
                    break
            else:
                stable_count = 0
            
            # Calculate desired velocities using PID
            vel_x = self.pid_x.update(error_x, current_time)
            vel_y = self.pid_y.update(error_y, current_time)
            vel_z = self.pid_z.update(error_z, current_time)
            
            # Send velocity command
            self.send_velocity(vel_x, vel_y, vel_z, 0)
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Dist: {distance:.2f}m | Pos: ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f}) | "
                      f"Vel: ({vel_x:.2f}, {vel_y:.2f}, {vel_z:.2f})")
            
            time.sleep(0.05)  # 20 Hz update rate
            iteration += 1
        
        if iteration >= max_iterations:
            print("\nWarning: Maximum iterations reached.")
            self.send_velocity(0, 0, 0, 0)
    
    def send_velocity(self, vx, vy, vz, yaw_rate):
        """Send velocity command to the quadcopter."""
        cmd = [
            "gz", "topic", "-t", self.cmd_vel_topic,
            "-m", "gz.msgs.Twist",
            "-p", f"linear: {{x: {vx}, y: {vy}, z: {vz}}}, angular: {{z: {yaw_rate}}}"
        ]
        subprocess.run(cmd, capture_output=True)
    
    def takeoff(self, height=2.0):
        """Takeoff to a specified height."""
        print(f"\nTaking off to {height}m...")
        self.goto_position(self.current_x, self.current_y, height, tolerance=0.6)
        print("Takeoff complete!")

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 goto_position.py <x> <y> <z>")
        print("Example: python3 goto_position.py 2 3 5")
        sys.exit(1)
    
    try:
        target_x = float(sys.argv[1])
        target_y = float(sys.argv[2])
        target_z = float(sys.argv[3])
    except ValueError:
        print("Error: Coordinates must be numbers")
        sys.exit(1)
    
    if target_z < 0.3:
        print("Warning: Z coordinate too low, setting minimum height of 0.3m")
        target_z = 0.3
    
    print("=" * 60)
    print("Quadcopter PID Position Control")
    print("=" * 60)
    
    controller = QuadcopterController()
    
    try:
        controller.takeoff(height=2.0)
        time.sleep(1)
        controller.goto_position(target_x, target_y, target_z)
        print("\nHolding position for 3 seconds...")
        time.sleep(3)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        controller.send_velocity(0, 0, 0, 0)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
