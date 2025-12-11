#!/usr/bin/env python3

import time
import math

import rclpy
from rclpy.node import Node
from djitellopy import Tello
import cv2
import numpy as np
import matplotlib.pyplot as plt



class TelloArTagMissionNode(Node):
    """
    ROS2 Node for AR-tag-guided Tello mission.

    Pipeline:
      1. Connect and takeoff
      2. Hover at a fixed height (we now just use Tello's default takeoff height)
      3. Rotate to search for an ArUco tag
      4. Once found, compute:
           - Angle to tag (yaw)
           - Distance to tag (using tag size)
           - Lateral offset
      5. Show a graph of the planned path
      6. Wait for user to press ENTER to approve
      7. Execute the planned path (rotate + short forward)
      8. Directly land (no manual down)
    """

    def __init__(self):
        super().__init__("tello_ar_tag_mission_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("hover_height", 70)        # kept but no longer used for extra up
        self.declare_parameter("marker_id", 1)            # ArUco ID
        self.declare_parameter("tag_size_cm", 17.7)       # your tag = 177 mm
        self.declare_parameter("search_timeout", 40.0)    # seconds
        self.declare_parameter("show_debug_window", True)

        self.hover_height = int(self.get_parameter("hover_height").value)
        self.marker_id = int(self.get_parameter("marker_id").value)
        self.tag_size_cm = float(self.get_parameter("tag_size_cm").value)
        self.search_timeout = float(self.get_parameter("search_timeout").value)
        self.show_debug = bool(self.get_parameter("show_debug_window").value)

        # Clamp hover height to safe range (not actually used anymore)
        self.hover_height = max(40, min(self.hover_height, 150))

        # Approximate focal length for Tello camera in pixels.
        # This is an empirical estimate; you can tune it if distances are off.
        self.focal_length_px = 920.0

        self.get_logger().info(
            f"Tello AR Tag Mission Node initialized\n"
            f"  Hover height (unused extra): {self.hover_height} cm\n"
            f"  Marker ID: {self.marker_id}\n"
            f"  Tag size: {self.tag_size_cm} cm\n"
            f"  Search timeout: {self.search_timeout} s"
        )

        # Tello & video
        self.tello = None
        self.frame_read = None
        self.connected = False

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()

    # ------------------------------------------------
    # Tello connection & video
    # ------------------------------------------------

    def connect_tello(self) -> bool:
        """Connect to Tello and start video stream."""
        try:
            self.get_logger().info("Connecting to Tello...")
            self.tello = Tello()
            self.tello.connect()

            battery = self.tello.get_battery()
            self.get_logger().info(f"Connected. Battery: {battery}%")
            if battery < 20:
                self.get_logger().error("Battery too low (<20%). Aborting flight.")
                return False

            # Start video
            self.get_logger().info("Starting video stream...")
            self.tello.streamon()
            self.frame_read = self.tello.get_frame_read()
            time.sleep(1.0)  # give stream time to start

            self.connected = True
            self.get_logger().info("Video stream started.")
            return True

        except Exception as e:
            self.get_logger().error(f"Failed to connect to Tello: {e}")
            return False

    def get_frame(self):
        """Safely get the latest frame from Tello camera."""
        if self.frame_read is None:
            return None
        frame = self.frame_read.frame
        if frame is None:
            return None
        return frame.copy()

    # ------------------------------------------------
    # ArUco detection
    # ------------------------------------------------

    def detect_marker(self, frame):
        """
        Detect the desired ArUco marker.

        Returns:
            (center_x, center_y, tag_width_px, corners) or None if not found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if ids is None or len(ids) == 0:
            return None

        ids = ids.flatten()
        chosen_idx = None

        for i, mid in enumerate(ids):
            if mid == self.marker_id:
                chosen_idx = i
                break

        if chosen_idx is None:
            return None

        marker_corners = corners[chosen_idx][0]  # shape (4,2)

        # Center
        c_x = int(np.mean(marker_corners[:, 0]))
        c_y = int(np.mean(marker_corners[:, 1]))

        # Approximate pixel width of the tag (distance between two adjacent corners)
        w_px = int(np.linalg.norm(marker_corners[0] - marker_corners[1]))

        return c_x, c_y, w_px, marker_corners

    def draw_debug(self, frame, cx, cy, marker_corners):
        """Draw detection overlays for debugging."""
        if frame is None:
            return

        h, w = frame.shape[:2]
        cx_img, cy_img = w // 2, h // 2

        # Marker outline
        if marker_corners is not None:
            cv2.polylines(
                frame,
                [marker_corners.astype(np.int32)],
                True,
                (0, 255, 0),
                2
            )

        # Image center
        cv2.circle(frame, (cx_img, cy_img), 4, (255, 0, 0), -1)

        # Marker center
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(frame, (cx_img, cy_img), (cx, cy), (0, 255, 255), 2)

        cv2.putText(
            frame,
            'AR-tag navigation',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow('Tello AR Tag', frame)
        cv2.waitKey(1)

    # ------------------------------------------------
    # Movement helpers
    # ------------------------------------------------

    def safe_move(self, command: str, distance_cm: int):
        """
        Wrapper for move commands with logging and small delays.
        command: 'forward', 'back', 'left', 'right', 'up', 'down'
        """
        distance_cm = int(max(20, min(distance_cm, 200)))  # clamp 20–200 cm
        self.get_logger().info(f"Command: {command} {distance_cm} cm")
        try:
            if command == 'forward':
                self.tello.move_forward(distance_cm)
            elif command == 'back':
                self.tello.move_back(distance_cm)
            elif command == 'left':
                self.tello.move_left(distance_cm)
            elif command == 'right':
                self.tello.move_right(distance_cm)
            elif command == 'up':
                self.tello.move_up(distance_cm)
            elif command == 'down':
                self.tello.move_down(distance_cm)
            else:
                self.get_logger().warn(f"Unknown move command: {command}")
                return
            time.sleep(1.2)  # allow motion to complete
        except Exception as e:
            self.get_logger().error(f"Move {command} failed: {e}")

    # ------------------------------------------------
    # Search for marker
    # ------------------------------------------------

    def search_for_marker(self):
        """
        Rotate slowly until the marker is found or we time out.

        Returns:
            (cx, cy, w_px, corners) or None
        """
        self.get_logger().info("Searching for marker by rotating...")
        start_time = time.time()
        last_rotate = 0.0
        rotate_interval = 2.0  # seconds between small yaw commands

        while time.time() - start_time < self.search_timeout:
            frame = self.get_frame()
            if frame is None:
                continue

            det = self.detect_marker(frame)
            if det is not None:
                cx, cy, w_px, corners = det
                self.get_logger().info(f"Marker detected at ({cx}, {cy}), width {w_px}px")
                if self.show_debug:
                    self.draw_debug(frame, cx, cy, corners)
                return cx, cy, w_px, corners

            # No marker yet; rotate a bit every few seconds
            if time.time() - last_rotate > rotate_interval:
                self.get_logger().info("Rotating clockwise to search...")
                try:
                    self.tello.rotate_clockwise(20)  # small yaw
                except Exception as e:
                    self.get_logger().error(f"Rotate failed: {e}")
                last_rotate = time.time()

            if self.show_debug:
                self.draw_debug(frame, *self._dummy_center_and_corners(frame))

        self.get_logger().warn("Search timeout reached – marker not found.")
        return None

    def _dummy_center_and_corners(self, frame):
        """Helper to avoid errors when drawing debug with no detection."""
        if frame is None:
            return 0, 0, None
        h, w = frame.shape[:2]
        return w // 2, h // 2, None

    # ------------------------------------------------
    # Path planning (angle + distance)
    # ------------------------------------------------

    def plan_path(self, dx_px: float, dy_px: float, w_px: float):
        """
        Compute angle and distance to the tag.

        dx_px: horizontal pixel offset (tag_x - center_x)
        dy_px: vertical pixel offset (tag_y - center_y) [not used in this basic plan]
        w_px:  perceived tag width in pixels

        Returns:
            angle_deg: yaw angle to rotate (positive = rotate clockwise)
            X: lateral offset in cm (right positive)
            Y: forward distance in cm (forward positive)
            Z: range estimate to tag in cm
        """
        f = self.focal_length_px
        W = self.tag_size_cm

        # Estimate range (distance along camera forward axis)
        if w_px <= 0:
            self.get_logger().warn("Tag pixel width is zero or negative; using fallback.")
            Z = 50.0
        else:
            Z = (W * f) / float(w_px)

        # Horizontal angle (radians) from image center
        angle_rad = math.atan2(dx_px, f)
        angle_deg = angle_rad * 180.0 / math.pi

        # Compute planar offsets in drone frame
        X = Z * math.sin(angle_rad)  # right is positive
        Y = Z * math.cos(angle_rad)  # forward is positive

        self.get_logger().info(
            f"PLAN: angle={angle_deg:.1f} deg, X={X:.1f} cm, Y={Y:.1f} cm, Z={Z:.1f} cm"
        )

        return angle_deg, X, Y, Z

    def show_path_graph_and_wait(self, X: float, Y: float):
        """
        Show a 2D graph of the planned path and wait for user approval.
        """
        self.get_logger().info("Showing planned path graph...")

        plt.figure()
        plt.plot([0, X], [0, Y], "-o")
        plt.scatter([0], [0], label="Drone start")
        plt.scatter([X], [Y], label="AR Tag")
        plt.text(X, Y, " AR Tag")
        plt.title("Planned Drone Path (cm)")
        plt.xlabel("Right (+)")
        plt.ylabel("Forward (+)")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()

        # Non-blocking show so ROS / Tello don't time out
        plt.show(block=False)
        plt.pause(0.1)

        # Wait for manual approval
        input("✔ Path planned. Press ENTER in this terminal to execute the planned path... ")

    # ------------------------------------------------
    # Execute path
    # ------------------------------------------------

    def execute_path(self, angle_deg: float, X: float, Y: float):
        """
        Rotate toward the tag, then move straight forward.
        We intentionally DO NOT apply lateral correction after rotation,
        because the forward direction is already aligned with the tag.

        修改点：
        - 前进距离上限从 150 cm 改为 60 cm，更保守。
        """

        # Step 1: Rotate to face the tag
        if angle_deg > 5.0:
            self.get_logger().info(f"Rotating clockwise by {angle_deg:.1f} deg")
            try:
                self.tello.rotate_clockwise(int(angle_deg))
            except Exception as e:
                self.get_logger().error(f"Rotate clockwise failed: {e}")
        elif angle_deg < -5.0:
            self.get_logger().info(f"Rotating counter-clockwise by {-angle_deg:.1f} deg")
            try:
                self.tello.rotate_counter_clockwise(int(-angle_deg))
            except Exception as e:
                self.get_logger().error(f"Rotate ccw failed: {e}")
        else:
            self.get_logger().info("Yaw angle small, no rotation needed.")

        time.sleep(1.0)

        # Step 2: Move forward along the aligned heading
        # 更保守：最多只飞 60 cm
        forward_cm = int(max(0.0, min(abs(Y), 150.0)))
        if forward_cm > 20:
            self.get_logger().info(
                f"Moving forward {forward_cm} cm toward tag (no lateral correction)."
            )
            self.safe_move("forward", forward_cm)
        else:
            self.get_logger().info("Forward distance small; skipping forward move.")

        # No lateral move on purpose
        self.get_logger().info("Skipping lateral (right/left) move after rotation.")

    # ------------------------------------------------
    # Full mission
    # ------------------------------------------------

    def execute_mission(self):
        """Full mission: takeoff, search, plan, approve, execute, land."""
        if not self.connected:
            self.get_logger().error("Not connected to Tello.")
            return False

        try:
            # Takeoff
            self.get_logger().info("Taking off...")
            self.tello.takeoff()
            time.sleep(3.0)
            self.get_logger().info("Takeoff complete.")

            # 不再额外 up hover_height，直接用 Tello 默认高度

            # Search for marker
            det = self.search_for_marker()
            if det is None:
                self.get_logger().error("Marker not found; landing safely.")
                self.tello.land()
                time.sleep(3.0)
                return False

            cx, cy, w_px, corners = det
            frame = self.get_frame()
            if frame is None:
                self.get_logger().error("Could not grab frame for planning; landing.")
                self.tello.land()
                time.sleep(3.0)
                return False

            h, w = frame.shape[:2]
            img_cx, img_cy = w // 2, h // 2

            dx_px = float(cx - img_cx)
            dy_px = float(cy - img_cy)

            self.get_logger().info(
                f"Pixel offsets: dx={dx_px:.1f}, dy={dy_px:.1f}, tag width={w_px}px"
            )

            if self.show_debug:
                self.draw_debug(frame, cx, cy, corners)

            # Path planning
            angle_deg, X, Y, Z = self.plan_path(dx_px, dy_px, w_px)

            # Show path graph & wait for approval
            self.show_path_graph_and_wait(X, Y)

            # Execute the planned path
            self.execute_path(angle_deg, X, Y)

            # 直接 land，不再手动 down 40
            self.get_logger().info("Landing...")
            try:
                self.tello.land()
                time.sleep(3.0)
            except Exception as e:
                self.get_logger().error(f"Land failed: {e}")
                return False

            self.get_logger().info("Mission complete – landed.")
            return True

        except Exception as e:
            self.get_logger().error(f"Mission failed: {e}")
            self.get_logger().warn("Attempting emergency landing...")
            try:
                if self.tello:
                    self.tello.land()
            except Exception:
                pass
            return False

    # ------------------------------------------------
    # Cleanup
    # ------------------------------------------------

    def cleanup(self):
        """Stop video, close windows, and end connection."""
        self.get_logger().info("Cleaning up Tello connection...")
        if self.tello:
            try:
                self.tello.streamoff()
            except Exception:
                pass
            try:
                self.tello.end()
            except Exception:
                pass

        if self.show_debug:
            cv2.destroyAllWindows()
        plt.close('all')
        self.get_logger().info("Cleanup done.")


def main(args=None):
    rclpy.init(args=args)
    node = TelloArTagMissionNode()

    try:
        if node.connect_tello():
            time.sleep(1.0)
            node.execute_mission()
        else:
            node.get_logger().error("Could not connect to Tello. Exiting.")
    except KeyboardInterrupt:
        # Ctrl+C 时立刻尝试降落
        try:
            if node.tello:
                node.get_logger().info("KeyboardInterrupt! Attempting immediate land...")
                node.tello.land()
        except Exception as e:
            node.get_logger().error(f"Immediate land on Ctrl+C failed: {e}")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


