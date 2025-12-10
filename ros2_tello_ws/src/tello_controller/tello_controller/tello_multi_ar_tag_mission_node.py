#!/usr/bin/env python3

import time
import math

import rclpy
from rclpy.node import Node
from djitellopy import Tello
import cv2
import numpy as np
import matplotlib.pyplot as plt


class TelloMultiArTagMissionNode(Node):
    """
    ROS2 Node for multi-ArUco-tag-guided Tello mission.

    Pipeline:
      1. Connect and takeoff
      2. Search and detect ALL visible ArUco tags
      3. For each tag:
           - Compute angle, distance, and lateral offset
           - Show planned path
      4. Allow user to select which tag to visit
      5. Execute path to selected tag
      6. Optionally visit next tag or land
    """

    def __init__(self):
        super().__init__("tello_multi_ar_tag_mission_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("hover_height", 70)
        self.declare_parameter("marker_ids", [0, 1, 2, 3])  # List of ArUco IDs to track
        self.declare_parameter("tag_size_cm", 15.0)
        self.declare_parameter("search_timeout", 40.0)
        self.declare_parameter("show_debug_window", True)
        self.declare_parameter("visit_all_tags", False)  # If True, visit all detected tags

        self.hover_height = int(self.get_parameter("hover_height").value)
        self.marker_ids = self.get_parameter("marker_ids").value
        self.tag_size_cm = float(self.get_parameter("tag_size_cm").value)
        self.search_timeout = float(self.get_parameter("search_timeout").value)
        self.show_debug = bool(self.get_parameter("show_debug_window").value)
        self.visit_all_tags = bool(self.get_parameter("visit_all_tags").value)

        self.hover_height = max(40, min(self.hover_height, 150))
        self.focal_length_px = 920.0

        self.get_logger().info(
            f"Tello Multi-AR Tag Mission Node initialized\n"
            f"  Hover height: {self.hover_height} cm\n"
            f"  Marker IDs: {self.marker_ids}\n"
            f"  Tag size: {self.tag_size_cm} cm\n"
            f"  Search timeout: {self.search_timeout} s\n"
            f"  Visit all tags: {self.visit_all_tags}"
        )

        # Tello & video
        self.tello = None
        self.frame_read = None
        self.connected = False

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Track visited tags
        self.visited_tags = set()

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

            self.get_logger().info("Starting video stream...")
            self.tello.streamon()
            self.frame_read = self.tello.get_frame_read()
            time.sleep(1.0)

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
    # ArUco detection (Multi-tag)
    # ------------------------------------------------

    def detect_all_markers(self, frame):
        """
        Detect ALL ArUco markers in the configured list.

        Returns:
            List of dicts: [
                {
                    'id': marker_id,
                    'cx': center_x,
                    'cy': center_y,
                    'width_px': tag_width_px,
                    'corners': marker_corners
                },
                ...
            ]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        detected = []

        if ids is None or len(ids) == 0:
            return detected

        ids = ids.flatten()

        for i, mid in enumerate(ids):
            if mid in self.marker_ids:
                marker_corners = corners[i][0]  # shape (4,2)

                # Center
                c_x = int(np.mean(marker_corners[:, 0]))
                c_y = int(np.mean(marker_corners[:, 1]))

                # Approximate pixel width
                w_px = int(np.linalg.norm(marker_corners[0] - marker_corners[1]))

                detected.append({
                    'id': int(mid),
                    'cx': c_x,
                    'cy': c_y,
                    'width_px': w_px,
                    'corners': marker_corners
                })

        return detected

    def draw_multi_debug(self, frame, detections):
        """Draw detection overlays for all detected tags."""
        if frame is None:
            return

        h, w = frame.shape[:2]
        cx_img, cy_img = w // 2, h // 2

        # Image center
        cv2.circle(frame, (cx_img, cy_img), 4, (255, 0, 0), -1)

        # Color map for different tags
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 165, 255),  # Orange
        ]

        for idx, det in enumerate(detections):
            color = colors[idx % len(colors)]
            
            # Marker outline
            cv2.polylines(
                frame,
                [det['corners'].astype(np.int32)],
                True,
                color,
                2
            )

            # Marker center
            cx, cy = det['cx'], det['cy']
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.line(frame, (cx_img, cy_img), (cx, cy), color, 2)

            # Label with ID
            visited_str = " (visited)" if det['id'] in self.visited_tags else ""
            label = f"ID:{det['id']}{visited_str}"
            cv2.putText(
                frame,
                label,
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Title
        cv2.putText(
            frame,
            f'Multi-AR Tag - {len(detections)} detected',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow('Tello Multi AR Tag', frame)
        cv2.waitKey(1)

    # ------------------------------------------------
    # Movement helpers
    # ------------------------------------------------

    def safe_move(self, command: str, distance_cm: int):
        """Wrapper for move commands with logging and small delays."""
        distance_cm = int(max(20, min(distance_cm, 200)))
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
            time.sleep(1.2)
        except Exception as e:
            self.get_logger().error(f"Move {command} failed: {e}")

    # ------------------------------------------------
    # Search for markers
    # ------------------------------------------------

    def search_for_markers(self):
        """
        Rotate and search until at least one marker is found or timeout.

        Returns:
            List of detections or None
        """
        self.get_logger().info("Searching for markers by rotating...")
        start_time = time.time()
        last_rotate = 0.0
        rotate_interval = 2.0

        while time.time() - start_time < self.search_timeout:
            frame = self.get_frame()
            if frame is None:
                continue

            detections = self.detect_all_markers(frame)
            
            if len(detections) > 0:
                self.get_logger().info(
                    f"Found {len(detections)} marker(s): "
                    f"{[d['id'] for d in detections]}"
                )
                if self.show_debug:
                    self.draw_multi_debug(frame, detections)
                return detections

            # No markers yet; rotate
            if time.time() - last_rotate > rotate_interval:
                self.get_logger().info("Rotating clockwise to search...")
                try:
                    self.tello.rotate_clockwise(20)
                except Exception as e:
                    self.get_logger().error(f"Rotate failed: {e}")
                last_rotate = time.time()

            if self.show_debug:
                self.draw_multi_debug(frame, [])

        self.get_logger().warn("Search timeout reached – no markers found.")
        return None

    # ------------------------------------------------
    # Path planning for multiple tags
    # ------------------------------------------------

    def plan_path_for_tag(self, detection, frame_shape):
        """
        Compute angle and distance to a specific tag.

        Returns:
            dict with keys: angle_deg, X, Y, Z
        """
        h, w = frame_shape[:2]
        img_cx, img_cy = w // 2, h // 2

        dx_px = float(detection['cx'] - img_cx)
        dy_px = float(detection['cy'] - img_cy)
        w_px = detection['width_px']

        f = self.focal_length_px
        W = self.tag_size_cm

        # Estimate range
        if w_px <= 0:
            Z = 50.0
        else:
            Z = (W * f) / float(w_px)

        # Horizontal angle
        angle_rad = math.atan2(dx_px, f)
        angle_deg = angle_rad * 180.0 / math.pi

        # Planar offsets
        X = Z * math.sin(angle_rad)
        Y = Z * math.cos(angle_rad)

        return {
            'angle_deg': angle_deg,
            'X': X,
            'Y': Y,
            'Z': Z
        }

    def show_multi_path_graph(self, tag_paths):
        """
        Show a 2D graph of planned paths to all detected tags.
        
        tag_paths: list of dicts with keys: id, angle_deg, X, Y, Z
        """
        self.get_logger().info("Showing planned paths to all detected tags...")

        plt.figure(figsize=(8, 8))
        
        colors = ['g', 'b', 'r', 'm', 'c', 'y']
        
        for idx, path in enumerate(tag_paths):
            color = colors[idx % len(colors)]
            marker_id = path['id']
            X, Y = path['X'], path['Y']
            
            visited_marker = '*' if marker_id in self.visited_tags else 'o'
            
            plt.plot([0, X], [0, Y], f"{color}-{visited_marker}", 
                    label=f"Tag ID {marker_id}")
            plt.text(X, Y, f" Tag {marker_id}", fontsize=10)

        plt.scatter([0], [0], c='black', s=100, marker='D', label="Drone")
        plt.title("Planned Paths to All Detected Tags")
        plt.xlabel("Right (+) cm")
        plt.ylabel("Forward (+) cm")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()

        plt.show(block=False)
        plt.pause(0.1)

    def select_target_tag(self, tag_paths):
        """
        Let user select which tag to visit.
        
        Returns:
            Selected tag dict or None
        """
        print("\n" + "="*50)
        print("DETECTED TAGS:")
        for idx, path in enumerate(tag_paths):
            visited = " (VISITED)" if path['id'] in self.visited_tags else ""
            print(f"  [{idx}] Tag ID {path['id']}: "
                  f"Angle={path['angle_deg']:.1f}°, "
                  f"Distance={path['Z']:.1f}cm{visited}")
        print("="*50)
        
        while True:
            choice = input("\nEnter tag number to visit (or 'q' to land): ").strip()
            
            if choice.lower() == 'q':
                return None
            
            try:
                idx = int(choice)
                if 0 <= idx < len(tag_paths):
                    return tag_paths[idx]
                else:
                    print(f"Invalid choice. Enter 0-{len(tag_paths)-1}")
            except ValueError:
                print("Invalid input. Enter a number or 'q'")

    # ------------------------------------------------
    # Execute path to specific tag
    # ------------------------------------------------

    def execute_path_to_tag(self, path):
        """
        Navigate to a specific tag.
        
        path: dict with keys: angle_deg, X, Y, Z
        """
        angle_deg = path['angle_deg']
        X = path['X']
        Y = path['Y']

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

        # Step 2: Move forward
        forward_cm = int(max(0.0, min(abs(Y), 150.0)))
        if forward_cm > 20:
            self.get_logger().info(f"Moving forward {forward_cm} cm toward tag.")
            self.safe_move("forward", forward_cm)
        else:
            self.get_logger().info("Forward distance small; skipping forward move.")

        self.get_logger().info(f"Arrived at tag ID {path['id']}")
        self.visited_tags.add(path['id'])

    # ------------------------------------------------
    # Full mission
    # ------------------------------------------------

    def execute_mission(self):
        """Full mission: takeoff, search, plan, select, execute."""
        if not self.connected:
            self.get_logger().error("Not connected to Tello.")
            return False

        try:
            # Takeoff
            self.get_logger().info("Taking off...")
            self.tello.takeoff()
            time.sleep(3.0)
            self.get_logger().info("Takeoff complete.")

            # Main loop: search and visit tags
            while True:
                # Search for markers
                detections = self.search_for_markers()
                if detections is None or len(detections) == 0:
                    self.get_logger().warn("No markers found; landing.")
                    break

                # Get current frame for planning
                frame = self.get_frame()
                if frame is None:
                    self.get_logger().error("Could not grab frame; landing.")
                    break

                # Plan paths to all detected tags
                tag_paths = []
                for det in detections:
                    path = self.plan_path_for_tag(det, frame.shape)
                    path['id'] = det['id']
                    tag_paths.append(path)
                    
                    self.get_logger().info(
                        f"Tag ID {det['id']}: "
                        f"angle={path['angle_deg']:.1f}°, "
                        f"X={path['X']:.1f}cm, "
                        f"Y={path['Y']:.1f}cm, "
                        f"Z={path['Z']:.1f}cm"
                    )

                # Show graph
                self.show_multi_path_graph(tag_paths)

                # Select target
                if self.visit_all_tags:
                    # Visit unvisited tags automatically
                    unvisited = [p for p in tag_paths if p['id'] not in self.visited_tags]
                    if len(unvisited) == 0:
                        self.get_logger().info("All tags visited!")
                        break
                    selected = unvisited[0]
                    print(f"\n[AUTO] Visiting Tag ID {selected['id']}")
                else:
                    # Manual selection
                    selected = self.select_target_tag(tag_paths)
                    if selected is None:
                        self.get_logger().info("User chose to land.")
                        break

                # Execute path to selected tag
                self.execute_path_to_tag(selected)

                # Check if should continue
                if not self.visit_all_tags:
                    cont = input("\nVisit another tag? (y/n): ").strip().lower()
                    if cont != 'y':
                        break

            # Land
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
    node = TelloMultiArTagMissionNode()

    try:
        if node.connect_tello():
            time.sleep(1.0)
            node.execute_mission()
        else:
            node.get_logger().error("Could not connect to Tello. Exiting.")
    except KeyboardInterrupt:
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