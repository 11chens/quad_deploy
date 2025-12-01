import argparse
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class VideoPublisher(Node):
    def __init__(self, video_path, ts_path, view=False):
        super().__init__("video_publisher")
        self.view = view
        self.pub = self.create_publisher(Image, "/camera/color/image_raw", 10)

        if not os.path.exists(video_path):
            self.get_logger().error(f"Video file not found: {video_path}")
            sys.exit(1)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file: {video_path}")
            sys.exit(1)

        self.timestamps = []
        try:
            with open(ts_path) as f:
                for line in f:
                    parts = line.strip().split(".")
                    if len(parts) >= 2:
                        self.timestamps.append((int(parts[0]), int(parts[1])))
        except Exception as e:
            self.get_logger().error(f"Failed to read timestamps: {e}")
            sys.exit(1)

        if not self.timestamps:
            self.get_logger().error("No timestamps found")
            sys.exit(1)

        self.i = 0
        self.t0_sys = time.time()
        # Calculate start time from first timestamp
        self.t0_rec = self.timestamps[0][0] + self.timestamps[0][1] * 1e-9

        self.get_logger().info(f"Starting video playback: {video_path}")
        self.get_logger().info(f"Loaded {len(self.timestamps)} timestamps.")
        self.timer = self.create_timer(0.001, self.timer_callback)

    def timer_callback(self):
        if self.i >= len(self.timestamps):
            self.get_logger().info("Video playback finished")
            sys.exit(0)

        # Target time for current frame
        t_rec = self.timestamps[self.i][0] + self.timestamps[self.i][1] * 1e-9
        t_target = t_rec - self.t0_rec

        t_elapsed = time.time() - self.t0_sys

        if t_elapsed >= t_target:
            ret, frame = self.cap.read()
            if ret:
                msg = Image()
                msg.header.stamp.sec = self.timestamps[self.i][0]
                msg.header.stamp.nanosec = self.timestamps[self.i][1]
                msg.header.frame_id = "camera_frame"
                msg.height = frame.shape[0]
                msg.width = frame.shape[1]
                msg.encoding = "bgr8"
                msg.is_bigendian = 0
                msg.step = frame.shape[1] * 3
                msg.data = frame.tobytes()
                self.pub.publish(msg)

                if self.view:
                    cv2.imshow("Video Playback", frame)
                    cv2.waitKey(1)

                if self.i % 30 == 0:
                    self.get_logger().info(f"Published frame {self.i}/{len(self.timestamps)}")

                self.i += 1
            else:
                self.get_logger().info(f"Video stream ended unexpectedly at frame {self.i}")
                sys.exit(0)


def main():
    rclpy.init()
    parser = argparse.ArgumentParser(description="Publish video frames to ROS topic.")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("ts_path", help="Path to the timestamp file")
    parser.add_argument("--view", action="store_true", help="Show video in a window locally")
    parser.add_argument("--delay", type=float, default=0.0, help="Start delay in seconds")
    args = parser.parse_args()

    if args.delay > 0:
        print(f"Waiting {args.delay:.3f} seconds before starting playback...")
        time.sleep(args.delay)

    node = VideoPublisher(args.video_path, args.ts_path, view=args.view)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
