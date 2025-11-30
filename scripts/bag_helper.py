#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time


def run_command(cmd, background=False):
    print(f"Running: {cmd}")
    if background:
        return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    else:
        return subprocess.run(cmd, shell=True)


def record_bag(output_dir, include_rgb=True):
    topics = ["/geometry_msgs/p_img_filtered", "/geometry_msgs/p_img"]

    if include_rgb:
        topics.insert(0, "/camera/color/image_raw")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bag_name = f"rosbag2_{timestamp}"
    full_path = os.path.join(output_dir, bag_name)

    cmd = f"ros2 bag record -o {full_path} {' '.join(topics)}"

    print(f"Starting recording to {full_path}...")
    print("Press Ctrl+C to stop recording.")

    try:
        proc = run_command(cmd, background=False)
    except KeyboardInterrupt:
        print("\nRecording stopped.")


def play_and_visualize(bag_path, show_raw=False):
    if not os.path.exists(bag_path):
        print(f"Error: Bag path '{bag_path}' does not exist.")
        return

    # Start visualization node in background
    print("Starting visualization node (vlm2ui.py)...")
    # Assuming the workspace is sourced and the script is executable or in python path
    # Adjust the path to vlm2ui.py as needed based on your workspace structure
    vlm2ui_path = "quad_deploy/nodes/homi/vlm2ui.py"
    # We try to find the absolute path if possible, or assume running from workspace root
    if not os.path.exists(vlm2ui_path):
        # Try absolute path based on known structure
        vlm2ui_path = "/home/jump/Project/quad_deploy/quad_deploy/nodes/homi/vlm2ui.py"

    if not os.path.exists(vlm2ui_path):
        print(f"Error: Could not find vlm2ui.py at {vlm2ui_path}")
        return

    vis_cmd = f"python3 {vlm2ui_path}"
    if show_raw:
        vis_cmd += " --raw"

    vis_proc = run_command(vis_cmd, background=True)

    # Give it a moment to start
    time.sleep(2)

    # Play bag
    print(f"Playing bag: {bag_path}...")
    play_cmd = f"ros2 bag play {bag_path}"

    try:
        run_command(play_cmd, background=False)
        print("Bag playback finished.")
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
    finally:
        print("Stopping visualization node...")
        os.killpg(os.getpgid(vis_proc.pid), signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description="Helper script to record or play+visualize ROS2 bags for VLM UI.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Record command
    record_parser = subparsers.add_parser("record", help="Record specific topics to a rosbag")
    record_parser.add_argument(
        "--output",
        "-o",
        default="/home/unitree/Data/rosbags",
        help="Output directory for the bag (default: current dir)",
    )
    record_parser.add_argument("--no-rgb", action="store_true", help="Do not record RGB image topic")

    # Play command
    play_parser = subparsers.add_parser("play", help="Play a rosbag and run visualization")
    play_parser.add_argument("bag_path", help="Path to the rosbag folder or file")
    play_parser.add_argument("--raw", action="store_true", help="Show raw image window.")

    args = parser.parse_args()

    if args.command == "record":
        record_bag(args.output, include_rgb=not args.no_rgb)
    elif args.command == "play":
        play_and_visualize(args.bag_path, show_raw=args.raw)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
