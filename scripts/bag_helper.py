#!/usr/bin/env python3
import argparse
import os
import re
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


def get_bag_start_time(bag_path):
    try:
        # Check if it's a directory or file
        if os.path.isdir(bag_path):
            # If directory, ros2 bag info works on the directory
            pass

        result = subprocess.run(f"ros2 bag info {bag_path}", shell=True, capture_output=True, text=True)
        # Look for "Start:             Dec  1 2025 23:36:07.123456789 (1764603367.123456789)"
        # Regex to capture the float inside parentheses
        match = re.search(r"Start:.*?\(([\d\.]+)\)", result.stdout)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error getting bag info for {bag_path}: {e}")
    return None


def get_video_start_time(ts_path):
    try:
        with open(ts_path) as f:
            line = f.readline().strip()
            if line:
                parts = line.split(".")
                if len(parts) >= 2:
                    return float(parts[0]) + float(parts[1]) * 1e-9
    except Exception as e:
        print(f"Error reading timestamp file {ts_path}: {e}")
    return None


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


def play_and_visualize(bag_paths, show_raw=False):
    # Ensure bag_paths is a list
    if isinstance(bag_paths, str):
        bag_paths = [bag_paths]

    for bag_path in bag_paths:
        if not os.path.exists(bag_path):
            print(f"Error: Bag path '{bag_path}' does not exist.")
            return

    # Start visualization node in background
    print("Starting visualization node (vlm2ui.py)...")
    # Assuming the workspace is sourced and the script is executable or in python path
    # Adjust the path to vlm2ui.py as needed based on your workspace structure
    vlm2ui_path = "quad_deploy/nodes/sigloma/vlm2ui.py"
    # We try to find the absolute path if possible, or assume running from workspace root
    if not os.path.exists(vlm2ui_path):
        # Try absolute path based on known structure
        vlm2ui_path = "/home/jump/Project/quad_deploy/quad_deploy/nodes/sigloma/vlm2ui.py"

    if not os.path.exists(vlm2ui_path):
        print(f"Error: Could not find vlm2ui.py at {vlm2ui_path}")
        return

    vis_cmd = f"python3 {vlm2ui_path}"
    if show_raw:
        vis_cmd += " --raw"

    vis_proc = run_command(vis_cmd, background=True)

    # Give it a moment to start
    time.sleep(2)

    # Play bags
    play_procs = []
    rosbag_args = []
    video_args = []

    for path in bag_paths:
        if path.endswith(".avi"):
            video_args.append(path)
        else:
            rosbag_args.append(path)

    # Calculate start times for synchronization
    bag_start_time = None
    if rosbag_args:
        # Use the first bag to determine start time
        bag_start_time = get_bag_start_time(rosbag_args[0])
        if bag_start_time:
            print(f"Bag start time: {bag_start_time}")

    video_start_times = {}
    for vid_path in video_args:
        ts_path = vid_path.replace(".avi", ".txt")
        if not os.path.exists(ts_path) and "_img_" in vid_path:
            ts_path = vid_path.replace("_img_", "_ts_").replace(".avi", ".txt")

        if os.path.exists(ts_path):
            t = get_video_start_time(ts_path)
            if t:
                video_start_times[vid_path] = (t, ts_path)
                print(f"Video {vid_path} start time: {t}")

    # Determine global start time
    start_times = []
    if bag_start_time:
        start_times.append(bag_start_time)
    for t, _ in video_start_times.values():
        start_times.append(t)

    min_start_time = min(start_times) if start_times else 0
    if min_start_time > 0:
        print(f"Global start time: {min_start_time}")

    try:
        # Start rosbag play
        if rosbag_args:
            delay = 0.0
            if bag_start_time and min_start_time > 0:
                delay = max(0.0, bag_start_time - min_start_time)

            print(f"Playing bags: {rosbag_args} with delay {delay:.3f}s...")
            cmd = f"sleep {delay} && ros2 bag play {' '.join(rosbag_args)}"
            play_procs.append(run_command(cmd, background=True))

        # Start video play
        for vid_path in video_args:
            ts_path = None
            delay = 0.0

            if vid_path in video_start_times:
                t, ts_path = video_start_times[vid_path]
                if min_start_time > 0:
                    delay = max(0.0, t - min_start_time)
            else:
                # Fallback logic if timestamp file wasn't found or parsed earlier
                ts_path = vid_path.replace(".avi", ".txt")
                if not os.path.exists(ts_path) and "_img_" in vid_path:
                    ts_path = vid_path.replace("_img_", "_ts_").replace(".avi", ".txt")

            if ts_path and os.path.exists(ts_path):
                print(f"Playing video: {vid_path} with delay {delay:.3f}s...")
                # Use absolute path to script if possible
                script_path = os.path.join(os.path.dirname(__file__), "play_video.py")
                if not os.path.exists(script_path):
                    script_path = "quad_deploy/scripts/play_video.py"

                cmd = f"python3 {script_path} {vid_path} {ts_path} --delay {delay}"
                play_procs.append(run_command(cmd, background=True))
            else:
                print(f"Warning: Timestamp file not found for {vid_path}, skipping video.")

        # Wait for all play processes to finish
        for proc in play_procs:
            proc.wait()

        print("All bag playbacks finished.")
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
        for proc in play_procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                pass
    finally:
        print("Stopping visualization node...")
        try:
            os.killpg(os.getpgid(vis_proc.pid), signal.SIGTERM)
        except:
            pass


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
    play_parser.add_argument("bag_paths", nargs="+", help="Paths to the rosbag folders or files (supports multiple)")
    play_parser.add_argument("--raw", action="store_true", help="Show raw image window.")

    args = parser.parse_args()

    if args.command == "record":
        record_bag(args.output, include_rgb=not args.no_rgb)
    elif args.command == "play":
        play_and_visualize(args.bag_paths, show_raw=args.raw)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
