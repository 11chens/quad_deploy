import argparse


def get_base_parser(description="Rosmanger example"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

    parser.add_argument(
        "--logdir",
        type=str,
        default="",
        help="Common directory for user's data (absolute path).",
    )

    parser.add_argument(
        "--nodryrun",
        action="store_false",
        dest="dry_run",
        default=True,
        help="Disable motor for testing. If you want to control robot, add --nodryrun.",
    )  # default: True, --nodryrun:False

    parser.add_argument(
        "--nosimrun",
        action="store_false",
        dest="sim_run",
        default=True,
        help="Run in simulation. If you want to deploy onboard, add --nosimrun.",
    )  # default: True, --nosimrun:False

    parser.add_argument(
        "--auto", action="store_true", help="Enable autonomous control, and override the joystick commands."
    )  # default: False, --auto:True

    return parser


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument("--wait_robot", action="store_true", default=True, help="Waiting for robot return lowstate")
    parser.add_argument("--wait_vlm", action="store_true", default=False, help="Waiting for VLM return highstate")
    args = parser.parse_args()
