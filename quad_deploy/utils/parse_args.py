import argparse


def parse_arguments(custom_parameters=[], description="Rosmanger example"):
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
        action="store_true",
        default=False,
        help="Disable motor for testing. If you want to control robot, add --nosimrun.",
    )  # default: False, --nodryrun:True

    parser.add_argument(
        "--nosimrun",
        action="store_true",
        default=False,
        help="Run in simulation. If you want to deploy onboard, add --nosimrun.",
    )  # default: False, --nosimrun:True

    parser.add_argument(
        "--auto", action="store_true", help="Enable autonomous control, and override the joystick commands."
    )  # default: False, --auto:True

    parser.add_argument(
        "--wait", action="store_true", help="Handshake and wait for other nodes."
    )  # default: False, --wait:True

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(
                        argument["name"], type=argument["type"], default=argument["default"], help=help_str
                    )
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(
                    argument["name"], action=argument["action"], default=argument["default"], help=help_str
                )

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    custom_parameters = [
        {"name": "--wait_robot", "action": "store_true", "default": True, "help": "Waiting for robot return lowstate"},
        {"name": "--wait_vlm", "action": "store_true", "default": False, "help": "Waiting for VLM return highstate"},
    ]
    args = parse_arguments(custom_parameters)
