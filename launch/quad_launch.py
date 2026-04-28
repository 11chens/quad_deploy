import os

from ros_base.launch.base_launcher import BaseLauncher
from ros_base.utils.args_debug import add_debug_mode

if __name__ == "__main__":
    parser = add_debug_mode(listen_port=9999)
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launch_cfg.yaml")
    parser.add_argument("config", nargs="?", default=default_cfg, help="Path to YAML config file")
    args = parser.parse_args()

    launcher = BaseLauncher(args.config)
    launcher.launch()
