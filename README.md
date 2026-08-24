# Unitree Go2 Deployment

This repository contains the deployment code for **Unitree Go2**.
**Strictly prohibited from external distribution or commercial use.**

# Installation

#### 1. Install ROS2 using [Robostack](https://robostack.github.io/GettingStarted.html#__tabbed_3_2) in a virtual environment

#### 2. Install `ros_base`

```bash
cd ~
git clone git@github.com:11chens/ros_base.git
cd ros_base
pip install -e .
```

#### 3. Install `quad_deploy`

```bash
cd ~/Projects/quad_deploy
pip install -e .
```

#### 4. Configure `pre-commit`

Enable automatic code format checking before each commit:

```bash
pre-commit install
```



#### 5. Install `unitree_sdk2_python`

```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```



#### 6. Install `unitree_mujoco_ros` (simulator)

The MuJoCo simulator used by the default launch config (and required for sim-to-sim) lives in a separate repository. Clone it under `~/Projects` so the launcher can locate it as a sibling of `quad_deploy`:

```bash
cd ~/Projects
git clone git@github.com:11chens/unitree_mujoco_ros.git
cd unitree_mujoco_ros
pip install -e .
```



#### 7. Install `ros2-keyboard` (keyboard control for simulation)

Simulation runs spawn a [`ros2-keyboard`](https://github.com/cmower/ros2-keyboard) window to drive the robot. Build it into a ROS2 workspace once:

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/cmower/ros2-keyboard.git
cd ~/ros2_ws
colcon build
source ~/ros2_ws/install/setup.bash
```



#### 8. Download model weights

The ONNX policy weights are hosted on [Google Drive](https://drive.google.com/drive/folders/1x_RYzCHrujCawauw_7cilkf5EsyE4KCq?usp=drive_link). The shared folder is named `onnx_models` and contains two subfolders — `sigloma` (used by the default **Loco** mode) and `sea_nav` (used by **SEA-Nav** mode). Download it into `~/Data/onboard_data/` so the layout matches the default `--data` paths:

```bash
mkdir -p ~/Data/onboard_data
cd ~/Data/onboard_data
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1x_RYzCHrujCawauw_7cilkf5EsyE4KCq
```

> If `gdown` fails (large-file virus-scan prompt, or the >50-file limit), just download the `onnx_models` folder from the browser and move it to `~/Data/onboard_data/onnx_models`.

Keep each subfolder exactly as downloaded. The key files the deploy scripts load are (other files may also be present):

```text
~/Data/onboard_data/onnx_models/
├── sigloma/                 # default Loco mode (--data default)
│   ├── loco_model/model.onnx
│   └── nav_model/model.onnx
└── sea_nav/                 # SEA-Nav mode (--data default)
    ├── loco_model/model.onnx
    └── nav_model/model.onnx
```



# Usage

The control stack ships **two run modes**, selected by the launch config passed to `quad_launch.py`:


| Mode        | Launch config               | Description                                                                                                                                                                     |
| ----------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Loco**    | `launch_cfg.yaml` (default) | Basic locomotion control, in MuJoCo or on the real robot.                                                                                                                       |
| **SEA-Nav** | `sea_launch_cfg.yaml`       | Autonomous point-goal navigation: a CBF-shielded navigation policy steers the locomotion policy toward a world-frame goal, with a safe-stop fallback when perception drops out. |




#### Prerequisites (both modes)

Activate the ROS2 conda environment and source the keyboard workspace so the in-sim control window works:

```bash
conda activate ros_env
source ~/ros2_ws/install/setup.bash
```

---



## Mode 1: Loco (default)

Run the launch script from the `quad_deploy` directory. With no config argument it defaults to `launch_cfg.yaml`, which starts locomotion control in MuJoCo:

```bash
cd ~/Projects/quad_deploy
python launch/quad_launch.py
```

**For Real Robot Deployment:** disable the simulation nodes and explicitly enable the real control node via command-line arguments (so you don't have to edit the YAML file):

```bash
cd ~/Projects/quad_deploy
python launch/quad_launch.py --disable RL_CONTROL_SIM MUJOCO_SIM --enable RL_CONTROL_REAL
```

*Expected output:*

```text
Launching System from: /home/robot/Projects/quad_deploy/launch/launch_cfg.yaml
Starting Session: [quad_system]
  -> [0] RL_CONTROL started.
  -> [1] MUJOCO_SIM started.

 Systems started in TMUX.
============================================================
 Session: quad_system
   -> Attach: tmux attach -t quad_system
============================================================
```

Attach to the tmux session to view the running nodes (top pane = `quad_deploy` control node, bottom pane = `mujoco` simulator):

```bash
tmux attach -t quad_system
```

**Keyboard controls.** A ROS2 keyboard window pops up; use it to drive the robot:

1. **Initialization:** wait for the robot to fully recover to a standing position.
2. **Start control:** press `X` to enter locomotion control mode.
3. **Movement commands:**
  - **Up / Down arrows**: move forward / backward.
  - **Left / Right arrows**: rotate (yaw).
  - `1` **/** `3`: move left / right (lateral).
  - `5` **/** `2`: pitch up / down.

---



## Mode 2: SEA-Nav (autonomous navigation)

SEA-Nav uses its own launch config. The simulator publishes a 2D lidar scan on `/rays` and the robot world pose on `/pose`; the navigation policy then steers the locomotion policy toward a goal defined in the world frame.

ONNX models are loaded from `~/Data/onboard_data/onnx_models/sea_nav/` (override with `--data`):

```text
~/Data/onboard_data/onnx_models/sea_nav/
├── nav_model/model.onnx
└── loco_model/model.onnx
```

**Simulation:**

```bash
cd ~/Projects/quad_deploy
python launch/quad_launch.py launch/sea_launch_cfg.yaml
tmux attach -t sea_system
```

The default `--seed 42` already matches the navigation goal. If you change the seed, set `--goal_x` / `--goal_y` on `SEA_CONTROL_SIM` in `launch/sea_launch_cfg.yaml` to the `[Scene] goal=(x, y)` printed by the simulator.

**Keyboard controls (state machine).** Drive the robot through the SEA-Nav FSM from the keyboard window:


| Key                        | Transition                                                              |
| -------------------------- | ----------------------------------------------------------------------- |
| `X`                        | after standing → human teleop                                           |
| `E`                        | human teleop → autonomous navigation (requires fresh `/rays` + `/pose`) |
| `D`                        | navigation / safe-stop → human teleop (manual takeover)                 |
| `A`                        | any state → emergency (motors off)                                      |
| `Q`                        | emergency → recovery                                                    |
| Arrows / `1` `3` / `5` `2` | manual teleop: forward·backward / yaw / lateral / pitch                 |


During `navigation` the policy drives automatically; if `/rays` or `/pose` go stale for more than 1 s the robot enters `safe_stop` and holds position until perception recovers and you press `E` again.

# Notes

- **Branching policy**: All development must be done on new branches.
**Direct merges into master branch are strictly prohibited without approval**.
- **Code quality**: Pre-commit will enforce code style checks before commits.
- **Confidentiality**: Internal use only. External distribution or commercial use is forbidden.



# Troubleshooting

When installing `unitree_sdk2_python`, you may see error when `pip install -e .`:

```bash
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

**Cause**: The system could not find the path to `cyclonedds`.

**Solution**: Compile and install `cyclonedds` first.

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```

After installation, enter the `unitree_sdk2_python` directory, set the environment variable, and reinstall:

```bash
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME=~/cyclonedds/install
pip install -e .
```

