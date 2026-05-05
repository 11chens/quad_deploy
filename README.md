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

# Usage

To run the system, follow these steps:

#### 1. Activate the Environment {Optional}
Activate your ROS2 conda environment:
```bash
conda activate ros_env
```

#### 2. Launch the System
Run the launch script from the `quad_deploy` directory. By default, this will run in MuJoCo Simulation.

**For MuJoCo Simulation (Default):**
```bash
cd ~/Projects/quad_deploy
python launch/quad_launch.py
```

**For Real Robot Deployment:**
To run on the real robot, you can disable the simulation nodes and explicitly enable the real control node using command-line arguments (so you don't have to modify the YAML file):
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

#### 3. Monitor the TMUX Session
Attach to the created tmux session to view the running nodes:
```bash
tmux attach -t quad_system
```
Inside the tmux session, the **top pane** displays the `quad_deploy` control node, and the **bottom pane** displays the `mujoco` simulation node.

#### 4. Keyboard Controls
A ROS2 keyboard control window will pop up. Use it to control the robot:
1. **Initialization:** Wait for the robot to fully recover to a standing position.
2. **Start RL:** Press **`X`** to enter Reinforcement Learning (RL) control mode.
3. **Movement Commands:**
   - **Up / Down arrows**: Move forward / backward.
   - **Left / Right arrows**: Rotate (Yaw angle).
   - **`1` / `3`**: Move left / right (Lateral).
   - **`5` / `2`**: Pitch up / down.

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
