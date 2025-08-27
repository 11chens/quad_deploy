# Unitree Go2 Deployment

This repository contains the deployment code for **Unitree Go2**.
**Strictly prohibited from external distribution or commercial use.**

# Installation

#### 1. Install ROS2 using [Robostack](https://robostack.github.io/GettingStarted.html#__tabbed_3_2) in a virtual environment


#### 2. Install `quad_deploy`
```bash
cd quad_deploy
pip install -e .
```

#### 3. Install `unitree_sdk2_python`
```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```
#### 4. Configure `pre-commit`

Enable automatic code format checking before each commit:
```bash
pre-commit install
```

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
