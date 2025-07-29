from setuptools import find_packages, setup

setup(
    name="quad_deploy",
    version="0.0.1",
    author="Shiyi Chen",
    author_email="",
    license="",
    packages=find_packages(),
    description="Unitree Onboard code supporting Unitree Go2 robot and auto-loading configurations",
    python_requires=">=3.6",
    install_requires=[
        "onnxruntime", "numpy", "numpy-quaternion", "debugpy", "mujoco", "pygame"
        # "rclpy",
    ],
)
