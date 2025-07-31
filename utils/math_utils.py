import numpy as np
import quaternion


class CircularBuffer:
    """A circular buffer with fixed length and filled with a default value."""

    def __init__(self, length: int):
        self._buffer: np.ndarray | None = None
        self._length = length
        self._num_pushes = 0  # in case of reset or the buffer is not full, this will be less than length

    def append(self, value: float):
        """Append a value to the buffer, if the buffer is full, the oldest value will be removed."""
        if self._buffer is None:
            self._buffer = np.zeros((self._length, ) + tuple(value.shape), dtype=np.float32)
        if self._num_pushes == 0:
            self._buffer[:] = value
        else:
            self._buffer = np.roll(self._buffer, -1, axis=0)
            self._buffer[-1] = value
        self._num_pushes += 1

    @property
    def buffer(self):
        return self._buffer

    def reset(self):
        if self._buffer is None:
            return
        self._buffer[:] = 0.0
        self._num_pushes = 0


def quat_rotate_inverse(q: np.quaternion, v: np.array):
    """q must be numpy-quaternion object in w, x, y, z order
    NOTE: non-batchwise version
    """
    q_inv = q.conjugate()
    return quaternion.rotate_vectors(q_inv, v)


def quat_yaw_component(q: np.quaternion):
    """Get the yaw angle from a quaternion in w, x, y, z order.
    The yaw angle is in radians and in the range of [-pi, pi].
    """
    # Extract the yaw angle from the quaternion
    yaw = np.arctan2(2 * (q.x * q.y + q.w * q.z), q.w**2 - q.x**2 - q.y**2 + q.z**2)
    # Normalize the yaw angle to be in the range of [-pi, pi]
    return warp2pi(yaw)


def warp2pi(angle_rad):
    if angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    elif angle_rad < -np.pi:
        angle_rad += 2 * np.pi
    return angle_rad


def quat_to_rpy(q: np.quaternion):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.asin(2 * (w * y - x * z))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return warp2pi(roll), warp2pi(pitch), warp2pi(yaw)


def transform_global_xy_to_robot_xy(global_xy, robot_xy, yaw):
    robot_x = robot_xy[0]
    robot_y = robot_xy[1]
    assert abs(robot_x) < 100 and abs(robot_y) < 100
    global_x = global_xy[0]
    global_y = global_xy[1]
    target_from_go1_xyz = [global_x - robot_x, global_y - robot_y]
    global_x_in_robot = target_from_go1_xyz[0] * np.cos(yaw) + target_from_go1_xyz[1] * np.sin(yaw)
    global_y_in_robot = -target_from_go1_xyz[0] * np.sin(yaw) + target_from_go1_xyz[1] * np.cos(yaw)
    return np.array([global_x_in_robot, global_y_in_robot])
