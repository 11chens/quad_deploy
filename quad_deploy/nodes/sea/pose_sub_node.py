"""SEA-Nav robot-pose subscriber.

Subscribes to the robot's 2D pose ``(x, y, theta)`` in the world frame and
exposes it as plain numpy arrays. The Nav agent owns the goal in the world
frame and computes ``goal_local_xy`` itself (using the pose received here),
so this node intentionally does not perform any frame transformation.

Design principles:
    * **Nothing hardcoded** - topic, frame id and timeout are all kwargs.
    * **No transformations** - the incoming ``(x, y, theta)`` is exposed
      verbatim.
    * **Safe-stop friendly** - downstream code uses ``is_fresh(timeout_s)``
      to decide when the pose stream is no longer reliable.
"""

import time

import numpy as np
from geometry_msgs.msg import Pose2D
from ros_base.nodes.base_node import BaseNode


class PoseSubNode(BaseNode):
    """Subscribe to ``geometry_msgs/Pose2D`` carrying the robot's world pose.

    The latest pose is exposed as:
        - ``self.robot_world_xy`` - ``np.ndarray`` shape ``(2,)`` in meters
        - ``self.robot_world_yaw`` - float, radians

    Currently only ``Pose2D`` is supported because (a) the
    ``unitree_mujoco_ros`` plugin publishes ``Pose2D`` and (b) the NavAgent
    needs yaw to do the world → base rotation, which ``PointStamped`` does not
    carry. Add new message types later if needed.
    """

    def __init__(
        self,
        pose_topic: str = "/pose",
        pose_frame_id: str = "world",
        pose_timeout_s: float = 0.5,
        pose_expected_freq_hz: float = 10.0,
        pose_qos_depth: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pose_topic = pose_topic
        self.frame_id = pose_frame_id
        self.timeout_s = float(pose_timeout_s)
        self.expected_freq_hz = float(pose_expected_freq_hz)

        # State buffers - initial pose is (0, 0, 0): treats "we haven't heard
        # anything yet" as "the robot sits at the origin facing +x". The real
        # gate for downstream agents is the `is_fresh()` check, not the value.
        self._robot_world_xy = np.zeros(2, dtype=np.float32)
        self._robot_world_yaw = 0.0
        self._last_msg_time = None  # time.monotonic() at last callback
        self._msg_count = 0

        self._sub = self.create_subscription(Pose2D, self.pose_topic, self._pose2d_callback, pose_qos_depth)

        self.logger.info(
            f"[PoseSubNode] subscribed to '{self.pose_topic}' "
            f"(msg_type=Pose2D, frame_id='{self.frame_id}', "
            f"timeout_s={self.timeout_s}, expected_freq={self.expected_freq_hz} Hz)"
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _pose2d_callback(self, msg: Pose2D):
        # Pose2D has no header; nothing to frame-check at the message level.
        self._update_pose(msg.x, msg.y, msg.theta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_pose(self, x, y, yaw):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(yaw)):
            # NOTE: once-only to stay compatible with both CustomLogger and
            # the plain rclpy logger used in standalone debug mode.
            self.logger.warning(
                f"[PoseSubNode] Received non-finite pose ({x}, {y}, {yaw}) on '{self.pose_topic}', ignoring.",
                once=True,
            )
            return

        self._robot_world_xy[0] = float(x)
        self._robot_world_xy[1] = float(y)
        self._robot_world_yaw = float(yaw)
        self._last_msg_time = time.monotonic()
        self._msg_count += 1

    # ------------------------------------------------------------------
    # Public API for agents / handlers
    # ------------------------------------------------------------------
    @property
    def robot_world_xy(self):
        """Latest robot position in the world frame, shape ``(2,)`` in meters."""
        return self._robot_world_xy

    @property
    def robot_world_yaw(self):
        """Latest robot yaw in the world frame, in radians."""
        return self._robot_world_yaw

    @property
    def msg_count(self):
        """How many pose messages have been received since startup."""
        return self._msg_count

    def time_since_last_msg(self):
        """Seconds since the last received pose message.

        Returns ``+inf`` if no message has ever been received.
        """
        if self._last_msg_time is None:
            return float("inf")
        return time.monotonic() - self._last_msg_time

    def is_fresh(self, timeout_s=None):
        """Whether the latest robot pose is fresh enough to be trusted.

        Args:
            timeout_s: override the default ``pose_timeout_s``. ``None`` falls
                back to the configured timeout.
        """
        if timeout_s is None:
            timeout_s = self.timeout_s
        return self.time_since_last_msg() <= timeout_s


def main():
    """Standalone debug entry: ``python pose_sub_node.py --topic /xxx``."""
    import rclpy
    from ros_base.utils.args_debug import add_debug_mode

    parser = add_debug_mode(listen_port=8882)
    parser.add_argument("--topic", type=str, default="/pose")
    parser.add_argument("--frame_id", type=str, default="world")
    parser.add_argument("--timeout_s", type=float, default=0.5)
    args = parser.parse_args()

    rclpy.init()
    node = PoseSubNode(
        node_name="pose_sub_node",
        pose_topic=args.topic,
        pose_frame_id=args.frame_id,
        pose_timeout_s=args.timeout_s,
    )
    try:
        node.start_spin_standalone()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
