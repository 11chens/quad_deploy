"""SEA-Nav ray-distance subscriber.

Subscribes to a 2D lidar-style range scan in the robot's base frame and
exposes it as a plain numpy array for downstream agents.

Design principles:
    * **Nothing hardcoded** - topic name, message type, expected number of
      rays, range, frame id and timeout are all kwargs, so the same node
      works with the MuJoCo plugin, an RPLIDAR driver, or any other lidar
      publisher.
    * **No transformations** - raw distances (meters) are exposed as-is. The
      Nav agent applies the ``log2`` and clip itself so the input distribution
      stays consistent with the policy's training observations.
    * **Safe-stop friendly** - the node never raises on stale or malformed
      messages; downstream code uses ``is_fresh(timeout_s)`` and
      ``time_since_last_msg()`` to decide when to fall back to safe-stop.
"""

import time

import numpy as np
from ros_base.nodes.base_node import BaseNode
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


class RaysSubNode(BaseNode):
    """Subscribe to 2D ray distance data published in the base frame.

    Supported message types (selected via ``rays_msg_type``):
        - ``"LaserScan"``   (``sensor_msgs/LaserScan``) - default, carries
          ``angle_min``/``angle_max``/``angle_increment`` for sanity checking.
        - ``"Float32MultiArray"`` (``std_msgs/Float32MultiArray``) - a plain
          float array fallback when the publisher cannot easily produce
          ``LaserScan`` (e.g. a mock publisher).

    The latest raw distances (meters) are exposed via ``self.rays`` and
    ``self.clipped_rays()``. ``log2`` encoding is **not** applied here - the
    Nav agent does it to match training.
    """

    SUPPORTED_MSG_TYPES = ("LaserScan", "Float32MultiArray")

    def __init__(
        self,
        rays_topic: str = "/rays",
        rays_msg_type: str = "LaserScan",
        rays_num_rays: int = 41,
        rays_range_min: float = 0.1,
        rays_range_max: float = 5.0,
        rays_frame_id: str = "base_link",
        rays_timeout_s: float = 0.5,
        rays_expected_freq_hz: float = 10.0,
        rays_qos_depth: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if rays_msg_type not in self.SUPPORTED_MSG_TYPES:
            raise ValueError(
                f"[RaysSubNode] Unsupported rays_msg_type='{rays_msg_type}'. Supported: {self.SUPPORTED_MSG_TYPES}"
            )

        self.rays_topic = rays_topic
        self.rays_msg_type = rays_msg_type
        self.num_rays = int(rays_num_rays)
        self.range_min = float(rays_range_min)
        self.range_max = float(rays_range_max)
        self.frame_id = rays_frame_id
        self.timeout_s = float(rays_timeout_s)
        self.expected_freq_hz = float(rays_expected_freq_hz)

        # State buffers - initialized to range_max so a freshly created node
        # returns a "safe / far away" observation if the agent reads before any
        # message arrives. The freshness check (`is_fresh`) is the real gate.
        self._rays = np.full(self.num_rays, self.range_max, dtype=np.float32)
        self._last_msg_time = None  # time.monotonic() at last callback
        self._last_header_stamp = None  # ROS stamp from msg.header (LaserScan only)
        self._msg_count = 0

        # Subscribe based on the configured message type
        if rays_msg_type == "LaserScan":
            self._sub = self.create_subscription(LaserScan, self.rays_topic, self._laserscan_callback, rays_qos_depth)
        else:  # "Float32MultiArray"
            self._sub = self.create_subscription(
                Float32MultiArray, self.rays_topic, self._float_array_callback, rays_qos_depth
            )

        self.logger.info(
            f"[RaysSubNode] subscribed to '{self.rays_topic}' "
            f"(msg_type={rays_msg_type}, num_rays={self.num_rays}, "
            f"range=[{self.range_min}, {self.range_max}] m, frame_id='{self.frame_id}', "
            f"timeout_s={self.timeout_s}, expected_freq={self.expected_freq_hz} Hz)"
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _laserscan_callback(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)

        # frame_id sanity (warn-once if mismatched, but still accept the data)
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            self.logger.warning(
                f"[RaysSubNode] frame_id mismatch on '{self.rays_topic}': "
                f"expected '{self.frame_id}', got '{msg.header.frame_id}'. "
                "Data still accepted.",
                once=True,
            )

        self._last_header_stamp = msg.header.stamp
        self._update_rays(ranges)

    def _float_array_callback(self, msg: Float32MultiArray):
        ranges = np.asarray(msg.data, dtype=np.float32)
        self._update_rays(ranges)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_rays(self, ranges: np.ndarray):
        # Dimension reconciliation: never crash, always end up with self.num_rays
        if ranges.shape[0] != self.num_rays:
            self.logger.warning(
                f"[RaysSubNode] Received {ranges.shape[0]} rays on '{self.rays_topic}', "
                f"expected {self.num_rays}. Will pad with range_max or truncate.",
                once=True,
            )
            if ranges.shape[0] < self.num_rays:
                pad = np.full(self.num_rays - ranges.shape[0], self.range_max, dtype=np.float32)
                ranges = np.concatenate([ranges, pad])
            else:
                ranges = ranges[: self.num_rays]

        # Sanitize non-finite values (NaN / +-Inf). LaserScan publishers may use
        # +inf to mean "no return" - we clamp these to range_max so the policy
        # sees a finite "far away" reading.
        if not np.all(np.isfinite(ranges)):
            ranges = np.where(np.isfinite(ranges), ranges, self.range_max)
            # NOTE: once-only to stay compatible with both CustomLogger and the
            # plain rclpy logger used in standalone debug mode.
            self.logger.warning(
                f"[RaysSubNode] Received non-finite ray values on '{self.rays_topic}', replaced with range_max.",
                once=True,
            )

        self._rays = ranges.astype(np.float32, copy=False)
        self._last_msg_time = time.monotonic()
        self._msg_count += 1

    # ------------------------------------------------------------------
    # Public API for agents / handlers
    # ------------------------------------------------------------------
    @property
    def rays(self):
        """Latest raw ray distances in meters, shape ``(num_rays,)``.

        Raw values: not clipped, not log2-encoded. The Nav agent applies
        ``log2 + clip`` itself before assembling the observation.
        """
        return self._rays

    @property
    def msg_count(self):
        """How many rays messages have been received since startup."""
        return self._msg_count

    def time_since_last_msg(self):
        """Seconds since the last received rays message.

        Returns ``+inf`` if no message has ever been received.
        """
        if self._last_msg_time is None:
            return float("inf")
        return time.monotonic() - self._last_msg_time

    def is_fresh(self, timeout_s=None):
        """Whether the latest rays are fresh enough to be trusted.

        Args:
            timeout_s: override the default ``rays_timeout_s``. ``None`` falls
                back to the configured timeout.
        """
        if timeout_s is None:
            timeout_s = self.timeout_s
        return self.time_since_last_msg() <= timeout_s

    def clipped_rays(self):
        """Convenience: rays clipped to ``[range_min, range_max]`` in meters."""
        return np.clip(self._rays, self.range_min, self.range_max)


def main():
    """Standalone debug entry: ``python rays_sub_node.py --topic /xxx``."""
    import rclpy
    from ros_base.utils.args_debug import add_debug_mode

    parser = add_debug_mode(listen_port=8881)
    parser.add_argument("--topic", type=str, default="/rays")
    parser.add_argument("--msg_type", type=str, default="LaserScan", choices=RaysSubNode.SUPPORTED_MSG_TYPES)
    parser.add_argument("--num_rays", type=int, default=41)
    parser.add_argument("--range_min", type=float, default=0.1)
    parser.add_argument("--range_max", type=float, default=5.0)
    parser.add_argument("--frame_id", type=str, default="base_link")
    parser.add_argument("--timeout_s", type=float, default=0.5)
    args = parser.parse_args()

    rclpy.init()
    node = RaysSubNode(
        node_name="rays_sub_node",
        rays_topic=args.topic,
        rays_msg_type=args.msg_type,
        rays_num_rays=args.num_rays,
        rays_range_min=args.range_min,
        rays_range_max=args.range_max,
        rays_frame_id=args.frame_id,
        rays_timeout_s=args.timeout_s,
    )
    try:
        node.start_spin_standalone()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
