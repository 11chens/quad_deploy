from keyboard_msgs.msg import Key


class KeyboardSubscriber:
    def __init__(self, ros_manager=None):
        self._init_keys()
        self.keydown_sub = ros_manager.create_subscription(Key, "/keydown", self._keydown_callback, 1)

    def _init_keys(self):
        self.S = False
        self.X = False
        self.R1 = False
        self.R2 = False
        self.L1 = False
        self.L2 = False

        self.up = False
        self.down = False
        self.right = False
        self.left = False

        self.cmd_vx = 0
        self.cmd_vy = 0
        self.cmd_vyaw = 0

    def _keydown_callback(self, msg: Key):
        self.S = msg.code == 115  # S
        self.X = msg.code == 120  # X
        self.R1 = msg.code == 113  # Q
        self.R2 = msg.code == 101  # E
        self.L1 = msg.code == 97  # A
        self.L2 = msg.code == 100  # D

        self.up = msg.code == 273
        self.down = msg.code == 274
        self.right = msg.code == 275
        self.left = msg.code == 276

        self.cmd_vx += float(self.up)
        self.cmd_vx -= float(self.down)
        self.cmd_vyaw += float(self.left)
        self.cmd_vyaw -= float(self.right)
        self.cmd_vx = max(min(self.cmd_vx, 1.0), -1)
        self.cmd_vyaw = max(min(self.cmd_vyaw, 1.0), -1)

    def reset(self):
        self._init_keys()
