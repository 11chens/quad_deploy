import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image


class ZedPublisher:
    def __init__(self, ros_manager=None):
        self.img_pub = ros_manager.create_publisher(Image, "/geometry_msgs/Image", 10)
        self.P_img_pub = ros_manager.create_publisher(Point, "/geometry_msgs/p_img", 10)

        freq_hz = 10
        self.timer = ros_manager.create_timer(1 / freq_hz, self._timer_callback)
        self.img_msg = Image()
        self.P_img_msg = Point()
        # TODO: camera config, setup Cutie

    def _timer_callback(self):
        # TODO: get img from camera, Cutie process, and publish
        # TODO: get point from img, compute P_img(u,v,depth), and publish
        self.img_pub.publish(self.img_msg)
        self.P_img_pub.publish(self.P_img_msg)
