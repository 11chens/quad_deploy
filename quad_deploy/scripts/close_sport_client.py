import time

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient


def close_sport_client():
    """Close Unitree sport client, prepare for RL control"""
    sc = SportClient()
    sc.SetTimeout(5.0)
    sc.Init()

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    status, result = msc.CheckMode()
    while result["name"]:
        sc.StandDown()
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1)

    return True


if __name__ == "__main__":
    ChannelFactoryInitialize(0, "eth0")
    close_sport_client()
