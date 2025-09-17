import ctypes
import os
import platform
import struct

import cyclonedds.idl as idl
from unitree_go.msg import LowCmd
from unitree_sdk2py.utils.singleton import Singleton


class CRC(Singleton):
    def __init__(self):
        # 4 bytes aligned, little-endian format.
        # size 812
        self.__packFmtLowCmd = "<4B4IH2x" + "B3x5f3I" * 20 + "4B" + "55Bx2I"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.platform = platform.system()
        if self.platform == "Linux":
            if platform.machine() == "x86_64":
                self.crc_lib = ctypes.CDLL(script_dir + "/lib/crc_amd64.so")
            elif platform.machine() == "aarch64":
                self.crc_lib = ctypes.CDLL(script_dir + "/lib/crc_aarch64.so")

            self.crc_lib.crc32_core.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32)
            self.crc_lib.crc32_core.restype = ctypes.c_uint32

    def Crc(self, msg: idl.IdlStruct):
        return self.__Crc32(self.__PackLowCmd(msg))

    def __PackLowCmd(self, cmd: LowCmd):
        origData = []
        origData.extend(cmd.head)
        origData.append(cmd.level_flag)
        origData.append(cmd.frame_reserve)
        origData.extend(cmd.sn)
        origData.extend(cmd.version)
        origData.append(cmd.bandwidth)

        for i in range(20):
            origData.append(cmd.motor_cmd[i].mode)
            origData.append(cmd.motor_cmd[i].q)
            origData.append(cmd.motor_cmd[i].dq)
            origData.append(cmd.motor_cmd[i].tau)
            origData.append(cmd.motor_cmd[i].kp)
            origData.append(cmd.motor_cmd[i].kd)
            origData.extend(cmd.motor_cmd[i].reserve)

        origData.append(cmd.bms_cmd.off)
        origData.extend(cmd.bms_cmd.reserve)

        origData.extend(cmd.wireless_remote)
        origData.extend(cmd.led)
        origData.extend(cmd.fan)
        origData.append(cmd.gpio)
        origData.append(cmd.reserve)
        origData.append(cmd.crc)

        return self.__Trans(struct.pack(self.__packFmtLowCmd, *origData))

    def __Trans(self, packData):
        calcData = []
        calcLen = (len(packData) >> 2) - 1

        for i in range(calcLen):
            d = (
                (packData[i * 4 + 3] << 24)
                | (packData[i * 4 + 2] << 16)
                | (packData[i * 4 + 1] << 8)
                | (packData[i * 4])
            )
            calcData.append(d)

        return calcData

    def _crc_py(self, data):
        bit = 0
        crc = 0xFFFFFFFF
        polynomial = 0x04C11DB7

        for i in range(len(data)):
            bit = 1 << 31
            current = data[i]

            for b in range(32):
                if crc & 0x80000000:
                    crc = (crc << 1) & 0xFFFFFFFF
                    crc ^= polynomial
                else:
                    crc = (crc << 1) & 0xFFFFFFFF

                if current & bit:
                    crc ^= polynomial

                bit >>= 1

        return crc

    def _crc_ctypes(self, data):
        uint32_array = (ctypes.c_uint32 * len(data))(*data)
        length = len(data)
        crc = self.crc_lib.crc32_core(uint32_array, length)
        return crc

    def __Crc32(self, data):
        if self.platform == "Linux":
            return self._crc_ctypes(data)
        else:
            return self._crc_py(data)
