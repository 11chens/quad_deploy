import time

import cv2
import numpy as np
import pyzed.sl as sl


class ZedCamera:
    # Resolution mode mapping
    RESOLUTION_MODES = {"HD720": sl.RESOLUTION.HD720, "VGA": sl.RESOLUTION.VGA}  # 1280x720  # 672x376

    # Depth mode mapping
    DEPTH_MODES = {
        "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
        "NEURAL": sl.DEPTH_MODE.NEURAL,
        "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
    }

    def __init__(self, resolution_mode="HD720", depth_mode="NEURAL", scale: int = 2, *args, **kwargs):
        """Class to interface with Zed Mini camera using pyzed.sl SDK.
        Args:
            resolution_mode (str): Resolution mode for the camera ("HD720" or "VGA").
            depth_mode (str): Depth mode for the camera ("NEURAL_LIGHT", "NEURAL", or "NEURAL_PLUS").
            scale (int): Scale factor for resizing the image.
                - scale = 1: original resolution (1280x720)
                - scale = 2: half resolution (640x360)
                - scale = 4: quarter resolution (320x180)
                - scale = 8: eighth resolution (160x90)
        """
        self.zed = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.K = None  # Camera intrinsics matrix
        self.HFOV = None  # Camera HFOV
        self.VFOV = None  # Camera VFOV
        self.DFOV = None  # Camera DFOV
        self.rpy = None  # Euler angles
        self.T = None  # Translation vector, camera optical center position in base coordinate
        self.camera_resolution = None  # Camera resolution setting
        self.camera_depth_mode = None  # Camera depth mode setting
        self.runtime_parameters = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.point_cloud = sl.Mat()
        self.image_depth = sl.Mat()
        self.current_image = None
        self.current_point_cloud = None
        self.current_image_depth = None
        self.image_width = None
        self.image_height = None
        self.image_depth_width = None
        self.image_depth_height = None
        self.point_cloud_width = None
        self.point_cloud_height = None
        self.depth_max = 10.0
        self.depth_min = 0.1
        self.scale = scale  # scale down the image for faster processing
        self.initialize_camera(resolution_mode, depth_mode)
        self.get_intrinsics()

    def initialize_camera(self, resolution_mode, depth_mode):
        """Initialize ZED camera and open the device."""
        print("Initializing ZED camera...")

        # Validate and set resolution mode
        if resolution_mode not in ZedCamera.RESOLUTION_MODES:
            raise ValueError(
                f"Invalid resolution_mode: {resolution_mode}. Available options:"
                f" {list(ZedCamera.RESOLUTION_MODES.keys())}"
            )
        self.camera_resolution = ZedCamera.RESOLUTION_MODES[resolution_mode]
        print(f"Resolution mode set to: {resolution_mode}")

        # Validate and set depth mode
        if depth_mode not in ZedCamera.DEPTH_MODES:
            raise ValueError(
                f"Invalid depth_mode: {depth_mode}. Available options: {list(ZedCamera.DEPTH_MODES.keys())}"
            )
        self.camera_depth_mode = ZedCamera.DEPTH_MODES[depth_mode]
        print(f"Depth mode set to: {depth_mode}")

        init_params = sl.InitParameters()
        init_params.camera_resolution = self.camera_resolution
        init_params.depth_mode = self.camera_depth_mode
        init_params.coordinate_units = sl.UNIT.METER

        self.zed = sl.Camera()
        self.zed.close()

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open the camera: {repr(status)}")
            self.zed = None
            return False
        print("ZED camera initialized successfully.")
        return True

    def get_intrinsics(self):
        """
        Get intrinsics matrix K (fx, fy, cx, cy) from ZED camera.
        """
        if self.zed is None:
            print("Error: Camera not initialized, unable to get intrinsics.")
            return False

        print("Getting camera intrinsics...")
        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters

        self.fx = calib.left_cam.fx
        self.fy = calib.left_cam.fy
        self.cx = calib.left_cam.cx
        self.cy = calib.left_cam.cy
        self.HFOV = calib.left_cam.h_fov
        self.VFOV = calib.left_cam.v_fov
        self.DFOV = calib.left_cam.d_fov
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        print("Intrinsics loaded successfully.")
        return True

    def set_extrinsics(self, rpy, T):
        """
        Manually set extrinsics: Euler angles rpy and translation vector T
        """
        self.rpy = np.array(rpy, dtype=np.float64)
        self.T = np.array(T, dtype=np.float64)

        print("Extrinsics setting completed.")
        print("Euler angles rpy:\n", self.rpy)
        print("Translation vector T:\n", self.T)
        return True

    def capture_image(self, *args, **kwargs):
        """
        Capture a frame, and save the color image.
        """
        if self.zed.grab(self.runtime_parameters) != sl.ERROR_CODE.SUCCESS:
            print("Error: Failed to capture a frame.")
            return False, None
        self.zed.retrieve_image(
            self.image, sl.VIEW.LEFT, sl.MEM.CPU, sl.Resolution(1280 // self.scale, 720 // self.scale)
        )
        raw_image = self.image.get_data()  # BGRA format
        self.current_image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)  # BGR format
        if self.image_width is None or self.image_height is None:
            self.image_width = self.image.get_width()
            self.image_height = self.image.get_height()
            print(f"Image resolution: {self.image_width} x {self.image_height}")
        return True, self.current_image

    def capture_point_cloud(self, *args, **kwargs):
        """
        Capture a frame, and save the point cloud data.
        """

        # Get depth point cloud
        self.zed.retrieve_measure(
            self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, sl.Resolution(1280 // self.scale, 720 // self.scale)
        )
        self.current_point_cloud = self.point_cloud
        if self.point_cloud_width is None or self.point_cloud_height is None:
            self.point_cloud_width = self.point_cloud.get_width()
            self.point_cloud_height = self.point_cloud.get_height()
            print(f"point_cloud resolution: {self.point_cloud_width} x {self.point_cloud_height}")
        return True, self.current_point_cloud

    def capture_depth(self, *args, **kwargs):
        """
        Capture a frame, and save the depth data.
        """
        if self.zed.grab(self.runtime_parameters) != sl.ERROR_CODE.SUCCESS:
            print("Error: Failed to capture a frame.")
            return False, None
        # Get depth data
        self.zed.retrieve_measure(
            self.image_depth, sl.MEASURE.DEPTH, sl.MEM.CPU, sl.Resolution(1280 // self.scale, 720 // self.scale)
        )
        self.current_image_depth = self.image_depth
        if self.image_depth_width is None or self.image_depth_height is None:
            self.image_depth_width = self.image_depth.get_width()
            self.image_depth_height = self.image_depth.get_height()
            print(f"image_depth resolution: {self.image_depth_width} x {self.image_depth_height}")
        return True, self.current_image_depth

    def get_depth_at_pixel_point_cloud(self, u, v):
        """
        Get depth value at pixel (u, v) from point cloud.
        """
        if self.current_point_cloud is None:
            print("Error: No available point cloud. Please capture a frame first.")
            return False

        u = int(max(0, min(u, self.point_cloud_width - 1)))
        v = int(max(0, min(v, self.point_cloud_height - 1)))

        try:
            ret = self.current_point_cloud.get_value(u, v)
            _, point = ret  # point = [X, Y, Z, RGBA]
            depth = float(point[2])
            if np.isfinite(depth) and depth > 0.0:
                return depth
            elif np.isnan(depth):
                return self.depth_min
            elif np.isinf(depth):
                return self.depth_max
            else:
                print(f"Warning: Depth invalid at pixel ({u}, {v}): {depth}")
                return False
        except Exception as e:
            print(f"Error: Failed to get depth at pixel ({u}, {v}): {e}")
            return False

    def get_depth_at_pixel_depth(self, u, v):
        """
        Get depth value at pixel (u, v).
        """
        if self.current_image_depth is None:
            print("Error: No depth data available. Please capture a frame first.")
            return False

        u = int(max(0, min(u, self.image_depth_width - 1)))
        v = int(max(0, min(v, self.image_depth_height - 1)))

        try:
            ret = self.current_image_depth.get_value(u, v)
            _, point = ret
            depth = float(point)
            if np.isfinite(depth) and depth > 0.0:
                return depth
            elif np.isnan(depth):
                return self.depth_min
            elif np.isinf(depth):
                return self.depth_max
            else:
                print(f"Warning: Depth invalid at pixel ({u}, {v}): {depth}")
                return False
        except Exception as e:
            print(f"Error: Failed to get depth at pixel ({u}, {v}): {e}")
            return False

    def close(self):
        """close the camera and release the resources."""
        if self.zed is not None:
            self.zed.close()
            self.zed = None
            print("ZED camera closed.")


if __name__ == "__main__":
    cam = ZedCamera("VGA", "NEURAL")
    depth_start = time.monotonic()
    NUM_ITR = 100
    # only show image
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    for i in range(NUM_ITR):
        _, image = cam.capture_image()
        _, image_depth = cam.capture_depth()
        depth = cam.get_depth_at_pixel_depth(320, 180)
        cv2.imshow("image", image)
        cv2.waitKey(1)
        print(f"Depth at step_{i}: {depth:.2f} m")

    depth_end = time.monotonic()
    depth_time = depth_end - depth_start
    depth_time /= NUM_ITR

    # print result
    print(f"Depth query time: {depth_time*1e3:.3f} ms")

    cam.close()
