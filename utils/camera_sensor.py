import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_np


class camera_sensor:
    fix_extrinsics = True  # If True, camera extrinsics are fixed; otherwise, randomized
    fix_intrinsics = True  # If True, camera intrinsics are fixed; otherwise, randomized
    img_width = 1280  # Image width in pixels
    img_height = 720  # Image height in pixels

    class intrinsics:  # Camera intrinsic parameters
        # Zed mini, HD720 mode
        horizontal_fov = 82
        fx = 731.995849609375
        fy = 731.995849609375
        cx = 620.0855102539062
        cy = 362.5731201171875

    class extrinsics:  # Camera extrinsic parameters
        translation = [0.35, 0.0, 0.0]  # Translation: forward, left, upward
        angles = [0.0, 0.0, 0.0]  # Euler angles: yaw, pitch, roll


class CameraSensor:
    """
    Camera model class for single-point processing, supporting transformations between base, camera, and image coordinate systems.

    Coordinate Systems:
    - Base: Robot reference frame, X forward, Y left, Z upward.
    - Camera: Camera optical frame, Z forward (toward scene), X right (image width), Y down (image height).
    - Image: 2D image plane, origin at top-left, u along width, v along height.

    This class provides functions for transformations (base → camera, camera → image, image → camera, camera → base)
    using predefined or randomized intrinsics and extrinsics for forward and inverse geometric transformations.
    """

    def __init__(self, cfg=camera_sensor):
        self.cfg = cfg
        self.fix_extrinsics = cfg.fix_extrinsics
        self.fix_intrinsics = cfg.fix_intrinsics
        self.intrinsics_cfg = cfg.intrinsics
        self.extrinsics_cfg = cfg.extrinsics
        self.img_width = self.cfg.img_width
        self.img_height = self.cfg.img_height

        if self.fix_intrinsics:
            self._init_fixed_intrinsics()
        else:
            self._init_random_intrinsics()

        if self.fix_extrinsics:
            self._init_fixed_extrinsics()
        else:
            self._init_random_extrinsics()

    def _init_fixed_intrinsics(self):
        """
        Initialize fixed camera intrinsics.

        Intrinsics include:
        - fx, fy: Focal length (pixels), controlling magnification in x and y directions.
        - cx, cy: Principal point (pixels), typically at image center.
        """
        self.fx = self.intrinsics_cfg.fx
        self.fy = self.intrinsics_cfg.fy
        self.cx = self.intrinsics_cfg.cx
        self.cy = self.intrinsics_cfg.cy

    def _init_random_intrinsics(self):
        """
        Initialize random camera intrinsics.

        Intrinsics include:
        - fx, fy: Focal length (pixels), controlling magnification in x and y directions.
        - cx, cy: Principal point (pixels), typically at image center.
        """
        # Example random initialization, ranges can be adjusted
        self.fx = np.random.uniform(80, 180)  # Random fx in [80, 180]
        self.fy = np.random.uniform(80, 180)  # Random fy in [80, 180]
        self.cx = self.img_width / 2  # cx at image width midpoint
        self.cy = self.img_height / 2  # cy at image height midpoint

    def _init_fixed_extrinsics(self):
        """
        Initialize fixed camera extrinsics (pose).

        Extrinsics consist of:
        - Rotation matrix R: Aligns base frame (X forward, Y left, Z up) to camera frame (X right, Y down, Z forward).
        - Translation vector T: Camera optical center position in base frame.
        """
        # Fixed Euler angles (degrees)
        angles = np.array(self.extrinsics_cfg.angles)  # [yaw, pitch, roll]
        # Generate rotation matrix from Euler angles (ZYX order)
        rmat = R_np.from_euler("ZYX", angles, degrees=True).as_matrix()
        r_align = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)  # Alignment matrix
        self.R = r_align @ rmat

        # Fixed translation
        self.T = np.array(self.extrinsics_cfg.translation, dtype=np.float32)  # [dx, dy, dz]

    def _init_random_extrinsics(self):
        """
        Initialize random camera extrinsics (pose).

        Extrinsics consist of:
        - Rotation matrix R: Describes rotation from base to camera frame, generated from random yaw, pitch, roll.
          yaw ∈ [-10°, 10°], pitch ∈ [-20°, 20°], roll ∈ [-10°, 10°].
          Uses ZYX Euler angles and aligns base (X forward, Y left, Z up) to camera (X right, Y down, Z forward).
        - Translation vector T: Camera optical center position in base frame.
          dx ∈ [0.3, 0.6], dy ∈ [-0.05, 0.05], dz ∈ [-0.2, 0.2].
        """
        # Random Euler angles (degrees)
        yaw = np.random.uniform(-10, 10)  # yaw in [-10°, 10°]
        pitch = np.random.uniform(-20, 20)  # pitch in [-20°, 20°]
        roll = np.random.uniform(-10, 10)  # roll in [-10°, 10°]
        angles = np.array([yaw, pitch, roll])
        # Generate rotation matrix from Euler angles (ZYX order)
        rmat = R_np.from_euler("ZYX", angles, degrees=True).as_matrix()
        r_align = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)  # Alignment matrix
        self.R = r_align @ rmat

        # Random translation
        dx = np.random.uniform(0.3, 0.6)  # x in [0.3, 0.6]
        dy = np.random.uniform(-0.05, 0.05)  # y in [-0.05, 0.05]
        dz = np.random.uniform(-0.2, 0.2)  # z in [-0.2, 0.2]
        self.T = np.array([dx, dy, dz], dtype=np.float32)

    def base_to_camera(self, P_base: np.ndarray) -> np.ndarray:
        """
        Transform a point from base coordinate system to camera coordinate system.

        Args:
            P_base (ndarray): Shape [3], point in base coordinate system.

        Returns:
            ndarray: Shape [3], point in camera coordinate system.

        Formula:
            P_cam = R * (P_base - T)
        where R is the rotation matrix, T is the camera optical center in base frame.
        """
        delta = P_base - self.T  # [3]
        P_camera = self.R @ delta  # [3]
        return P_camera

    def camera_to_image(self, P_camera: np.ndarray) -> np.ndarray:
        """
        Project a point from camera coordinate system to normalized image plane.

        Args:
            P_camera (ndarray): Shape [3], point in camera coordinate system.

        Returns:
            ndarray: Shape [2], normalized image coordinates in [0,1].
                     Point outside FOV or with Z<=0 is set to [-1, -1].
        """
        X, Y, Z = P_camera[0], P_camera[1], P_camera[2]
        # Compute pixel coordinates using pinhole camera model
        u = self.fx * X / Z + self.cx
        v = self.fy * Y / Z + self.cy
        # Normalize to [0, 1]
        u_norm = u / self.img_width
        v_norm = v / self.img_height

        coords = np.array([u_norm, v_norm])
        # Check if point is within FOV and has positive depth
        if Z <= 0 or u_norm < 0 or u_norm > 1 or v_norm < 0 or v_norm > 1:
            coords = np.array([-1.0, -1.0])
        return coords

    def transform(self, P_base: np.ndarray):
        """
        Full transformation from base to camera to image coordinate system.

        Args:
            P_base (ndarray): Shape [3], point in base coordinate system.

        Returns:
            Tuple[ndarray, ndarray]: Camera coordinates [3], normalized image coordinates [2].
        """
        P_camera = self.base_to_camera(P_base)
        P_image = self.camera_to_image(P_camera)
        return P_camera, P_image

    def image_to_camera(self, uv_norm: np.ndarray, depth: float) -> np.ndarray:
        """
        Back-project normalized image coordinates and depth to camera coordinate system.

        Args:
            uv_norm (ndarray): Shape [2], normalized image coordinates in [0,1].
            depth (float): Depth value (Z component).

        Returns:
            ndarray: Shape [3], point in camera coordinate system.
        """
        u, v = uv_norm[0], uv_norm[1]
        # Convert normalized coordinates to pixel coordinates
        u_pix = u * self.img_width
        v_pix = v * self.img_height
        # Compute camera coordinates using back-projection
        x = (u_pix - self.cx) * depth / self.fx
        y = (v_pix - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z])  # [3]

    def camera_to_base(self, P_camera: np.ndarray) -> np.ndarray:
        """
        Transform a point from camera coordinate system to base coordinate system.

        Args:
            P_camera (ndarray): Shape [3], point in camera coordinate system.

        Returns:
            ndarray: Shape [3], point in base coordinate system.

        Formula:
            P_base = R^T * P_camera + T
        where R^T is the transpose of the rotation matrix, T is the camera optical center.
        """
        return self.R.T @ P_camera + self.T  # [3]

    def inverse_transform(self, uv_norm: np.ndarray, depth: float):
        """
        Back-project from normalized image coordinates and depth to camera and base coordinate systems.

        Args:
            uv_norm (ndarray): Shape [2], normalized image coordinates.
            depth (float): Depth value.

        Returns:
            Tuple[ndarray, ndarray]: Camera coordinates [3], base coordinates [3].
        """
        P_camera = self.image_to_camera(uv_norm, depth)
        P_base = self.camera_to_base(P_camera)
        return P_camera, P_base

    def visualize_img(self, img):
        """
        Display an image using OpenCV.

        Args:
            img (ndarray): Image to display (BGR format).
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    def visualize_img_coords(self, P_base, P_camera, P_image):
        """
        Visualize a single point's projection in the camera view.
        If normalized image coordinates contain -1, display a message indicating the point is out of view or behind the camera.
        Otherwise, draw the point and overlay base, camera, and normalized image coordinates.

        Args:
            P_base: Array of length 3, point in base coordinate system.
            P_camera: Array of length 3, point in camera coordinate system.
            P_image: Array of length 2, normalized image coordinates.

        Returns:
            bool: False if ESC key is pressed (exit), True otherwise.
        """
        image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        if (P_image == -1).any():
            cv2.putText(
                image,
                "Out of view or behind camera",
                (10, self.img_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        else:
            u_pixel = int(P_image[0] * self.img_width)
            v_pixel = int(P_image[1] * self.img_height)

            # Draw projected point
            cv2.circle(image, (u_pixel, v_pixel), 5, (0, 0, 255), -1)

            # Overlay normalized image, base, and camera coordinates
            text = f"Normal P_img: ({P_image[0]:.3f}, {P_image[1]:.3f})"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            text_base = f"P_base: ({P_base[0]:.3f}, {P_base[1]:.3f}, {P_base[2]:.3f})"
            cv2.putText(image, text_base, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            text_camera = f"P_cam: ({P_camera[0]:.3f}, {P_camera[1]:.3f}, {P_camera[2]:.3f})"
            cv2.putText(image, text_camera, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Camera View", image)

        key = cv2.waitKey(1)

        if key == 27:
            cv2.destroyAllWindows()
            return False

        return True


def verify_camera_transform(camera: CameraSensor, num_points: int = 10) -> float:
    """
    Generate random points in base coordinate system, perform forward and inverse transformations,
    and verify transformation accuracy.

    Args:
        camera (CameraSensor): Camera object.
        num_points (int): Number of test points.

    Returns:
        float: Maximum reconstruction error across all test points.
    """
    np.random.seed(42)
    max_error = 0.0

    for _ in range(num_points):
        # Generate random base coordinate point
        P_base = np.array([np.random.uniform(0.5, 6.0), np.random.uniform(-2.0, 2.0), np.random.uniform(-1.0, 2.0)])

        # Forward transformation: base → camera → image
        P_camera, P_image = camera.transform(P_base)

        # Skip if point is out of view
        if (P_image == -1).any():
            print(f"[Skipped] Point {P_base} is out of view or behind camera")
            continue

        u_norm, v_norm = P_image
        depth = P_camera[2]
        uv_norm = np.array([u_norm, v_norm])

        # Inverse transformation: image → camera → base
        P_camera_recon, P_base_recon = camera.inverse_transform(uv_norm, depth)

        # Compute reconstruction error
        error = np.linalg.norm(P_base - P_base_recon)
        max_error = max(max_error, error)

        if error > 1e-4:
            print(f"[Warning] Point {P_base} reconstruction error: {error:.6f}")
        else:
            print(f"[Correct] Point {P_base} transformation consistent")

    # Print maximum error
    print("\nDone")
    print(f"Maximum error: {max_error:.6e}")
    return float(max_error)


if __name__ == "__main__":
    cam = CameraSensor(cfg=camera_sensor)
    verify_camera_transform(cam, num_points=20)
