import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch 
import cv2
from sklearn.linear_model import LinearRegression
from PIL import Image
import torchvision.transforms as T
from simple_lama_inpainting import SimpleLama
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser
import sys
import shutil
import math 
from scene.colmap_loader import rotmat2qvec, qvec2rotmat
from scene.cameras import PseudoCamera
from pathlib import PosixPath
import warnings 
warnings.filterwarnings('ignore')


class DepthMap:
    _instance = None

    @classmethod
    def get_instance(cls, model_type: str="DPT_Large", gpu_id: int=0):
        if cls._instance is None:
            cls._instance = DepthMap(model_type, gpu_id)
        return cls._instance
    
    def __init__(self, model_type: str="DPT_Large", gpu_id: int = 0):
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def depth(self, imagepath):
        if isinstance(imagepath, str):
            img = cv2.imread(imagepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else: 
            img = imagepath
        inp = self.transform(img)
        inp = inp.to(self.device)
        with torch.no_grad():
            prediction = self.model(inp)
            prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()
        return output

def _get_K(input_path):
    """
    Reads the 'cameras.txt' file in the given input directory and extracts
    the intrinsic camera matrix K for a PINHOLE model.

    Expected format of the line:
    CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy

    Parameters:
        input_path (str): Path to the folder containing 'cameras.txt'.

    Returns:
        np.ndarray: 3x3 camera intrinsic matrix K.
    Note:
        This will work in PINHOLE model only
    """
    camera_file = f"{input_path}/cameras.txt"
    with open(camera_file, 'r') as f:
        lines = f.readlines()

    # Skip commented and header lines
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) < 8:
            raise ValueError("Unexpected line format in cameras.txt")

        # Extract parameters
        fx, fy, cx, cy = map(float, parts[4:8])

        # Construct intrinsic matrix
        K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ])
        return K

    raise ValueError("No valid camera line found in cameras.txt")

def _quaternion_to_R(qw, qx, qy, qz):
    """
    Convert a quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    # Normalize quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def _get_2d_3d_corr(input_path):
    """
    Extracts 2D–3D correspondences and camera poses (R, t)
    from COLMAP-style SfM text files.

    Returns:
        dict: {
            image_name: {
                "pixel_2d": [[x, y], ...],
                "coordinate_3D": [[X, Y, Z], ...],
                "q": np.ndarray(4,)
                "R": np.ndarray(3,3),
                "t": np.ndarray(3,1)
            }
        }
    """

    images_file = os.path.join(input_path, "images.txt")
    points3d_file = os.path.join(input_path, "points3D.txt")

    # --- Step 1: Parse 3D points file ---
    point3d_dict = {}
    with open(points3d_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            point3d_id = int(parts[0])
            X, Y, Z = map(float, parts[1:4])
            point3d_dict[point3d_id] = [X, Y, Z]

    # --- Step 2: Parse images file ---
    corr_dict = {}
    with open(images_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue

        # Parse image header
        header = line.split()
        image_id = int(header[0])
        qw, qx, qy, qz = map(float, header[1:5])
        tx, ty, tz = map(float, header[5:8])
        image_name = header[-1]

        R = _quaternion_to_R(qw, qx, qy, qz)
        t = np.array([[tx], [ty], [tz]])

        corr_dict[image_name] = {
            "pixel_2d": [],
            "coordinate_3D": [],
            "q" : np.array([qw, qx, qy, qz]),
            "R": R,
            "t": t
        }

        # Next line has POINTS2D
        i += 1
        if i >= len(lines):
            break

        pts_line = lines[i].strip().split()
        for j in range(0, len(pts_line), 3):
            x, y = map(float, pts_line[j:j+2])
            point3d_id = int(pts_line[j+2])
            if point3d_id == -1:
                continue
            if point3d_id in point3d_dict:
                X, Y, Z = point3d_dict[point3d_id]
                corr_dict[image_name]["pixel_2d"].append([round(x), round(y)])
                corr_dict[image_name]["coordinate_3D"].append([X, Y, Z])
        corr_dict[image_name]['pixel_2d'] = np.array(corr_dict[image_name]['pixel_2d'])
        corr_dict[image_name]['coordinate_3D'] = np.array(corr_dict[image_name]["coordinate_3D"])
        i += 1

    return corr_dict

def _robust_depth_regression(midas_pixel_depths, camera_depths, depth_midas):
    """
    Fit robust linear regression from MiDaS to SFM depths.
    Inverts first, then shifts for numerical stability.
    
    Args:
        midas_pixel_depths: 1D np.array of MiDaS depths at selected pixels
        camera_depths: 1D np.array of SFM depths (target)
        depth_midas: 2D np.array (H,W) of MiDaS depths for prediction
    
    Returns:
        depth_pred: 2D np.array (H,W) predicted SFM depths
        reg: fitted LinearRegression object
        scaler_X, scaler_y: fitted StandardScaler objects
    """
    epsilon = -depth_midas.min() + 2.0
    
    # --- Step 1: Invert first ---
    X = 1.0 / (midas_pixel_depths + epsilon)
    
    # --- Step 2: Shift to positive range ---
    X = X - X.min() + epsilon
    X = X.reshape(-1,1)
    
    y = camera_depths.reshape(-1,1)
    
    # --- Step 3: Standardize X and y ---
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # --- Step 4: Fit regression ---
    reg = LinearRegression()
    reg.fit(X_scaled, y)
    
    # --- Step 5: Prepare full MiDaS map ---
    X_full = 1.0 / (depth_midas + epsilon)
    X_full = X_full - X_full.min() + epsilon
    X_full = X_full.reshape(-1,1)
    X_full_scaled = scaler_X.transform(X_full)
    
    # --- Step 6: Predict and inverse scale ---
    depth_pred_scaled = reg.predict(X_full_scaled)
    depth_pred = depth_pred_scaled.reshape(depth_midas.shape)
    
    # --- Step 7: Clip negative values ---
    depth_pred = np.clip(depth_pred, 0, None)
    
    return depth_pred

def _predict_pixelwise_depth(image, pixels: np.ndarray, camera_depths: np.ndarray):
    """
    pixels: shape (N, 2) - N pixels for which I have camera depths value available
    camera_depths: (N,) - N depth values obtained from SFM
    """
    if isinstance(image, str) or isinstance(image, PosixPath):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    midas = DepthMap.get_instance()
    midas_out = midas.depth(image)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, image.shape[0]-1)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, image.shape[1]-1)
    midas_pixel_depths  = midas_out[pixels[:, 1], pixels[:, 0]]
    depth_pred = _robust_depth_regression(midas_pixel_depths, camera_depths, midas_out)

    return depth_pred

def _warp_weighted(I1, R1, t1, depth1, R2, t2, K, r=2.0):
    """
    Forward warp an image using normalized weighted splatting.
    Implements spatial bilinear weights and depth weighting.

    Parameters:
        I1 : np.ndarray
            HxW or HxWxC source image or depth map
        R1, t1 : np.ndarray
            Camera1 (world to camera) rotation (3x3) and translation (3x1)
        depth1 : np.ndarray
            Depth map in camera1 coordinates
        R2, t2 : np.ndarray
            Camera2 (world to camera) rotation and translation
        K : np.ndarray
            Camera intrinsic matrix
        r : float
            Depth weight exponent (higher = nearer points dominate more)

    Returns:
        I2 : np.ndarray
            Warped image in camera2 coordinates
    """
    H, W = depth1.shape
    I1 = I1.astype(np.float32)

    # Add channel dimension if grayscale
    if I1.ndim == 2:
        I1 = I1[..., None]
    C = I1.shape[2]

    # Accumulators
    I2_sum = np.zeros((H, W, C), dtype=np.float32)
    weight_sum = np.zeros((H, W, C), dtype=np.float32)

    # Source pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u_flat = u.flatten()
    v_flat = v.flatten()
    z_flat = depth1.flatten()
    ones = np.ones_like(u_flat)

    # Back-project to camera1 coordinates
    pixels_hom = np.stack([u_flat, v_flat, ones], axis=0)  # 3xN
    xyz_cam1 = np.linalg.inv(K) @ pixels_hom * z_flat
    xyz_world = R1.T @ (xyz_cam1 - t1)  # 3xN

    # Project world points to camera2
    xyz_cam2 = R2 @ xyz_world + t2
    z2 = xyz_cam2[2, :]
    pixels2_hom = K @ xyz_cam2
    u2 = pixels2_hom[0, :] / z2
    v2 = pixels2_hom[1, :] / z2

    # Depth weight
    Wd = 1.0 / (1.0 + z_flat) ** r

    # Floor coordinates and fractional offsets for bilinear splatting
    u0 = np.floor(u2).astype(int)
    v0 = np.floor(v2).astype(int)
    du = u2 - u0
    dv = v2 - v0

    # Iterate over 4 neighbors
    for i_shift in [0, 1]:
        for j_shift in [0, 1]:
            u_idx = u0 + i_shift
            v_idx = v0 + j_shift

            # Spatial weight
            Wp = (1 - np.abs(i_shift - du)) * (1 - np.abs(j_shift - dv))
            weight = Wp * Wd

            # Mask valid target pixels
            mask = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
            if not np.any(mask):
                continue

            if C == 1:
                I2_sum[v_idx[mask], u_idx[mask], 0] += weight[mask] * I1[v_flat[mask], u_flat[mask], 0]
                weight_sum[v_idx[mask], u_idx[mask], 0] += weight[mask]
            else:
                I2_sum[v_idx[mask], u_idx[mask], :] += (weight[mask][:, None] * I1[v_flat[mask], u_flat[mask], :])
                weight_sum[v_idx[mask], u_idx[mask], :] += weight[mask][:, None]

    # Normalize
    nonzero = weight_sum > 0
    I2 = np.zeros_like(I2_sum)
    I2[nonzero] = I2_sum[nonzero] / weight_sum[nonzero]

    # Remove channel dimension for grayscale
    if C == 1:
        I2 = I2[..., 0]

    return I2

def _focal2fov(focal, pixels):
    """Convert focal length to field of view"""
    return 2 * math.atan(pixels / (2 * focal))

def _fov2focal(fov, pixels):
    """Convert field of view to focal length"""
    return pixels / (2 * math.tan(fov / 2))

def _getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """
    Match 3DGS camera transform exactly
    R: 3x3 rotation (world to camera)
    t: 3x1 or (3,) translation
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()  # Camera to world rotation
    Rt[:3, 3] = t.flatten()
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return Rt

def _warp_weighted_3dgs(I1, R1, t1, depth1, R2, t2, FoVx, FoVy, width, height, 
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, r=2.0):
    """
    Forward warp matching 3DGS camera convention exactly.
    
    Parameters:
        I1 : np.ndarray (HxW or HxWxC) - source image
        R1, t1 : Camera1 extrinsics (R: 3x3, t: 3x1 or (3,))
        depth1 : Depth map in camera1 coordinates
        R2, t2 : Camera2 extrinsics
        FoVx, FoVy : Field of view in radians
        width, height : Image dimensions
        trans : Translation offset
        scale : Scale factor
        r : Depth weight exponent
    
    Returns:
        I2 : Warped image
    """
    H, W = depth1.shape
    assert H == height and W == width, f"Depth shape {depth1.shape} != image size ({height}, {width})"
    
    I1 = I1.astype(np.float32)
    if I1.ndim == 2:
        I1 = I1[..., None]
    C = I1.shape[2]
    
    # Convert FoV to focal lengths
    fx = _fov2focal(FoVx, W)
    fy = _fov2focal(FoVy, H)
    cx = W / 2.0
    cy = H / 2.0
    
    # Intrinsics matrix
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    K_inv = np.linalg.inv(K)
    
    # Get world-to-view transforms matching 3DGS
    W2V1 = _getWorld2View2(R1, t1, trans, scale)  # 4x4
    W2V2 = _getWorld2View2(R2, t2, trans, scale)  # 4x4
    
    # Extract camera centers in world coordinates
    V2W1 = np.linalg.inv(W2V1)
    V2W2 = np.linalg.inv(W2V2)
    
    # Accumulators
    I2_sum = np.zeros((H, W, C), dtype=np.float32)
    weight_sum = np.zeros((H, W, C), dtype=np.float32)
    
    # Source pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u_flat = u.flatten()
    v_flat = v.flatten()
    z_flat = depth1.flatten()
    
    # Back-project to camera1 3D coordinates
    pixels_hom = np.stack([u_flat, v_flat, np.ones_like(u_flat)], axis=0)  # 3xN
    xyz_cam1_3d = K_inv @ pixels_hom * z_flat[None, :]  # 3xN
    
    # Convert to homogeneous coordinates
    xyz_cam1 = np.vstack([xyz_cam1_3d, np.ones((1, xyz_cam1_3d.shape[1]))])  # 4xN
    
    # Transform to world coordinates
    xyz_world = V2W1 @ xyz_cam1  # 4xN
    
    # Transform to camera2 coordinates
    xyz_cam2 = W2V2 @ xyz_world  # 4xN
    
    # Project to camera2 image plane
    xyz_cam2_3d = xyz_cam2[:3, :]  # 3xN
    z2 = xyz_cam2_3d[2, :]
    
    # Filter out points behind camera
    valid = z2 > 0
    if not np.any(valid):
        return I2_sum[..., 0] if C == 1 else I2_sum
    
    # Project using intrinsics
    pixels2_hom = K @ xyz_cam2_3d[:, valid]  # 3xN_valid
    u2 = pixels2_hom[0, :] / z2[valid]
    v2 = pixels2_hom[1, :] / z2[valid]
    
    # Depth weighting
    z2_valid = z2[valid]
    Wd = 1.0 / (z2_valid ** r + 1e-8)
    
    # Valid source indices
    valid_idx = np.where(valid)[0]
    v_src = v_flat[valid_idx]
    u_src = u_flat[valid_idx]
    
    # Bilinear splatting
    u0 = np.floor(u2).astype(int)
    v0 = np.floor(v2).astype(int)
    du = u2 - u0
    dv = v2 - v0
    
    for i_shift in [0, 1]:
        for j_shift in [0, 1]:
            u_tgt = u0 + i_shift
            v_tgt = v0 + j_shift
            
            # Bilinear weight
            Wp = (1 - np.abs(i_shift - du)) * (1 - np.abs(j_shift - dv))
            weight = Wp * Wd
            
            # Mask for valid target pixels
            mask = (u_tgt >= 0) & (u_tgt < W) & (v_tgt >= 0) & (v_tgt < H)
            if not np.any(mask):
                continue
            
            v_m = v_tgt[mask]
            u_m = u_tgt[mask]
            v_sm = v_src[mask]
            u_sm = u_src[mask]
            w_m = weight[mask]
            
            # Accumulate
            I2_sum[v_m, u_m, :] += I1[v_sm, u_sm, :] * w_m[:, None]
            weight_sum[v_m, u_m, :] += w_m[:, None]
    
    # Normalize
    I2 = np.zeros_like(I2_sum)
    nonzero = weight_sum > 1e-8
    I2[nonzero] = I2_sum[nonzero] / weight_sum[nonzero]
    
    if C == 1:
        I2 = I2[..., 0]
    
    return I2

def _generate_inpaint_mask(I2, valid_threshold=0.01, dilate_iter=5):
    """
    Generate a binary mask for missing pixels in a warped image.
    Args:
        I2: (H, W, 3) or (H, W) warped image, values in [0,1].
        valid_threshold: pixels below this intensity are treated as holes.
        dilate_iter: number of dilation iterations to smooth/expand mask.
    Returns:
        mask: binary (H, W) np.uint8 mask, 1 for missing pixels.
    """

    # Convert to grayscale if needed
    if I2.ndim == 3:
        gray = cv2.cvtColor((I2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (I2 * 255).astype(np.uint8)

    # Threshold: 0 where warped values exist, 1 where holes exist
    mask = (gray < valid_threshold * 255).astype(np.uint8)

    # Dilate to fill small gaps
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    return mask

def _visualise_points(image_path, points, savepath=None):
    """
    Visualize an image with 2D points overlaid as green dots.

    Parameters:
        image_path (str): Path to the image file (e.g. 'sample.png').
        points (array-like): Nx2 array or list of [x, y] pixel coordinates.
    """
    # Read image
    img = mpimg.imread(image_path)

    # Convert to numpy array
    pts = np.array(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an Nx2 array or list")

    # Display image
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    
    # Overlay points (x = column, y = row)
    plt.scatter(pts[:, 0], pts[:, 1], s=10, c='lime', marker='o', edgecolors='black', linewidths=0.5)
    
    # Aesthetic settings
    # plt.title(f"2D Points on {image_path.split('/')[-1]}")
    plt.axis('off')
    plt.tight_layout()
    if savepath is not None: 
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

def _perturb_camera(R, t, rot_deg=5.0, trans_mag=0.1):
    """
    Add a small perturbation to a given rotation and translation.

    Parameters:
        R : np.ndarray
            Original 3x3 rotation matrix
        t : np.ndarray
            Original 3x1 translation vector
        rot_deg : float
            Maximum rotation in degrees for perturbation
        trans_mag : float
            Maximum translation magnitude for perturbation

    Returns:
        R_new : np.ndarray
            Perturbed rotation matrix
        t_new : np.ndarray
            Perturbed translation vector
    """
    # Small random rotation
    theta = np.deg2rad(rot_deg)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    ux, uy, uz = axis
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    R_perturb = np.array([
        [cos_t + ux**2*(1-cos_t),      ux*uy*(1-cos_t) - uz*sin_t, ux*uz*(1-cos_t) + uy*sin_t],
        [uy*ux*(1-cos_t) + uz*sin_t,   cos_t + uy**2*(1-cos_t),    uy*uz*(1-cos_t) - ux*sin_t],
        [uz*ux*(1-cos_t) - uy*sin_t,   uz*uy*(1-cos_t) + ux*sin_t, cos_t + uz**2*(1-cos_t)]
    ])

    R_new = R_perturb @ R 

    t_new = t + trans_mag * np.random.randn(3,1)

    return R_new, t_new

def _interpolate_q(t, q, other_q, trans_mag=0.1):
    """
    Interpolate a camera angle by linear interpolation with other cameras

    Parameters:
        R : np.ndarray
            Original 3x3 rotation matrix
        t : np.ndarray
            Original 3x1 translation vector
        q : np.ndarray
            (4, )
        other_q : List[np.ndarray]

    Returns:
        R_new : np.ndarray
            Perturbed rotation matrix
        t_new : np.ndarray
            Perturbed translation vector
    """
    # randomly select a q 
    random_idx = np.random.randint(0, len(other_q)-1)
    other_q = other_q[random_idx]
    alpha = np.random.random() 
    interpolated_q = alpha*q + (1-alpha)*other_q
    
    R_new = _quaternion_to_R(*list(interpolated_q))
    t_new = t + trans_mag * np.random.randn(3,1)

    return R_new, t_new

def _perturb_q(t, q, trans_mag=0.1, q_mag=0.01):
    """
    Perturb small noise to quarternion and t, normlise quarternion

    Parameters:
        t : np.ndarray
            Original 3x1 translation vector
        q : np.ndarray
            (4, )

    Returns:
        R_new : np.ndarray
            Perturbed rotation matrix
        t_new : np.ndarray
            Perturbed translation vector
    """
    # randomly select a q 
    q_pert = q + np.random.uniform(-1, 1, size=(4,))*q_mag
    q_pert = q_pert/ np.linalg.norm(q_pert)
    R_new = _quaternion_to_R(*list(q_pert))
    t_new = t + trans_mag * np.random.randn(3,1)

    return R_new, t_new

def run_augment_pipeline(input_path, output_path, n_new_images):
    """
    The input path should contain 'images' folder and 'sparse/0' folder for sfm params.
    """
    # get 2D-3D correspondence dictionary first
    corr_dict = _get_2d_3d_corr(input_path/'sparse/0')
    K = _get_K(input_path/'sparse/0')
    inpainter = SimpleLama()
    
    # all quarternions - required for R perturbation
    all_q = {} 
    for imagefile in corr_dict.keys():
        all_q[imagefile] = corr_dict[imagefile]['q'] 

    for imagefile in corr_dict.keys(): 
        pixels = corr_dict[imagefile]['pixel_2d']
        sfm_3d = corr_dict[imagefile]['coordinate_3D']
        R, t = corr_dict[imagefile]['R'], corr_dict[imagefile]['t']
        camera_3d = R@sfm_3d.T + t 
        camera_depths = camera_3d[2, :]

        # q and other q
        q = corr_dict[imagefile]['q']
        other_q = [corr_dict[key]['q'] for key in corr_dict.keys() if key != imagefile]

        # depth prediction in actual scale
        pred_depth = _predict_pixelwise_depth(input_path/f'images/{imagefile}', pixels, camera_depths)
        
        # for each input image do new_images perturbation and generate 10 perturbed poses
        I_original = cv2.imread(input_path/f'images/{imagefile}')
        I_original = cv2.cvtColor(I_original, cv2.COLOR_BGR2RGB)
        for i in range(n_new_images):
            # R_new, t_new = _perturb_camera(R, t, rot_deg=2, trans_mag=0.001) 
            # R_new, t_new = _interpolate_q(t, q, other_q, trans_mag=0.001)
            R_new, t_new = _perturb_q(t, q, q_mag=0.03, trans_mag=0.005)

            I_pseudo = _warp_weighted(I_original, R, t, pred_depth, R_new, t_new, K)
            I_pseudo = I_pseudo/np.max(I_pseudo)
            
            # inpainting
            mask = _generate_inpaint_mask(I_pseudo)
            mask_pil = Image.fromarray((mask*255).astype(np.uint8), mode="L")
            I2_pil_hole = Image.fromarray((I_pseudo*255).astype(np.uint8))
            I_pseudo_pil = inpainter(I2_pil_hole, mask_pil)

            I_pseudo_pil = I_pseudo_pil.resize((I_pseudo.shape[1], I_pseudo.shape[0]))
            I_pseudo_pil.save(output_path/f'{imagefile}_{i}.png')

class PseudoViewGeneratorTraining: 
    def __init__(self, input_path): 
        corr_dict = _get_2d_3d_corr(f'{input_path}/sparse/0')
        self.depths = {}
        for imagefile in corr_dict.keys():
            pixels = corr_dict[imagefile]['pixel_2d']
            sfm_3d = corr_dict[imagefile]['coordinate_3D']
            R, t = corr_dict[imagefile]['R'], corr_dict[imagefile]['t']
            camera_3d = R@sfm_3d.T + t 
            camera_depths = camera_3d[2, :]
            pred_depth = _predict_pixelwise_depth(f'{input_path}/images/{imagefile}', pixels, camera_depths)
            self.depths[imagefile] = pred_depth
    
    def generate_pseudo_view(self, view_cam): 
        """
        Returns pseudo views and pseudo cameras
        """
        original_image = self.tensor_to_rgb_image(view_cam.original_image)
        # generate pseudo camera
        q = rotmat2qvec(view_cam.R)
        R_new, t_new = _perturb_q(view_cam.T.reshape((-1,1)), q, trans_mag=1.0, q_mag=0.01)
        pseudo_cam = PseudoCamera(R_new, t_new.flatten(), view_cam.FoVx,
                          view_cam.FoVy,
                          view_cam.image_width,
                          view_cam.image_height)
        # warped image 
        I_pseudo = _warp_weighted_3dgs(original_image, view_cam.R, view_cam.T.reshape((-1,1)),
                               self.depths[view_cam.image_name], 
                               R_new, t_new, view_cam.FoVx, view_cam.FoVy,
                               view_cam.image_width, view_cam.image_height
                            )
        I_pseudo = I_pseudo/np.max(I_pseudo)
        
        # mask generation
        mask = _generate_inpaint_mask(I_pseudo)
        
        # tensor conversion
        I_pseudo = torch.tensor(np.transpose(I_pseudo, (2,0,1)), device=view_cam.original_image.device)
        mask = torch.tensor(mask, device=view_cam.original_image.device)

        return pseudo_cam, I_pseudo, 1.0-mask

    def tensor_to_rgb_image(self, tensor):
        """
        Convert a PyTorch image tensor (values in [0,1]) to a NumPy RGB image (uint8, [0,255]).
        Automatically detaches and moves to CPU.
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().clamp(0, 1)
        else:
            raise TypeError("Input must be a torch.Tensor")

        # Handle (C,H,W) or (1,C,H,W)
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return img

if __name__=="__main__": 
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--n_new_images", default=3, type=int)
    parser.add_argument("--stereo_fusion", action='store_true')

    args = parser.parse_args()

    # the input path should contain train and test folder - inside train there will be images and sparse/0 folder
    input_path = Path(args.input_path).expanduser()  
    input_path = input_path.resolve()
    trainpath = input_path/'train'
    testpath = input_path/'test'
    if not trainpath.exists() or not testpath.exists(): 
        raise FileNotFoundError('FILE NOT EXISTS')
    
    # This folder will contain all the files after running COLMAP
    args = parser.parse_args()
    output_path = Path(args.output_path).expanduser()  
    output_path = output_path.resolve()
    Path.mkdir(output_path, exist_ok=True, parents=True)

    # Create temporary folder for storing augmented images before colmap
    tempfolder = Path.cwd()/'temp'
    if tempfolder.exists():
        shutil.rmtree(tempfolder)
    Path.mkdir(tempfolder/'train', parents=True)
    Path.mkdir(tempfolder/'test', parents=True)
    exit1 = os.system(f'cp -r {args.input_path}/train/images/* temp/train')
    exit2 = os.system(f'cp -r {args.input_path}/test/images/* temp/test')
    if exit1 != 0 or exit2 != 0: 
        print(f"Something went wrong {exit1} {exit2}")
        sys.exit(1)
    
    # Now run augmentaion pipeline
    run_augment_pipeline(input_path=input_path/'train', output_path=tempfolder/'train', n_new_images=args.n_new_images)
    
    # Now run COLMAP again on the augmented dataset
    cmd = f'python convert.py -s {str(tempfolder)} -d {str(output_path)}'
    if args.stereo_fusion:
        cmd = cmd + ' --stereo_fusion'
    exitcode = os.system(cmd)
    if exitcode != 0: 
        print(f'Error in COLMAP {exitcode}')
        sys.exit(1)
    shutil.rmtree(tempfolder)



    
    

    


        



