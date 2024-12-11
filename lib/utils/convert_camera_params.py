import pickle
import numpy as np
import torch
import cv2
#from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

def load_K_Rt_from_P(filename=None, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    print(f"P shape: {P.shape}")
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]  # Normalize K

    # Intrinsics matrix
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    # Pose matrix (extrinsics)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def convert_camera_params(P, scale_x=1080, scale_y=1920):
    if isinstance(P, torch.Tensor):
        print(f"Converting P from torch.Tensor to np.ndarray")
        P = P.detach().cpu().numpy()

    if P.shape == (4, 4):
        P = P[:3, :]
    
    intrinsics, pose = load_K_Rt_from_P(P=P)
    pose_inv = np.linalg.inv(pose)
      
    intrinsics_modified = intrinsics.copy()

    intrinsics_modified[0, 0] /= 2  # Halve fx
    intrinsics_modified[1, 1] /= 2  # Halve fy
    intrinsics_modified[0, 2] /= 2  # Halve cx
    intrinsics_modified[1, 2] /= 2  # Halve cy
    intrinsics_modified[0, 2] += 0.5  # Adjust cx
    intrinsics_modified[1, 2] += 0.5  # Adjust cy
    
    scaling_matrix = np.array([[scale_x, 0, 0, 0],
                           [0, scale_y, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    projection_matrix = scaling_matrix @ intrinsics_modified @ pose_inv

    return projection_matrix


def scale_convert_camera_params(P, new_width=512, new_height=512, original_width=1080, original_height=1920):
    if isinstance(P, torch.Tensor):
        print(f"Converting P from torch.Tensor to np.ndarray")
        P = P.detach().cpu().numpy()

    if P.shape == (4, 4):
        P = P[:3, :]

    intrinsics, pose = load_K_Rt_from_P(P=P)
    pose_inv = np.linalg.inv(pose)
        
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    
    intrinsics_modified = intrinsics.copy()

    intrinsics_modified[0, 0] *= scale_x  # scale fx
    intrinsics_modified[1, 1] *= scale_y  # scale fy
    intrinsics_modified[0, 2] *= scale_x  # scale cx
    intrinsics_modified[1, 2] *= scale_y  # scale cy
    
    print(f"Modified Intrinsics scaled and so: \n{intrinsics_modified}")
    
    scaling_matrix = np.array([[scale_x, 0, 0, 0],
                           [0, scale_y, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    projection_matrix = scaling_matrix @ intrinsics_modified @ pose_inv

    return projection_matrix


def convert_opencv_to_pytorch3d(cam_intr, cam_extr, render_size):
    """
    Converts OpenCV camera parameters to PyTorch3D camera objects using cameras_from_opencv_projection.
    
    Args:
        cam_intr (np.ndarray or torch.Tensor): The camera intrinsic matrix.
        cam_extr (np.ndarray or torch.Tensor): The camera extrinsic matrix (inverse of pose).
        render_size (int): The desired render size for the images.
    
    Returns:
        CamerasBase: The camera object compatible with PyTorch3D rendering.
    """
    print(f"\nDEBUG for convert_opencv_to_pytorch3d:")
    print(f"\nType cam intrin: {type(cam_intr)}")
    print(f"\nType cam extrin: {type(cam_extr)}")

    if not isinstance(cam_intr, torch.Tensor):
        cam_intr = torch.from_numpy(cam_intr)
    if not isinstance(cam_extr, torch.Tensor):
        cam_extr = torch.from_numpy(cam_extr)

    print(f"cam_intr.shape: {cam_intr.shape}")
    if cam_intr.shape == (4, 4):
        K = cam_intr[:3, :3]
    else:
        K = cam_intr
    print("Extracted Intrinsic Matrix K:")
    print(K)
    # Camera parameters prep

    # cameras = cameras_from_opencv_projection(
    #         camera_matrix=cam_intr.unsqueeze(0),
    #         R=cam_extr[:3, :3].unsqueeze(0),
    #         tvec=cam_extr[:3, 3].unsqueeze(0),
    #         image_size=torch.tensor([self.render_size, self.render_size]).unsqueeze(0)
    #     ).cuda()

    camera_matrix = K.unsqueeze(0).float()
    #camera_matrix = cam_intr.unsqueeze(0),
    # print(f"camera_matrix.shape: {camera_matrix.shape}")
    R = cam_extr[:3, :3].unsqueeze(0).float()
    # print(f"R shape: {R.shape}")
    tvec = cam_extr[:3, 3].unsqueeze(0).float()
    # print(f" tvec shape: {tvec.shape}")
    image_size = torch.tensor([render_size, render_size]).unsqueeze(0)
    
    cameras = cameras_from_opencv_projection(
        camera_matrix=camera_matrix,
        R=R,
        tvec=tvec,
        image_size=image_size
    ).cuda()

    return cameras