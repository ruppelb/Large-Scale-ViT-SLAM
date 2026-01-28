import torch
from vggt.vggt.utils.geometry import closed_form_inverse_se3

def averagePoseEncodings(pose_encodings: torch.Tensor) -> torch.Tensor:
    """
    Average given pose encodings. Quaternions are averaged using using Markley's method (without uniform weights)
    
    Args:
        pose_encodings (torch.Tensor): shape (B, N, 7), batch of sets of pose_encodings.
    
    Returns:
        torch.Tensor: shape (B, 4), average pose encodings.
    """
    B, N, _ = pose_encodings.shape

    translations = pose_encodings[...,:3]
    quaternions = pose_encodings[..., 3:7]

    # average translations
    avg_translation = torch.mean(translations,dim=1, keepdim=True)

    # normalize quaternions just in case
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    weights = (torch.ones(B, N, device=quaternions.device, dtype=quaternions.dtype) / N).unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    outer_products = quaternions.unsqueeze(-1) * quaternions.unsqueeze(-2) # (B, N, 4, 4)
    M = (weights * outer_products).sum(dim=1)  # (B, 4, 4)

    # compute eigenvalues/vectors of M
    _, eigvecs = torch.linalg.eigh(M)  # (B, 4), (B, 4, 4)
    
    # select eigenvector with largest eigenvalue
    max_eigvec = eigvecs[..., -1]  # (B, 4)

    # normalize
    avg_quat = max_eigvec / max_eigvec.norm(dim=-1, keepdim=True)

    return torch.cat([avg_translation, avg_quat.unsqueeze(1)], dim=-1).float()

def unproject_depth_map_to_point_map(depth_map: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Unproject depth map to 3D world coordinates.
    Args:
        depth_map: (B, S, H, W) depth maps in camera space
        extrinsics: (B, S, 3, 4) world-to-camera matrices
        intrinsics: (B, S, 3, 3) camera intrinsics matrices
    Returns:
        world_coords: (B, S, H, W, 3) 3D world coordinates
    """ 
    B, S, H, W, _ = depth_map.shape
    device = depth_map.device

    with torch.amp.autocast("cuda", enabled=False):
        pixel_coords = generate_3D_pixel_grid(H,W,device).view(-1, 3)  # (H*W, 3)

        # unproject to rays
        intrinsics_cam_inv = torch.inverse(intrinsics)  # (B, S, 3, 3)
        pixel_coords = pixel_coords.t().unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H*W)
        rays = intrinsics_cam_inv @ pixel_coords  # (B, S, 3, H*W)
        rays = rays.permute(0,1,3,2).contiguous()  # (B, S, H*W, 3)

        # scale by depth
        depths_flat = depth_map.view(B, S, -1, 1)  # (B, S, H*W, 1)
        cam_coords = rays * depths_flat  # (B, S, H*W, 3)

        # homogenize camera coords
        cam_coords_h = torch.cat([cam_coords, torch.ones_like(cam_coords[..., :1])], dim=-1)  # (B, S, H*W, 4)
        poses = closed_form_inverse_se3(extrinsics.reshape(B*S,3,4)).reshape(B,S,4,4)

        # transform to world
        world_coords = (poses @ cam_coords_h.transpose(-1,-2)).transpose(-1,-2)  # (B, S, H*W, 4)
        world_coords = world_coords[..., :3] / world_coords[..., 3:]
        world_coords = world_coords.view(B, S, H, W, 3)

    return world_coords

def project_world_points_to_pixels(world_points: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor) -> torch.Tensor:
    """
    Project 3D world points to pixel coordinates.
    Args:
        world_points: (B, S, H, W, 3) 3D world coordinates
        extrinsics_cam: (B, S, 3, 4) world-to-camera matrices
        intrinsics_cam: (B, S, 3, 3) camera intrinsics matrices
    Returns:
        pixels: (B, S, H, W, 3) pixel coordinates in homogeneous form (u, v, w)
        valid_mask: (B, S, H, W) boolean mask indicating valid projections
    """

    B, S, H, W, _ = world_points.shape

    with torch.amp.autocast("cuda", enabled=False):
        # homogenize camera coords
        world_points_h = torch.cat([world_points, torch.ones_like(world_points[..., :1])], dim=-1).view(B,S,-1, 4)  # (B, S, H*W, 4)

        # transform to camera
        cam_points = (extrinsics_cam @ world_points_h.transpose(-1,-2)).transpose(-1,-2) # (B, S, H*W, 3)

        # project to pixels
        pixels = (intrinsics_cam @ cam_points.transpose(-1,-2)).transpose(-1,-2)
        valid_mask = torch.logical_and(torch.abs(pixels[..., 2]) > 1e-8, torch.abs(pixels[..., 2]) < 100.0)
        pixels[valid_mask] = pixels[valid_mask] / torch.abs(pixels[valid_mask][..., 2]).unsqueeze(-1) # keep depth coordinate, so we can penalize if projected point is behind cam
        pixels = pixels.view(B,S,H,W,3)
        valid_mask = valid_mask.view(B,S,H,W)

    return pixels, valid_mask

def compute_relative_poses(extrinsics: torch.Tensor, offset : int = 1, toNext : bool = True) -> torch.Tensor:
    """
    Compute relative poses between offset frames.

    Args:
        extrinsics: (B, S, 3, 4)  world-to-camera matrices for a sequence of S frames
        offset: int, offset between frames to compute relative poses (for offset=1, compute between consecutive frames)
        toNext: bool, if True compute poses from s -> s+offset, else s+offset -> s

    Returns:
        rel_poses: (B, S-offset, 3, 4) relative poses between frames
    """
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device

    if S <= offset:
        raise Exception("To small sequence for offset")

    with torch.amp.autocast("cuda", enabled=False):
        # convert to homogeneous (B, S, 4, 4)
        w2c = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)
        w2c[:, :, :3, :4] = extrinsics

        # Inverse: camera->world
        c2w = torch.inverse(w2c)  # (B, S, 4, 4)

        if toNext:
            # s -> s+offset
            rel = w2c[:,offset:] @ c2w[:, :-offset]  # (B, S-offset, 4, 4)
        else:
            # s+offset -> s
            rel = w2c[:,:-offset] @ c2w[:, offset:]  # (B, S-offset, 4, 4)

    return rel[:, :, :3, :4]

def generate_3D_pixel_grid(H: int, W: int, device) -> torch.Tensor:
    """
    Generate a grid of pixel coordinates in homogeneous form.
    Args:
        H: int, height of the image
        W: int, width of the image
        device: torch.device, device to create the tensor on
    Returns:
        pixel_grid: (H, W, 3) tensor of pixel coordinates in homogeneous form
    """
    u, v = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy"
    )

    return torch.stack((u, v, torch.ones_like(u)), dim=-1).float()