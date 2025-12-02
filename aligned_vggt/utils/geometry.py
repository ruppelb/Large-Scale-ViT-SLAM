import torch
from vggt.utils.geometry import closed_form_inverse_se3

def unproject_depth_map_to_point_map(depth_map: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    
    B, S, H, W, _ = depth_map.shape
    device = depth_map.device

    with torch.amp.autocast("cuda", enabled=False): #with torch.cuda.amp.autocast(enabled=False):

        if torch.isnan(depth_map).any() or torch.isinf(depth_map).any():
            print("\nNan input depth map")

        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            print("\nNan input poses")

        if torch.isnan(intrinsics).any() or torch.isinf(intrinsics).any():
            print("\nNan input intr")

        pixel_coords = generate_3D_pixel_grid(H,W,device).view(-1, 3)  # [H*W, 3]

        if torch.isnan(pixel_coords).any() or torch.isinf(pixel_coords).any():
            print("\nNan after generating grid")

        #valid_mask = torch.logical_and(intrinsics_cam[...,0, 0]> 1e-8,intrinsics_cam[...,1, 1]> 1e-8, depth_map > 1e-8)
        # Unproject to rays
        intrinsics_cam_inv = torch.inverse(intrinsics)  # [B,S,3,3]

        if torch.isnan(intrinsics_cam_inv).any() or torch.isinf(intrinsics_cam_inv).any():
            print("\nNan after computing inverse intrinsics")

        pixel_coords = pixel_coords.t().unsqueeze(0).unsqueeze(0)  # [1,1,3,H*W]
        rays = intrinsics_cam_inv @ pixel_coords  # [B,S,3,H*W]
        rays = rays.permute(0,1,3,2).contiguous()  # [B,S,H*W,3]

        if torch.isnan(rays).any() or torch.isinf(rays).any():
            print("\nNan after unprojecting rays")

        # Scale by depth
        depths_flat = depth_map.view(B, S, -1, 1)  # [B,S,H*W,1]
        cam_coords = rays * depths_flat  # [B,S,H*W,3]

        if torch.isnan(cam_coords).any() or torch.isinf(cam_coords).any():
            print("\nNan after depth scaling")

        # Homogenize camera coords
        cam_coords_h = torch.cat([cam_coords, torch.ones_like(cam_coords[..., :1])], dim=-1)  # [B,S,H*W,4]

        if torch.isnan(cam_coords_h).any() or torch.isinf(cam_coords_h).any():
            print("\nNan after homogenize")

        poses = closed_form_inverse_se3(extrinsics.reshape(B*S,3,4)).reshape(B,S,4,4)

        if torch.isnan(poses).any() or torch.isinf(poses).any():
            print("\nNan after inverting extr")

        # Transform to world
        world_coords = (poses @ cam_coords_h.transpose(-1,-2)).transpose(-1,-2)  # [B,S,H*W,4]

        if torch.isnan(world_coords).any() or torch.isinf(world_coords).any():
            print("\nNan after applying extr")

        world_coords = world_coords[..., :3] / world_coords[..., 3:]

        if torch.isnan(world_coords).any() or torch.isinf(world_coords).any():
            print("\nNan after dividing by 4")

        world_coords = world_coords.view(B, S, H, W, 3)

    return world_coords

def project_world_points_to_pixels(world_points: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor) -> torch.Tensor:
    B, S, H, W, _ = world_points.shape

    with torch.amp.autocast("cuda", enabled=False): #with torch.cuda.amp.autocast(enabled=False):

        if torch.isnan(world_points).any() or torch.isinf(world_points).any():
            print("Nan input points")

        if torch.isnan(extrinsics_cam).any() or torch.isinf(extrinsics_cam).any():
            print("Nan input extr")

        if torch.isnan(intrinsics_cam).any() or torch.isinf(intrinsics_cam).any():
            print("Nan input intr")

        # Homogenize camera coords
        world_points_h = torch.cat([world_points, torch.ones_like(world_points[..., :1])], dim=-1).view(B,S,-1, 4)  # [B,S,H*W,4]

        if torch.isnan(world_points_h).any() or torch.isinf(world_points_h).any():
            print("Nan after homogenize")

        cam_points = (extrinsics_cam @ world_points_h.transpose(-1,-2)).transpose(-1,-2) # [B,S,H*W,3]

        if torch.isnan(cam_points).any() or torch.isinf(cam_points).any():
            print("Nan after extrinsics applied")

        pixels = (intrinsics_cam @ cam_points.transpose(-1,-2)).transpose(-1,-2)

        if torch.isnan(pixels).any() or torch.isinf(pixels).any():
            print("Nan after intrinsics applied")

        valid_mask = torch.logical_and(torch.abs(pixels[..., 2]) > 1e-8, torch.abs(pixels[..., 2]) < 100.0)

        pixels[valid_mask] = pixels[valid_mask] / torch.abs(pixels[valid_mask][..., 2]).unsqueeze(-1) #keep depth coordinate, so we can penalize if projected point is behind cam
        
        if torch.isnan(pixels).any() or torch.isinf(pixels).any():
            print("Nan after depth division")
        
        pixels = pixels.view(B,S,H,W,3)
        valid_mask = valid_mask.view(B,S,H,W)

    return pixels, valid_mask

def compute_relative_poses(extrinsics: torch.Tensor, offset : int = 1, toNext = True) -> torch.Tensor:
    """
    Compute relative poses between offset frames.

    Args:
        extrinsics: [B, S, 3, 4]  world->camera matrices

    Returns:
        rel_poses: [B, S-offset, 3, 4]  relative poses (s -> s+offset)
    """
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device

    if S <= offset:
        raise Exception("To small sequence for offset")

    with torch.amp.autocast("cuda", enabled=False):# with torch.cuda.amp.autocast(enabled=False):
        # Convert to homogeneous [B,S,4,4]
        w2c = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)
        w2c[:, :, :3, :4] = extrinsics

        # Inverse: camera->world
        c2w = torch.inverse(w2c)  # [B,S,4,4]

        if toNext:
            #s -> s+offset
            # Relative transform
            rel = w2c[:,offset:] @ c2w[:, :-offset]  # [B,S-offset,4,4]
        else:
            #s+offset -> s
            rel = w2c[:,:-offset] @ c2w[:, offset:]  # [B,S-offset,4,4]

    return rel[:, :, :3, :4]

def generate_3D_pixel_grid(H,W,device):
    # Generate grid of pixel coordinates
    u, v = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy"
    )

    return torch.stack((u, v, torch.ones_like(u)), dim=-1).float()