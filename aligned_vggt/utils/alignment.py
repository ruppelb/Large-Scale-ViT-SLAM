import numpy as np
import torch
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from vggt.vggt.utils.geometry import closed_form_inverse_se3

def umeyama(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991

    Args:
        x: observed data (3xn array) (should be aligned to y)
        y: reference data (3xn array)
    Returns:
        r: rotation matrix (3x3)
        t: translation vector (3x1)
        s: scale (float)
    """

    if x.shape != y.shape:
        assert False, "x shape not equal to y shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s))
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def methodOfHorn(model: np.ndarray, data: np.ndarray, align_scale: bool=True) -> tuple:
    """
    Align two trajectories using the method of Horn (closed-form).
    Adapted from https://github.com/raulmur/evaluate_ate_scale

    Args:
        model: first trajectory (3xn) (should be aligned to second)
        data: second trajectory (3xn)
        align_scale: whether to align scale or not (bool)

    Output:
        rot: rotation matrix (3x3)
        trans: translation vector (3x1)
        s: scale (float)
    """

    if model.shape != data.shape:
        assert False, "model shape not equal to data shape"

    model = np.asmatrix(model)
    data = np.asmatrix(data)

    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    if align_scale:
        rotmodel = rot*model_zerocentered
        dots = 0.0
        norms = 0.0

        for column in range(data_zerocentered.shape[1]):
            dots += np.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
            normi = np.linalg.norm(model_zerocentered[:,column])
            norms += normi*normi

        s = (dots/norms).item()
    else:
        s = 1.0
    
    trans = data.mean(1) - s*rot * model.mean(1)

    return np.asarray(rot), np.asarray(trans).squeeze(-1), np.asarray(s)

def scale_lse_solver(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes optimal scaling factor to align two sets of registered points.

    Args:
        x: observed data (nx3 array)
        y: reference data (nx3 array)
    Returns:
        scale: scaling factor (float)
    """

    if x.shape != y.shape:
        assert False, "x shape not equal to y shape"

    scale = np.sum(x * y)/np.sum(x ** 2)

    return np.abs(scale) #avoid negative scales

def per_frame_scale_alignment_from_poses(predictions: dict, batch: dict) -> None:
    """
    Compute one robust L1-optimal scale per frame.
    Solves for each frame s:
        a_s = argmin_a  sum_i w_i * |a * x_i - y_i|
    where i indexes 3D points in frame s.
    Args:
        predictions: dict containing predicted pose encodings under key "pose_enc"
        batch: dict containing ground truth extrinsics under key "extrinsics"
    Returns:
        predictions: dict with in-place modified pose encodings and added key "alignment_scales"
    """
    B,S = batch["extrinsics"].shape[:2]

    gt_positions = batch["extrinsics"][...,:3,3].detach().cpu().numpy()
    pred_positions = predictions["pose_enc"][...,:3].detach().cpu().numpy() # (B, S, 3)

    for b in range(B):
        frame_scales = []
        for s in range(S):
            if s == 0:
                frame_scale = 1.0 # can't compute lse scale factor since first frame has 0 translation
            else:
                frame_scale = scale_lse_solver(pred_positions[b,s],gt_positions[b,s])

            # inplace operations
            predictions["pose_enc"][b,s,:3] *= frame_scale
            frame_scales.append(frame_scale)
            if "depth" in predictions:
                predictions["depth"][b,s] *= frame_scale

            if "world_points" in predictions:
                predictions["world_points"][b,s] *= frame_scale
 
    predictions["alignment_scales"] = frame_scales

def per_chunk_scale_alignment_from_poses(predictions: dict, batch: dict) -> None:
    """
    Compute one robust L1-optimal scale per chunk.
    Solves for each chunk c:
        a_c = argmin_a  sum_i w_i * |a * x_i - y_i|
    where i indexes all 3D points in chunk c.
    Args:
        predictions: dict containing predicted pose encodings under key "pose_enc"
        batch: dict containing ground truth extrinsics under key "extrinsics"
    Returns:
        predictions: dict with in-place modified pose encodings and added key "alignment_scales_per_chunk"
    """
    C = len(batch["extrinsics"])
    B = batch["extrinsics"][0].shape[0]

    chunk_scales = []
    for c in range(C):
        gt_positions_chunk = batch["extrinsics"][c][...,:3,3].detach().cpu().numpy()
        pred_positions_chunk = predictions["pose_enc"][c][...,:3].detach().cpu().numpy() # (B, S, 3)
        
        batch_scales = []
        for b in range(B):
            batch_scale = scale_lse_solver(pred_positions_chunk[b],gt_positions_chunk[b])

            predictions["pose_enc"][c][b,:,:3] *= batch_scale
            batch_scales.append(batch_scale)

            if b == B-1:
                chunk_scales.append(torch.tensor(batch_scales))
            
            if "depth" in predictions:
                predictions["depth"][c][b,...] *= batch_scale

            if "world_points" in predictions:
                predictions["world_points"][c][b,...] *= batch_scale
    
    predictions["alignment_scales_per_chunk"] = chunk_scales


def scale_alignment_from_poses(predictions: dict, batch: dict, seq_width: int=-1) -> None:
    """
    Compute one robust L1-optimal scale per batch element (shape [B]).
    Solves for each batch b:
        a_b = argmin_a  sum_i w_i * |a * x_i - y_i|
    where i indexes S frames.
    The Scale alignment is the same wether we compute it over c2w or w2c, so we use w2c here.
    Args:
        predictions: dict containing predicted pose encodings under key "pose_enc"
        batch: dict containing ground truth extrinsics under key "extrinsics"
        seq_width: number of frames to consider for alignment (int, default -1 uses full sequence)
    Returns:
        predictions: dict with in-place modified pose encodings and added key "alignment_scales"
    """
    B = batch["extrinsics"].shape[0]

    if seq_width == -1:
        seq_width = batch["extrinsics"].shape[1]

    batch_scales = []
    gt_positions = batch["extrinsics"][:,:seq_width,:3,3].detach().cpu().numpy()
    pred_positions = predictions["pose_enc"][:,:seq_width,:3].detach().cpu().numpy() #B,S,3

    for b in range(B):
        batch_scale = scale_lse_solver(pred_positions[b],gt_positions[b])

        # inplace operations
        predictions["pose_enc"][b,:,:3] *= batch_scale

        batch_scales.append(batch_scale)
        if "depth" in predictions:
            predictions["depth"][b,...] *= batch_scale

        if "world_points" in predictions:
            predictions["world_points"][b,...] *= batch_scale
    
    predictions["alignment_scales"] = batch_scales

def scale_align_from_depths(predictions: dict, batch: dict) -> None:
    """
    Compute one robust L1-optimal scale per batch element (shape [B]).

    Solves for each batch b:
        a_b = argmin_a  sum_i w_i * |a * x_i - y_i|
    where i indexes S*H*W pixels.

    Args:
        predictions: dict containing predicted depths under key "depth" and depth confidence under key "depth conf"
        batch: dict containing ground truth depths under key "depths" and point masks under key "point_masks"
    Returns:
        predictions: dict with in-place modified depths, world points and pose encodings, and added key "alignment_scales"
    """
    d_pred = predictions["depth"]
    conf = predictions["depth_conf"]
    d_gt = batch["depths"]
    mask = batch["point_masks"]

    # unpack shapes
    B, S, H, W, _ = d_pred.shape
    N = S * H * W

    # flatten spatial + sequence dims → (B, N)
    x = d_pred.reshape(B, N)
    y = d_gt.reshape(B, N)
    m = mask.reshape(B, N).float()
    w_conf = conf.reshape(B, N)

    with torch.amp.autocast("cuda", enabled=False):
        with torch.no_grad():
            # weight with inv depth (clamp very small values to fraction of weighted mean to avoid huge contributions)
            valid_depth = y * m
            sum_valid = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_depth = (valid_depth.sum(dim=-1, keepdim=True) / sum_valid)  # (B, 1)
            min_depth = 0.1 * mean_depth  # (B, 1)

            y_clamped = torch.max(y, min_depth)  # clamp each batch individually
            w_depth = 1.0 / y_clamped.clamp_min(1e-6)

            # total weights
            w = m * w_conf * w_depth  # (B, N)

            # ---- Weighted median L1 solver ----
            # avoid negative depths (theoretically depths returned by the model should always be positive, but just in case)
            sign = torch.sign(x)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            x_pos = x * sign
            y_pos = y * sign

            r = y_pos / x_pos.clamp_min(1e-6)     # ratios
            w_eff = w * x_pos                     # effective L1 weight

            # sort ratios
            r_sorted, idx = torch.sort(r, dim=-1)
            w_sorted = torch.gather(w_eff, -1, idx)

            cumsum = w_sorted.cumsum(-1)
            total = cumsum[:, -1:]
            target = 0.5 * total

            # weighted median index
            idx_med = torch.searchsorted(cumsum, target, side="left")
            idx_med = idx_med.clamp(max=N-1)

            scales = torch.gather(r_sorted, -1, idx_med).squeeze(-1)  # (B, )

        # avoid negative scales
        scales[scales <= 0] *= -1

        # inplace operations
        predictions["depth"] *= scales[:, None, None, None, None]

        if "world_points" in predictions:
            predictions["world_points"] *= scales[:, None, None, None, None]

        if "pose_enc" in predictions:
            predictions["pose_enc"][...,:3] *= scales[:, None, None]

        predictions["alignment_scales"] = [scales[b].item() for b in range(B)]

def umeyama_alignment_from_poses(predictions: dict, batch: dict, seq_width: int) -> None:
    """
    Compute Sim(3) alignment from predicted to ground truth poses using Umeyama method.
    Args:
        predictions: dict containing predicted pose encodings under key "pose_enc"
        batch: dict containing ground truth extrinsics under key "extrinsics"
        seq_width: number of frames to consider for alignment (int)
    Returns:
        None (in-place modification of pred dict)
    """
    B = batch["extrinsics"].shape[0]
    
    # avoid autocasting (closed_form_inverse_se3 or apply_sim3_alignment will autocast to mixed precision if enabled)
    with torch.amp.autocast("cuda", enabled=False):

        # convert w2c to c2w
        gt_poses = closed_form_inverse_se3(batch["extrinsics"][:,:seq_width].reshape(B*seq_width,3,4)).reshape(B,seq_width,4,4).detach().cpu().numpy()
        gt_positions = gt_poses[...,:3,3] #B,S,3

        # convert w2c to c2w
        pred_extr, _ = pose_encoding_to_extri_intri(predictions["pose_enc"][:,:seq_width],batch["images"].shape[-2:])
        pred_positions = closed_form_inverse_se3(pred_extr.reshape(B*seq_width,3,4)).reshape(B,seq_width,4,4)[...,:3,3].detach().cpu().numpy() # (B, S, 3)
        
        batch_transforms = []
        batch_scales = []
        for b in range(B):
            r, t, c = umeyama(pred_positions[b].transpose(),gt_positions[b].transpose())

            # convert to 4x4 matrices
            pose = np.pad(r, ((0, 1), (0, 1)), mode="constant")
            pose[:3, 3] = t
            pose[3, 3] = 1.

            batch_transforms.append(pose)
            batch_scales.append(c)

        batch_transforms = np.array(batch_transforms)
        batch_scales = np.array(batch_scales)

        # apply alignments
        adjusted_pose_encs, adjusted_points, adjusted_depth = apply_sim3_alignment(batch_transforms,batch_scales, [predictions["pose_enc"]],batch["images"].shape[-2:],predictions["world_points"] if "world_points" in predictions else None, predictions["depth"] if "depth" in predictions else None)
        predictions["pose_enc"] = adjusted_pose_encs[0]
        if "world_points" in predictions:
            predictions["world_points"] = adjusted_points
        if "depth" in predictions:
            predictions["depth"] = adjusted_depth

def umeyama_alignment_from_points(pred_points, pred_confidence, target_points, target_point_mask, confidence_threshold: int) -> tuple:
    """
    Compute Sim(3) alignment from predicted to target 3D points using Umeyama method.
    Args:
        pred_points (tensor or numpy array): predicted 3D points (B, 3, H, W)
        pred_confidence (tensor or numpy array): predicted confidence for 3D points (B, H, W)
        target_points (tensor or numpy array): target 3D points (B, 3, H, W)
        target_point_mask (tensor or numpy array): binary mask indicating valid target points (B, H, W)
        confidence_threshold (int): percentile threshold for predicted confidence (0-100)
    Returns:
        batch_poses: numpy array of Sim(3) transformation matrices (B, 4, 4)
        batch_cs: numpy array of scale factors (B,)
    """
    # convert to numpy
    if isinstance(pred_points, torch.Tensor):
        pred_points = pred_points.detach().cpu().numpy()
        
    if isinstance(target_points, torch.Tensor):
        target_points = target_points.detach().cpu().numpy()
    
    if isinstance(pred_confidence, torch.Tensor):
        pred_confidence = pred_confidence.detach().cpu().numpy()

    if isinstance(target_point_mask, torch.Tensor):
        target_point_mask = target_point_mask.detach().cpu().numpy()

    B = pred_points.shape[0]
    batch_poses = []
    batch_cs = []
    for b in range(B):

        batch_points = pred_points[b] # (3, H, W)
        batch_target_points = target_points[b] # (3, H, W)
        batch_pred_confidence = pred_confidence[b] # (H, W)
        batch_target_point_mask = target_point_mask[b] # (H, W)

        # percentage threshold to conf value
        pred_conf_threshold = np.percentile(batch_pred_confidence, confidence_threshold)
        conf_mask = (batch_target_point_mask > 0) & (batch_pred_confidence >= pred_conf_threshold) & (batch_pred_confidence > 1e-5)

        # select points with high confidence
        batch_points = batch_points[conf_mask]
        batch_target_points = batch_target_points[conf_mask]

        r, t, c = umeyama(batch_points.reshape(3, -1),batch_target_points.reshape(3, -1))

        # convert to 4x4 matrices
        pose = np.pad(r, ((0, 1), (0, 1)), mode="constant")
        pose[:3, 3] = t
        pose[3, 3] = 1.

        batch_poses.append(pose)
        batch_cs.append(c)

    return np.array(batch_poses), np.array(batch_cs)

def apply_sim3_alignment_on_dict(pred: dict, images_size: tuple, alignment_poses: np.ndarray, alignment_scales: np.ndarray) -> None:
    """
    Apply Sim(3) alignment to all relevant entries in prediction dict.
    Args:
        pred: dict containing predicted pose encodings under key "pose_enc", and optionally "
                world_points" and "depth"
        images_size: size of input images (H, W)
        alignment_poses: numpy array of Sim(3) transformation matrices (B, 4, 4)
        alignment_scales: numpy array of scale factors (B,)
    Returns:
        None (in-place modification of pred dict)
    """

    adjusted_pose_encs, adjusted_points, adjusted_depth = apply_sim3_alignment(alignment_poses, alignment_scales, pred["pose_enc"], images_size, pred["world_points"] if "world_points" in pred else None, pred["depth"] if "depth" in pred else None)
    pred["pose_enc"] = adjusted_pose_encs
    if "world_points" in pred:
        pred["world_points"] = adjusted_points
    if "depth" in pred:
        pred["depth"] = adjusted_depth
    

def apply_sim3_alignment(alignment_transforms: np.ndarray, alignment_scales: np.ndarray, pose_encodings: torch.Tensor, images_size: tuple, points: torch.Tensor=None, depths: torch.Tensor=None) -> tuple:
    """
    Apply Sim(3) alignment to pose encodings, 3D points and depths.
    Args:
        alignment_transforms: numpy array of Sim(3) transformation matrices (B, 4, 4)
        alignment_scales: numpy array of scale factors (B,)
        pose_encodings: tensor of pose encodings (B, S, 6)
        images_size: size of input images (H, W)
        points: optional tensor of 3D points (B, S, H, W, 3)
        depths: optional tensor of depths (B, S, H, W, 1)
    Returns:
        pose_encodings: tensor of aligned pose encodings (B, S, 6)
        points: tensor of aligned 3D points (B, S, H, W, 3)
        depths: tensor of aligned depths (B, S, H, W, 1)
    """
    
    B = alignment_transforms.shape[0]

    with torch.amp.autocast("cuda", enabled=False):

        #prepare scales
        alignment_scales = torch.tensor(alignment_scales).float().to(pose_encodings.device)

        #prepare se(3) alignment matrices
        alignment_transforms = torch.from_numpy(alignment_transforms).float().to(pose_encodings.device) # (B, 4, 4)

        #apply alignments
        extr, intr = pose_encoding_to_extri_intri(pose_encodings, images_size)

        extr = apply_sim3_alignment_on_w2c(extr, alignment_transforms, alignment_scales)

        pose_encodings = extri_intri_to_pose_encoding(extr, intr, images_size) # (B, S, 6)
            
        if points is not None:
            points = apply_sim3_alignment_on_point_maps(points, alignment_transforms, alignment_scales)

        if depths is not None:
            #we do not need to apply pose alignment to unprojected depth maps, since the alignment pose for camera and 3d points would collapse to identity
            depths *= alignment_scales.view(B, 1, 1, 1, 1) # (B, S, H, W, 1)

    return pose_encodings, points, depths

def apply_sim3_alignment_on_point_maps(point_maps: torch.Tensor, alignment_transforms: torch.Tensor, alignment_scales: torch.Tensor) -> torch.Tensor:
    """
    Apply Sim(3) alignment to 3D point maps.
    Args:
        point_maps: tensor of 3D points (B, S, H, W, 3)
        alignment_transforms: tensor of Sim(3) transformation matrices (B, 4, 4)
        alignment_scales: tensor of scale factors (B,)
    Returns:
        point_maps: tensor of aligned 3D points (B, S, H, W, 3)
    """
    # add batch dimension if we receive non batched input
    if len(point_maps.shape) == 4:
        point_maps = point_maps.unsqueeze(0)
        alignment_transforms = alignment_transforms.unsqueeze(0)
        alignment_scales = alignment_scales.unsqueeze(0)
    
    assert point_maps.shape[0] == alignment_transforms.shape[0] == alignment_scales.shape[0], "Inputs must have matching batch dimension"

    B, S, H, W, _ = point_maps.shape

    with torch.amp.autocast("cuda", enabled=False):
        # apply_scale
        point_maps = point_maps * alignment_scales.view(B,1,1,1,1) # (B, S, H, W, 3)

        # convert points to homogeneous coordinates
        point_maps = torch.cat([point_maps, torch.ones_like(point_maps[..., :1])], dim=-1) # (B, S, H, W, 4)
        point_maps = point_maps.view(B, -1, 4) # (B, S*H*W, 4)

        # prepare alignment matrices
        alignment_transforms = alignment_transforms.unsqueeze(1) # (B, 1, 4, 4)
        alignment_transforms = alignment_transforms.expand(-1, S*H*W, -1, -1) # (B, S*H*W, 4, 4)

        # apply se(3) alignment
        point_maps = torch.matmul(alignment_transforms, point_maps.unsqueeze(-1)).squeeze(-1) # (B, S*H*W, 4)

    return point_maps.view(B, S, H, W, 4)[...,:3] # (B, S, H, W, 3)

def apply_sim3_alignment_on_w2c(extr: torch.Tensor, alignment_transform: torch.Tensor, alignment_scales: torch.Tensor) -> torch.Tensor:
    """
    Apply Sim(3) alignment to world-to-camera extrinsics.
    Args:
        extr: tensor of world-to-camera extrinsics (B, S, 3, 4)
        alignment_transform: tensor of Sim(3) transformation matrices (B, 4, 4)
        alignment_scales: tensor of scale factors (B,)
    Returns:
        extr: tensor of aligned world-to-camera extrinsics (B, S, 3, 4)
    """
    # add batch dimension if we receive non-batched input
    if len(extr.shape) == 3:
        extr = extr.unsqueeze(0)
        alignment_transform = alignment_transform.unsqueeze(0)
        alignment_scales = alignment_scales.unsqueeze(0)

    assert extr.shape[0] == alignment_transform.shape[0] == alignment_scales.shape[0], "Inputs must have matching batch dimension"

    B,S = extr.shape[:2]

    with torch.amp.autocast("cuda", enabled=False):
        # convert to c2w and apply transform
        poses = closed_form_inverse_se3(extr.reshape(B*S,3,4)).reshape(B,S,4,4)
        poses = apply_sim3_alignment_on_c2w(poses,alignment_transform,alignment_scales)
    
        # convert back to w2c
        extr = closed_form_inverse_se3(poses.reshape(B*S,4,4)).reshape(B,S,4,4)

    return extr

def apply_sim3_alignment_on_c2w(poses: torch.Tensor, alignment_transform: torch.Tensor, alignment_scales: torch.Tensor) -> torch.Tensor:
    """
    Apply Sim(3) alignment to camera-to-world poses.
    Args:
        poses: tensor of camera-to-world poses (B, S, 4, 4)
        alignment_transform: tensor of Sim(3) transformation matrices (B, 4, 4)
        alignment_scales: tensor of scale factors (B,)
    Returns:
        poses: tensor of aligned camera-to-world poses (B, S, 4, 4)
    """
    # add batch dimension if we receive non-batched input
    if len(poses.shape) == 3:
        poses = poses.unsqueeze(0)
        alignment_transform = alignment_transform.unsqueeze(0)
        alignment_scales = alignment_scales.unsqueeze(0)

    assert poses.shape[0] == alignment_transform.shape[0] == alignment_scales.shape[0], "Inputs must have matching batch dimension"

    B, S, _, _ = poses.shape

    with torch.amp.autocast("cuda", enabled=False):
        
        # convert to 4x4 matrices
        if poses.shape[-2] != 4:
            poses = torch.nn.functional.pad(poses, (0,0,0,1,0,0), mode="constant")
            poses[:, 3, 3] = 1.

        # apply scale
        poses[:, : , :3, 3] = poses[:, : , :3, 3] * alignment_scales.view(B,1,1) # (B, S, 3)

        # prepare alignment matrices
        alignment_transform = alignment_transform.unsqueeze(1) # (B, 1, 4, 4)
        alignment_transform = alignment_transform.expand(-1,S,-1,-1) # (B, S, 4, 4)

        # apply se(3) alignment
        poses = torch.matmul(alignment_transform, poses) # (B, S, 4, 4)
    return poses

        
       