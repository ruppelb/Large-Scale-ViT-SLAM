from typing import Tuple
import torch
from torchmetrics import Metric
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AbsoluteTrajectoryError(Metric):
    """
    TorchMetrics implementation of Absolute Trajectory Error (ATE).
    Computes RMSE of translational component between predicted and ground truth poses.
    Args:
        detailed (bool): If True, computes additional statistics (mean, median, std, min, max, per-dim RMSE).
    """

    full_state_update = False

    def __init__(self, detailed: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.add_state("errors", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("per_dim_errors", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.detailed = detailed

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with predicted and ground truth positions.
        
        Args:
            preds: predicted positions, shape (N, 4, 4)
            target: ground truth positions, shape (N, 4, 4)
        """

        assert preds.shape == target.shape, "Preds and targets must have the same shape"
        assert preds.shape[-2:] == (4, 4), "Poses must be 4x4 matrices"

        pred_xyz = preds[:, :3, 3]
        gt_xyz = target[:, :3, 3]

        alignment_error = pred_xyz - gt_xyz # (N, 3)
        trans_error = torch.linalg.norm(alignment_error, dim=1) # (N,)

        # store errors as tensors for distributed reduction
        self.errors = torch.cat([self.errors, trans_error.detach().to(self.errors.device)])
        self.per_dim_errors = torch.cat([self.per_dim_errors, alignment_error.detach().to(self.per_dim_errors.device)])

    def compute(self):
        """
        Compute final ATE statistics.
        """
        errors = self.errors # (N,)
        per_dim = self.per_dim_errors # (N, 3)

        rmse = torch.sqrt(torch.mean(errors ** 2)).item()
        rmse_per_dim = torch.sqrt(torch.mean(per_dim ** 2, dim=0)).tolist()

        if not self.detailed:
            return {"ate_rmse": rmse}
        else:
            mean = torch.mean(errors).item()
            median = torch.median(errors).item()
            std = torch.std(errors).item()
            min_val = torch.min(errors).item()
            max_val = torch.max(errors).item()

            return {
                "ate_rmse": rmse,
                "ate_mean": mean,
                "ate_median": median,
                "ate_std": std,
                "ate_min": min_val,
                "ate_max": max_val,
                "ate_rmse_per_dim": rmse_per_dim
            }
    
    def plot(self, preds: torch.Tensor, target: torch.Tensor, title: str = None, outpath: str = None) -> Tuple[dict, str]:
        """
        Optional visualization of GT vs predicted trajectory.
        Computes metric and plots results.
        
        Args:
            preds: predicted positions, shape (N, 4, 4)
            target: ground truth positions, shape (N, 4, 4)
            title: optional title for the plot
            outpath: optional output path to save the plot
        Returns:
            dict: computed ATE metrics
            str: path to saved plot
        """

        assert preds.shape == target.shape, "Preds and targets must have the same shape"
        assert preds.shape[-2:] == (4, 4), "Poses must be 4x4 matrices"

        # compute metric
        pred_xyz = preds[:, :3, 3]
        gt_xyz = target[:, :3, 3]

        alignment_error = pred_xyz - gt_xyz # (N, 3)
        trans_error = torch.linalg.norm(alignment_error, dim=1) # (N,)

        rmse = torch.sqrt(torch.mean(trans_error ** 2)).item()
        rmse_per_dim = torch.sqrt(torch.mean(alignment_error ** 2, dim=0)).tolist()

        # plot results
        pred_xyz = pred_xyz.cpu().numpy()
        gt_xyz = gt_xyz.cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'k-', label="Ground Truth")
        ax.plot(pred_xyz[:, 0], pred_xyz[:, 2], 'b-', label="Prediction")

        for (x1, _, z1), (x2, _, z2) in zip(gt_xyz, pred_xyz):
            ax.plot([x1, x2], [z1, z2], 'r-', alpha=0.5, lw=0.5)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.legend()
        if title:
            fig.suptitle(title,fontsize=10, fontweight='bold')

        ax.set_title(f"ATE RMSE: {rmse:.3f} m, per-dim RMSE: x:{rmse_per_dim[0]:.3f} m, y:{rmse_per_dim[1]:.3f} m, z:{rmse_per_dim[2]:.3f} m",fontsize=10)
        
        if outpath:
            plt.savefig(f"{outpath}traj_ate.png", dpi=300)

            # additionally save data in npy file (for comparing against other methods)
            data={'pred_xyz' : pred_xyz, 'gt_xyz' : gt_xyz, 'rmse' : np.array(rmse), 'rmse_per_dim' : np.array(rmse_per_dim)}
            np.save(f"{outpath}traj_ate.npy",data)
        plt.close(fig)

        return {'ate_rmse': rmse}, f"{outpath}traj_ate.png"

class RelativePoseError(Metric):
    """
    TorchMetrics implementation of Relative Pose Error (RPE). 
    Computes RMSE of translational and rotational components between predicted and ground truth relative poses.
    Args:
        delta (int): frame difference for relative pose computation (default: 1 for consecutive frames)
        detailed (bool): If True, computes additional statistics (mean, median, std, min, max) for both translation and rotation.
    """

    full_state_update = False
    def __init__(self, delta: int = 1, detailed: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.add_state("trans_errors", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("rot_errors", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.detailed = detailed
        self.delta = delta

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predicted and ground truth positions.

        Args:
            preds: predicted poses, shape (N, 4, 4)
            target: ground truth poses, shape (N, 4, 4)
        """
        assert preds.shape == target.shape, "Preds and targets must have the same shape"
        assert preds.shape[-2:] == (4, 4), "Poses must be 4x4 matrices"

        N = preds.shape[0]
        if N <= self.delta:
            return

        pred_rel = torch.linalg.inv(preds[:-self.delta]) @ preds[self.delta:]
        gt_rel = torch.linalg.inv(target[:-self.delta]) @ target[self.delta:]

        err = torch.linalg.inv(gt_rel) @ pred_rel

        # translation error
        trans_error = torch.linalg.norm(err[:, :3, 3], dim=1)

        # rotation error
        rot_trace = torch.sum(torch.diagonal(err[:, :3, :3], dim1=-2, dim2=-1), dim=1)
        rot_error = torch.acos(torch.clamp((rot_trace - 1) / 2, -1.0, 1.0))

        self.trans_errors = torch.cat([self.trans_errors, trans_error.detach().to(self.trans_errors.device)])
        self.rot_errors = torch.cat([self.rot_errors, rot_error.detach().to(self.rot_errors.device)])

    def compute(self):
        """
        Compute final RPE statistics.
        """
        trans = self.trans_errors
        rot = self.rot_errors

        trans_rmse = torch.sqrt(torch.mean(trans ** 2)).item() if len(trans) > 0 else 0.0
        rot_rmse = torch.rad2deg(torch.sqrt(torch.mean(rot ** 2))).item() if len(rot) > 0 else 0.0


        if not self.detailed:
            return {"rpe_trans_rmse": trans_rmse, "rpe_rot_rmse": rot_rmse}
        else:
            trans_mean = torch.mean(trans).item() if len(trans) > 0 else 0.0
            trans_median = torch.median(trans).item() if len(trans) > 0 else 0.0
            trans_std = torch.std(trans).item() if len(trans) > 0 else 0.0
            trans_min = torch.min(trans).item() if len(trans) > 0 else 0.0
            trans_max = torch.max(trans).item() if len(trans) > 0 else 0.0

            rot_mean = torch.rad2deg(torch.mean(rot)).item() if len(rot) > 0 else 0.0
            rot_median = torch.rad2deg(torch.median(rot)).item() if len(rot) > 0 else 0.0
            rot_std = torch.rad2deg(torch.std(rot)).item() if len(rot) > 0 else 0.0
            rot_min = torch.rad2deg(torch.min(rot)).item() if len(rot) > 0 else 0.0
            rot_max = torch.rad2deg(torch.max(rot)).item() if len(rot) > 0 else 0.0

            return {
                "rpe_trans_rmse": trans_rmse,
                "rpe_trans_mean": trans_mean,
                "rpe_trans_median": trans_median,
                "rpe_trans_std":trans_std,
                "rpe_trans_min":trans_min,
                "rpe_trans_max":trans_max,
                "rpe_rot_rmse": rot_rmse,
                "rpe_rot_mean": rot_mean,
                "rpe_rot_median":rot_median,
                "rpe_rot_std":rot_std,
                "rpe_rot_min":rot_min,
                "rpe_rot_max":rot_max,
            }
    
    def plot(self, preds: torch.Tensor, target: torch.Tensor, title: str = None, outpath: str = None):
        """
        Optional visualization RPE errors.
        Computes metric and plots results.

        Args:
            preds: predicted poses, shape (N, 4, 4)
            target: ground truth poses, shape (N, 4, 4)
            title: optional title for the plot
            outpath: optional output path to save the plot
        Returns:
            dict: computed RPE metrics
            str: path to saved plot
        """

        assert preds.shape == target.shape, "Preds and targets must have the same shape"
        assert preds.shape[-2:] == (4, 4), "Poses must be 4x4 matrices"

        N = preds.shape[0]
        if N <= self.delta:
            return

        pred_rel = torch.linalg.inv(preds[:-self.delta]) @ preds[self.delta:]
        gt_rel = torch.linalg.inv(target[:-self.delta]) @ target[self.delta:]

        err = torch.linalg.inv(gt_rel) @ pred_rel

        # translation error
        trans_error = torch.linalg.norm(err[:, :3, 3], dim=1)

        # rotation error (angle)
        rot_trace = torch.sum(torch.diagonal(err[:, :3, :3], dim1=-2, dim2=-1), dim=1)
        rot_error = torch.acos(torch.clamp((rot_trace - 1) / 2, -1.0, 1.0))

        trans_rmse = torch.sqrt(torch.mean(trans_error ** 2)).item()
        rot_rmse = torch.rad2deg(torch.sqrt(torch.mean(rot_error ** 2))).item()

        steps = range(len(trans_error))
        fig, ax1 = plt.subplots(figsize=(7, 4))

        ax1.plot(steps, trans_error.cpu().numpy(), 'b-', label="Translational Error [m]")
        ax1.set_xlabel("Frame index")
        ax1.set_ylabel("Translation [m]", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.plot(steps, torch.rad2deg(rot_error).cpu().numpy(), 'r-', label="Rotational Error [deg]")
        ax2.set_ylabel("Rotation [deg]", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        if title:
            fig.suptitle(title,fontsize=10, fontweight='bold')

        ax1.set_title(f"Trans RMSE: {trans_rmse:.3f} m, Rot RMSE: {rot_rmse:.3f} deg",fontsize=10)

        fig.tight_layout()
        
        if outpath:
            plt.savefig(f"{outpath}traj_rpe.png", dpi=300)

            # additionally save data in npy file (for comparing against other methods)
            data={'steps': steps, 'trans_error' : trans_error.cpu().numpy(), 'rot_error' : torch.rad2deg(rot_error).cpu().numpy(), 'trans_rmse' : np.array(trans_rmse),  'rot_rmse' : np.array(rot_rmse)}
            np.save(f"{outpath}traj_rpe.npy",data)
        plt.close(fig)

        return {'rpe_trans_rmse': trans_rmse, 'rpe_rot_rmse': rot_rmse}, f"{outpath}traj_rpe.png"
    

class ScaleConsistency(Metric):
    """
    TorchMetrics implementation of Scale Consistency metric.
    Computes variance of per-frame scale factors between predicted and ground truth trajectories.
    """
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("var_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with predicted and ground truth positions.
        
        Args:
            preds: predicted poses, shape (N, 4, 4)
            target: ground truth poses, shape (N, 4, 4)
        """

        assert preds.shape == target.shape, "Preds and targets must have the same shape"
        assert preds.shape[-2:] == (4, 4), "Poses must be 4x4 matrices"

        # omit first frame, since it always has zero translation
        pred_t = preds[1:, :3, 3]
        gt_t = target[1:, :3, 3]

        # compute per-frame scale factors
        numer = (gt_t * pred_t).sum(dim=-1)
        denom = (pred_t.pow(2).sum(dim=-1)).clamp_min(1e-8)
        scale_factors = numer / denom

        # variance per trajectory
        scale_var = scale_factors.var(dim=-1, unbiased=False)

        self.var_sum += scale_var
        self.count += 1

    def compute(self):
        """
        Compute final scale consistency metric.
        """
        scale_var = self.var_sum / self.count if self.count > 0 else 0.0
        return {"scale_var" : scale_var}

    def plot(self, preds: torch.Tensor, target: torch.Tensor, title: str = None, outpath: str = None) -> Tuple[dict, str]:
        """
        Optional visualization of per-frame scale factors.
        Computes metric and plots results.

        Args:
            preds: predicted poses, shape (N, 4, 4)
            target: ground truth poses, shape (N, 4, 4)
            title: optional title for the plot
            outpath: optional output path to save the plot
        Returns:
            dict: computed scale variance
            str: path to saved plot
        """

        # compute metric
        assert preds.shape == target.shape, "Preds and targets must have the same shape"
        assert preds.shape[-2:] == (4, 4), "Poses must be 4x4 matrices"

        # omit first frame, since it always has zero translation
        pred_t = preds[1:, :3, 3]
        gt_t = target[1:, :3, 3]

        # compute per-frame scale factors
        numer = (gt_t * pred_t).sum(dim=-1)
        denom = (pred_t.pow(2).sum(dim=-1)).clamp_min(1e-8)
        scale_factors = numer / denom

        # variance per trajectory
        scale_var = scale_factors.var(dim=-1, unbiased=False)

        steps = range(1,len(scale_factors)+1)
        fig, ax1 = plt.subplots(figsize=(7, 4))

        ax1.plot(steps, scale_factors.cpu().numpy(), 'b-', label="Per-frame Scale Factors")
        ax1.set_xlabel("Frame index")
        ax1.set_ylabel("Scale Factor", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        if title:
            fig.suptitle(title,fontsize=10, fontweight='bold')

        ax1.set_title(f"Scale Variance: {scale_var:.3f}",fontsize=10)

        fig.tight_layout()
        
        if outpath:
            plt.savefig(f"{outpath}traj_scale_cons.png", dpi=300)

            # additionally save data in npy file (for comparing against other methods)
            data={'steps': steps, 'scale_factors' : scale_factors.cpu().numpy(), 'scale_var' : scale_var.cpu().numpy()}
            np.save(f"{outpath}traj_scale_cons.npy",data)
        plt.close(fig)

        return {'scale_var': scale_var}, f"{outpath}traj_scale_cons.png"