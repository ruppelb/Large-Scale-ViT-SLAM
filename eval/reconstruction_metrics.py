from typing import Tuple
import torch
from torchmetrics import Metric
from pytorch3d.ops import knn_points
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ChamferDistanceMetrics(Metric):
    """
    TorchMetrics implementation of Chamfer Distance, Completion, Accuracy, and Precision/Recall.
    Uses PyTorch3D knn_points for nearest neighbor search.
    Args:
        norm: int, norm to use for distance computation (1 for L1, 2 for L2) 
        max_dist: float or None, maximum distance to consider (clamps distances)
        rmse: bool, whether to compute RMSE or mean distances
    """

    full_state_update = False

    def __init__(self, norm: int = 2, max_dist: bool = None, rmse: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.norm = norm
        self.max_dist = max_dist
        self.rmse = rmse

        self.add_state("pred_to_gt", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("gt_to_pred", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")


    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with predicted and ground truth point clouds.

        Args:
            preds: (N_pred, 3) predicted point clouds
            target: (N_gt, 3) ground truth point clouds
        """
        assert preds.ndim == 2 and target.ndim == 2

        #add batch dimension:
        preds = preds.unsqueeze(0)
        target = target.unsqueeze(0)

        # Compute nearest neighbor distances using PyTorch3D knn_points
        dists_pred_to_gt = knn_points(preds, target, K=1, norm=self.norm).dists[..., 0] # (1, Np)
        dists_gt_to_pred = knn_points(target, preds, K=1, norm=self.norm).dists[..., 0] # (1, Ng)

        if self.max_dist is not None:
            dists_pred_to_gt = torch.clamp(dists_pred_to_gt, max=self.max_dist)
            dists_gt_to_pred = torch.clamp(dists_gt_to_pred, max=self.max_dist)

        self.pred_to_gt = torch.cat([self.pred_to_gt, dists_pred_to_gt.flatten().detach().to(self.pred_to_gt.device)])
        self.gt_to_pred = torch.cat([self.gt_to_pred, dists_gt_to_pred.flatten().detach().to(self.gt_to_pred.device)])

    def compute(self):
        """
        Return Chamfer Distance, Completion, Accuracy, and Precision/Recall.
        """

        if self.rmse:
            acc_rmse = torch.sqrt((self.pred_to_gt ** 2).mean()) if len(self.pred_to_gt) > 0 else 0.0
            comp_rmse = torch.sqrt((self.gt_to_pred ** 2).mean()) if len(self.pred_to_gt) > 0 else 0.0
            chamfer_rmse = 0.5 * acc_rmse + 0.5 * comp_rmse

            results = {
                "chamfer_distance_rmse": chamfer_rmse,
                "accuracy_rmse": acc_rmse,
                "completion_rmse": comp_rmse
            }
        else:
            acc = self.pred_to_gt.mean().item() if len(self.pred_to_gt) > 0 else 0.0
            comp = self.gt_to_pred.mean().item() if len(self.gt_to_pred) > 0 else 0.0
            chamfer = 0.5 * acc + 0.5 * comp

            results = {
                "chamfer_distance": chamfer,
                "accuracy": acc,
                "completion": comp
            }

        return results

    def plot(self, preds: torch.Tensor, target: torch.Tensor, title: str = None, outpath: str = None) -> Tuple[dict, str]:
        """
        Optional visualization of GT vs predicted trajectory.
        Computes metric and plots results.
        Args:
            preds: (N_pred, 3) predicted point clouds
            target: (N_gt, 3) ground truth point clouds
            title: Optional title for the plot
            outpath: Optional path to save the plot
        Returns:
            outdict: Dictionary with computed metrics
        """

        assert preds.ndim == 2 and target.ndim == 2

        # add batch dimension:
        preds = preds.unsqueeze(0)
        target = target.unsqueeze(0)

        # compute nearest neighbor distances using PyTorch3D knn_points
        dists_pred_to_gt = knn_points(preds, target, K=1, norm=self.norm).dists[..., 0] # (1, N_pred)
        dists_gt_to_pred = knn_points(target, preds, K=1, norm=self.norm).dists[..., 0] # (1, N_gt)

        if self.max_dist is not None:
            dists_pred_to_gt = torch.clamp(dists_pred_to_gt, max=self.max_dist)
            dists_gt_to_pred = torch.clamp(dists_gt_to_pred, max=self.max_dist)

        dists_pred_to_gt = dists_pred_to_gt.squeeze(0)
        dists_gt_to_pred = dists_gt_to_pred.squeeze(0)

        if self.rmse:
            acc = torch.sqrt((dists_pred_to_gt ** 2).mean()) if len(dists_pred_to_gt) > 0 else 0.0
            comp = torch.sqrt((dists_gt_to_pred ** 2).mean()) if len(dists_gt_to_pred) > 0 else 0.0
        else:
            acc = dists_pred_to_gt.mean().item() if len(dists_pred_to_gt) > 0 else 0.0
            comp = dists_gt_to_pred.mean().item() if len(dists_gt_to_pred) > 0 else 0.0

        chamfer = 0.5 * acc + 0.5 * comp

        # plot chamfer distance distribution
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(dists_pred_to_gt.cpu().numpy(), bins=100, alpha=0.5, label='Pred→GT')
        ax.hist(dists_gt_to_pred.cpu().numpy(), bins=100, alpha=0.5, label='GT→Pred')

        ax.set_xlabel("Distance")
        ax.set_ylabel("Count")
        ax.legend()

        if title:
            fig.suptitle(title,fontsize=10, fontweight='bold')

        if self.rmse:
            ax.set_title(f"Chamfer RMSE: {chamfer:.3f} m, Accuracy RMSE: {acc:.3f} m, Completion RMSE: {comp:.3f} m",fontsize=10)
            outdict = {'chamfer_distance_rmse': chamfer, 'accuracy_rmse': acc, 'completion_rmse': comp}
        else:
            ax.set_title(f"Chamfer: {chamfer:.3f} m, Accuracy: {acc:.3f} m, Completion: {comp:.3f} m",fontsize=10)
            outdict = {'chamfer_distance': chamfer, 'accuracy': acc, 'completion': comp}
        
        if outpath:
            plt.savefig(f"{outpath}rec_chamfer_distribution.png", dpi=300)

            # additionally save data in npy file (for comparing against other methods)
            data={'dists_pred_to_gt' : dists_pred_to_gt.cpu().numpy(), 'dists_gt_to_pred' : dists_gt_to_pred.cpu().numpy(), 'chamfer' : chamfer.cpu().numpy(), 'acc' : acc.cpu().numpy(), 'comp' : comp.cpu().numpy()}
            np.save(f"{outpath}rec_chamfer_distribution.npy",data)
            
        plt.close(fig)

        return outdict , f"{outpath}rec_chamfer_distribution.png"