import os
import glob
import pandas as pd
import cv2
import numpy as np
import wandb
import argparse
from typing import Any, Dict
from math import ceil,floor
import logging
import time

from hydra import initialize, compose
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

import torch
import torch.utils
from torch.utils.data import Dataset

from huggingface_hub import PyTorchModelHubMixin  # used for model hub

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint,TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import grad_norm
#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3
from train_utils.freeze import freeze_modules
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

from aligned_vggt.utils.data import alignAndConvertOutputs, generate_chunks, chunk_batch
class StepProgressBar(TQDMProgressBar):

    def on_train_epoch_start(self, trainer, *_):
        if self._leave:
            self.train_progress_bar = self.init_train_tqdm()

        if self.train_progress_bar.total != trainer.max_steps:
            self.train_progress_bar.total = trainer.max_steps

        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        idx = self.trainer.global_step - 1
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, idx)
    
    
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx = 0):
        if not self.has_dataloader_changed(dataloader_idx):
            return
        
        self.val_progress_bar.reset(None)
        self.val_progress_bar.initial = 0
        desc = self.sanity_check_description if trainer.sanity_checking else self.validation_description
        self.val_progress_bar.set_description(f"{desc} DataLoader {dataloader_idx}")
    
class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath = None, filename = None, monitor = None, verbose = False, save_last = None, last_ckpt_path = None, save_top_k = 1, save_weights_only = False, mode = "min", auto_insert_metric_name = True, every_n_train_steps = None, train_time_interval = None, every_n_epochs = None, save_on_train_epoch_end = None, enable_version_counter = True):
        super().__init__(dirpath, filename, monitor, verbose, save_last, save_top_k, save_weights_only, mode, auto_insert_metric_name, every_n_train_steps, train_time_interval, every_n_epochs, save_on_train_epoch_end, enable_version_counter)
        self.last_ckpt_path = last_ckpt_path

    def on_validation_end(self, trainer, pl_module):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        pass

    #def _save_checkpoint(self, trainer, filepath):
    #    #adapt file path to avoid racing condition (TODO: debug further why this even can happen in the first place)
    #    filepath = f"{filepath.split('.')[0]}_rank[{trainer.global_rank}].ckpt"
    #    super()._save_checkpoint(trainer, filepath)

    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath = None):
        filepath = self.format_checkpoint_name(monitor_candidates)

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while self.file_exists(filepath, trainer) and filepath != del_filepath:
                filepath = self.format_checkpoint_name(monitor_candidates, ver=version_cnt)
                version_cnt += 1

        filepath = f"{filepath.split('.ckpt')[0]}_rank[{trainer.global_rank}].ckpt"

        return filepath
   
    def _save_last_checkpoint(self, trainer, monitor_candidates):
        if not self.save_last:
            return

        if self.last_ckpt_path:
            filepath = self.last_ckpt_path
        else:
            filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)

            if self._enable_version_counter:
                version_cnt = self.STARTING_VERSION
                while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
                    filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt)
                    version_cnt += 1

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        if self.save_last == "link": #changed logic here. If we not have already created a checkpoint, skip adding link
            if self._last_checkpoint_saved and self.save_top_k != 0:
                self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)
        else:
            self._save_checkpoint(trainer, filepath)
        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)
        

    def on_train_end(self, trainer, pl_module):
        #delete latest checkpoint file if it exists
        if trainer.is_global_zero:
            if self.last_ckpt_path and os.path.exists(self.last_ckpt_path):
                os.remove(self.last_ckpt_path)
                print(f"Removed last checkpoint link {self.last_ckpt_path}")
        trainer.strategy.barrier("Removing last checkpoint link")

#adapted from https://actamachina.com/posts/visual-odometry
class KITTIOdometryDatasetOld(Dataset):
    def __init__(self, data_dir, samplingMode, sequence_ids, n=5, overlap=1):
        """Initialize the KITTI Odometry Dataset.

        Parameters:
        - data_dir: Directory containing KITTI odometry data
        - sequence_ids: List of sequence IDs to load
        - n: Number of frames in a sequence chunk
        - overlap: Overlap between sequence chunks
        """
        
        self.samplingMode = samplingMode
        # List to store data samples
        data = []

        # Load data for each specified sequence
        for seq_id in sequence_ids:
            image_paths = sorted(glob.glob(f"{data_dir}/sequences/{seq_id}/image_2/*"))

            #load extrinsics
            pose_data = pd.read_csv(f"{data_dir}/poses/{seq_id}.txt", header=None, sep='\s+')
            extrinsics = pose_data.to_numpy().reshape(-1, 3, 4)
            extrinsics = np.pad(extrinsics, ((0, 0), (0, 1), (0, 0)), mode="constant")
            extrinsics[:,3,3] = 1.0
            extrinsics = np.linalg.inv(extrinsics) #invert values to w->c

            #load intrinsics
            calib_data = pd.read_csv(f"{data_dir}/sequences/{seq_id}/calib.txt", header=None, sep='\s+',index_col=0)
            projectionMatrix = calib_data.loc["P2:"].to_numpy().reshape(3, 4)
            intrinsics,_,_,_,_,_,_= cv2.decomposeProjectionMatrix(projectionMatrix)
            if len(intrinsics.shape) == 2:
                #assume same intrinsics for all frame
                intrinsics = np.repeat(intrinsics[None,...],extrinsics.shape[0],axis=0)
            
            #TODO: opt apply Augmentation. VGGT apply augmentation at two points. 
            #Augmentation regarding orientation or size (which also affects pose or world points) is applied when initally preparing data.
            #Color augmentation (jitter, blur, gray scale, etc) is applied in getitem() method
            
            if samplingMode == "overlappingEqualChunks" or samplingMode == "debug_sameChunk" or samplingMode == "debug_equalChunks" or samplingMode == "debug_randomChunk":
                # Break sequence into chunks
                for i in range(0, len(pose_data) - n, n - overlap):
                    data.append(
                        {
                            "image_path": image_paths[i : i + n],
                            "extrinsic": extrinsics[i : i + n],
                            "intrinsic": intrinsics[i : i + n],
                        }
                    )
            elif samplingMode == "randomChunks":
                #TODO: implement random sampling (randomly sample 2–24 consecutive frames)
                len(image_paths)
                pass

        self.df = pd.DataFrame(data)

    def __getitem__(self, index):
        if self.samplingMode == "debug_sameChunk":
            # For debugging, only return the first sample
            index = 0
        
        entry = self.df.iloc[index]

        image_paths = entry["image_path"]
        extrinsics = torch.Tensor(entry["extrinsic"])
        intrinsics = torch.Tensor(entry["intrinsic"])

        #convert poses to be relative to first frame in sequence
        chunk_origin = closed_form_inverse_se3(extrinsics[None,0]).squeeze(0)
        extrinsics = chunk_origin @ extrinsics

        #Transform and stack images
        images = load_and_preprocess_images(image_paths)

        return images, extrinsics[:,:3,:4], intrinsics

    def __len__(self):
        return len(self.df)

class KITTIOdometryDataModuleOld(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        samplingMode: str,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_sequence_ids = ["04","01", "02", "05", "06", "07", "08", "09","10"]
        self.val_sequence_ids = ["03", "00"]
        #self.predict_sequence_ids = ["05"]
        #self.test_sequence_ids = ["05"]

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = KITTIOdometryDatasetOld(
                self.hparams.data_dir,
                self.hparams.samplingMode,
                sequence_ids=self.train_sequence_ids,
            )
            self.val_dataset = KITTIOdometryDatasetOld(
                self.hparams.data_dir,
                self.hparams.samplingMode,
                sequence_ids=self.val_sequence_ids,
            )
        elif stage == "validate":
            self.val_dataset = KITTIOdometryDatasetOld(
                self.hparams.data_dir,
                self.hparams.samplingMode,
                sequence_ids=self.val_sequence_ids,
            )
        """
        elif stage == "test":
            self.test_dataset = KITTIOdometryDatasetOld(
                self.hparams.data_dir,
                sequence_ids=self.predict_sequence_ids,
                transform=self.transform,
            )
        elif stage == "predict":
            self.predict_dataset = KITTIOdometryDatasetOld(
                self.hparams.data_dir,
                sequence_ids=self.predict_sequence_ids,
                transform=self.transform,
            )
        """

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
    
    """
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
    """

class DynamicDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_cfg: Dict[str, Any],
        seed: int
    ):
        super().__init__()
        self.cfg = data_cfg
        self.seed_value = seed

    def setup(self, stage):
        self.train_dataset = None
        self.val_dataset = None

        if stage == "validate" or stage == "fit":
            self.val_dataset = instantiate(self.cfg.val, _recursive_=False)
            self.val_dataset.seed = self.seed_value

        if stage == "fit":
            self.train_dataset = instantiate(self.cfg.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

        if stage == "test":
            self.test_dataset = instantiate(self.cfg.test, _recursive_=False)
            self.test_dataset.seed = self.seed_value
    
    def train_dataloader(self):
        return self.train_dataset.get_loader(epoch=self.trainer.global_step)
    
    def val_dataloader(self):
        return self.val_dataset.get_loader(epoch=self.trainer.global_step)
    
    def test_dataloader(self):
        return self.test_dataset.get_loader(epoch=self.trainer.global_step)
    
    def on_before_batch_transfer(self, batch, dataloader_idx):

        with torch.amp.autocast("cuda", enabled=False): #with torch.cuda.amp.autocast(enabled=False):
        
            #only performs first frame alignment, no normalization
            normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
                normalize_camera_extrinsics_and_points_batch(
                    extrinsics=batch["extrinsics"],
                    cam_points=batch["cam_points"],
                    world_points=batch["world_points"],
                    depths=batch["depths"],
                    point_masks=batch["point_masks"],
                    scale_by_points=False
                )

            # Replace the original values in the batch with the adjusted ones.
            batch["extrinsics"] = normalized_extrinsics
            batch["cam_points"] = normalized_cam_points
            batch["world_points"] = normalized_world_points
            batch["depths"] = normalized_depths

        return batch
    

class LitModel(L.LightningModule, PyTorchModelHubMixin):
    def __init__(self, cfg):
        super().__init__()

        self.loss = instantiate(cfg.loss, _recursive_=False)
        self.loss.setupScheduling(cfg.max_steps)

        self.optim_cfg = cfg.optim
        self.metrics = instantiate(cfg.metrics, _recursive_=False)
        
        self.mode = cfg.mode
        self.train_num_overlap = cfg.num_overlap
        self.val_num_overlap = cfg.metrics.overlap
        self.train_chunk_width = cfg.chunk_width
        self.val_chunk_width = cfg.metrics.chunk_width
        self.train_sample_mode = cfg.sample_mode
        self.val_sample_mode = cfg.metrics.full_seq_sample_mode

        if cfg.checkpoint.model_checkpoint_path is not None:
            self.model = instantiate(cfg.model, _recursive_=False)
            self._load_model_checkpoint(cfg.checkpoint.model_checkpoint_path, cfg.checkpoint.fallback_checkpoint_path if "fallback_checkpoint_path" in cfg.checkpoint else None) #_load_from_state_dict(cfg.checkpoint.model_checkpoint_path)
        else:
            self.model = instantiate({"_target_": cfg.model._target_}, _recursive_=False).from_pretrained(cfg.checkpoint.from_pretrained,cache_dir="models") #VGGT.from_pretrained("facebook/VGGT-1B" ,cache_dir="models")
            self.model.set_config(cfg.model)

        #freeze modules
        if getattr(self.optim_cfg, "frozen_module_names", None):
            
            print(
                f"[Start] Freezing modules: {self.optim_cfg.frozen_module_names}"
            )
            
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_cfg.frozen_module_names,
            )

            print(
                f"[Done] Freezing modules: {self.optim_cfg.frozen_module_names}"
            )

        self.save_hyperparameters()
        

    #NOT REQUIRED ANYMORE INSTEAD RELOAD DATALOADER EVERY EPOCH
    #Required since by default, set_epoch() is only called on distributed sampler and not on our custom batch sampler
    #def on_train_epoch_start(self):
    #   self.trainer.train_dataloader.batch_sampler.set_epoch(self.trainer.current_epoch)
    #def on_validation_epoch_start(self):
    #   self.trainer.val_dataloaders.batch_sampler.set_epoch(self.trainer.current_epoch)

    def training_step(self, batch, batch_idx):
        #print(f"\nTrain IDS: {batch['ids']}")
        
        if self.trainer.is_global_zero:
            self.log("train/trajectory_length", batch["images"].shape[1],rank_zero_only=True)
            self.log("train/batch_size", batch["images"].shape[0],rank_zero_only=True)

        predictions = self.forward(batch)

        #compute loss
        loss_dict = self.loss(predictions,batch,self.trainer.global_step) #compute_loss(predictions, batch)
        
        if self.trainer.is_global_zero:
            #log losses
            for key in loss_dict.keys():
                self.log(f"train/{key}", loss_dict[key],rank_zero_only=True)

        loss = loss_dict["objective"]

        return loss

    def validation_step(self, batch, batch_idx):

        #print(f"\nVal IDS: {batch['ids']}")
        
        if self.trainer.is_global_zero:
            self.log("val/trajectory_length", batch["images"].shape[1], rank_zero_only=True)
            self.log("val/batch_size", batch["images"].shape[0], rank_zero_only=True)

        predictions = self.forward(batch)

        #compute loss
        loss_dict = self.loss(predictions,batch,self.trainer.global_step)
        
        #log losses
        for key in loss_dict.keys():
           self.log(f"val/{key}", loss_dict[key], sync_dist=True)

        loss = loss_dict["objective"]
        
        batch_metrics_dict, seq_metrics_dict = self.metrics(predictions, batch, self.model, self.trainer)

        #log metrics
        for key in batch_metrics_dict.keys():
            self.log(f"batch_metrics/{key}", batch_metrics_dict[key], sync_dist=True)

        if self.trainer.is_global_zero:
            for (key,value) in seq_metrics_dict.items():
                self.log(key, value, rank_zero_only=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        predictions = self.forward(batch)

        #compute loss
        #loss_dict = self.loss(predictions,batch,self.trainer.global_step)
        
        #log losses
        #for key in loss_dict.keys():
        #   self.log(f"val/{key}", loss_dict[key], sync_dist=True)

        #loss = loss_dict["objective"]
        
        batch_metrics_dict, seq_metrics_dict = self.metrics(predictions, batch, self.model, self.trainer)

        #log metrics
        for key in batch_metrics_dict.keys():
            self.log(f"batch_metrics/{key}", batch_metrics_dict[key], sync_dist=True)

        if self.trainer.is_global_zero:
            for (key,value) in seq_metrics_dict.items():
                self.log(key, value, rank_zero_only=True)
        
        return None
    
    def forward(self, batch: dict):

        #TODO handle case when we have no batch dim
        B, S, _, _, _ = batch['images'].shape

        #sample random sequence width
        """
        #calculate max chunk_width, so that we have at least two full chunks
        rev_chunk_widths = np.arange(self.chunk_width[0],self.chunk_width[1]+1)[::-1]
        valid_chunk_widths = (S / (rev_chunk_widths - self.num_overlap)) >= 2
        max_chunk_width = rev_chunk_widths[np.argmax(valid_chunk_widths)]
        random_chunk_width = np.random.randint(self.chunk_width[0],max_chunk_width+1)
        """

        if self.training:
            chunk_width = self.train_chunk_width
            sample_mode = self.train_sample_mode
            num_overlap = self.train_num_overlap
        else:
            chunk_width = self.val_chunk_width
            sample_mode = self.val_sample_mode
            num_overlap = self.val_num_overlap
            
        #calculate width that at least one full chunk exists
        rev_chunk_widths = np.arange(chunk_width[1],chunk_width[0]-1, -1)
        valid_chunk_widths = (S / rev_chunk_widths) >= 1
        max_chunk_width = rev_chunk_widths[np.argmax(valid_chunk_widths)]
        random_chunk_width = np.random.randint(chunk_width[0],max_chunk_width+1)

        #sample random num_overlap (so that its smaller than chunk width)
        rev_overlaps = np.arange(num_overlap[1],num_overlap[0]-1,-1)
        valid_overlaps = rev_overlaps < random_chunk_width
        max_overlap = rev_overlaps[np.argmax(valid_overlaps)]
        random_overlap = np.random.randint(num_overlap[0],max_overlap+1)

        if self.trainer.is_global_zero:
            if self.training:
                self.log("train/chunk_width", random_chunk_width, rank_zero_only=True)
                self.log("train/chunk_overlap", random_overlap, rank_zero_only=True)
            elif (self.mode == "train" or self.mode == "validation"):
                self.log("val/chunk_width", random_chunk_width, rank_zero_only=True)
                self.log("val/chunk_overlap", random_overlap, rank_zero_only=True)

        indices = generate_chunks(S,sample_mode,random_chunk_width,random_overlap)

        chunked_batch = chunk_batch(batch,indices)

        #get model predictions
        last_chunk_outputs = None
        for i in range(len(indices)):
            gt_poses = chunked_batch["extrinsics"][i] if sample_mode == "chunk_gt" or sample_mode == "two_chunks" else None
            predictions = self.model(chunked_batch['images'][i],random_overlap,last_chunk_outputs,True,gt_poses=gt_poses)

            last_chunk_outputs = predictions
        
        #perform alignment if necessary and convert to tensors
        #only align during val or testing (since we want network to learn scale/pose alignment)
        #self.hparams.cfg.gt_alignment_type if not self.training else ""
        alignAndConvertOutputs(predictions,batch,chunked_batch,self.hparams.cfg.gt_alignment_type, self.train_chunk_width, 0 if self.training else random_overlap)

        return predictions
        

    def configure_optimizers(self):
        #only optimize camera head parameters
        max_lr = self.optim_cfg.options.lr.max_value
        min_lr = self.optim_cfg.options.lr.min_value
        linear_steps_percent = self.optim_cfg.options.lr.linear_steps
        #weight_decay = self.optim_cfg.options.weight_decay.value #is constant in default vggt config
        
        #VGGT uses adamw with lr of 0.0002 for 160k iterations (iteration = one batch applied to model) in total
        named_parameters = dict(self.model.named_parameters())
        #print(named_parameters)
        optimizer = instantiate(self.optim_cfg.optimizer, named_parameters.values())
        #optimizer = torch.optim.AdamW(self.camera_head.parameters(), lr=max_lr, weight_decay=weight_decay)
        
        #Compute total number of iterations. Adapted from here https://github.com/Lightning-AI/pytorch-lightning/issues/5449
        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)

        if self.trainer.max_steps > -1:
            iterations = self.trainer.max_steps
        else:
            iterations = dataset_size * self.trainer.max_epochs // self.trainer.accumulate_grad_batches

        warmup_iterations = int(max(linear_steps_percent * iterations,1.0))
        #warmup_epochs = int(warmup_iterations // dataset_size)
        
        #Also employ cosine lr scheduler with warmup of 8k iterations (5% of total)
        #scheduler = LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs,self.trainer.max_epochs) #based on epochs
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (min((step + 1) / warmup_iterations, 1.0) * (max_lr - min_lr) + min_lr) / max_lr)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=min_lr) #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations / 8)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iterations])
        
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
    
    def on_before_optimizer_step(self, optimizer):
        # log camerahead gradients
        norm_type = 2.0
        norms = grad_norm(self.model, norm_type=norm_type)
        if f"grad_{norm_type}_norm_total" in norms:
            if self.trainer.is_global_zero:
                self.log(f"grad_{norm_type}_norm_total",norms[f"grad_{norm_type}_norm_total"],rank_zero_only=True)

    """
    def on_after_backward(self):
        for name,p in self.named_parameters():
            if p.grad is None:
                print(name)
        print("on_before_opt exit")
    """

    def _load_model_checkpoint(self, ckpt_path: str, fallback_ckpt_path: str = None):
        """Loads state dict of checkpoint from the given path."""
        print(f"Loading checkpoint from {ckpt_path}")
        
        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model_state_dict = {k.split('.',1)[1]: v for k, v in model_state_dict.items()} # remove 'model.' prefix

        missing, unexpected = self.model.load_state_dict(model_state_dict, strict= fallback_ckpt_path is None)
        print(f" Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        if missing and fallback_ckpt_path is not None:
            print(f"Loading fallback checkpoint from {fallback_ckpt_path}")

            with g_pathmgr.open(fallback_ckpt_path, "rb") as f:
                fallback_ckpt = torch.load(f, map_location="cpu")

            fallback_state = fallback_ckpt["state_dict"] if "state_dict" in fallback_ckpt else fallback_ckpt

            filled = {}
            for key in missing:
                if key in fallback_state:
                    model_state_dict[key] = fallback_state[key]
                    filled[key] = True

            print(f"Filled {len(filled)} missing keys from fallback: {list(filled.keys()) or 'None'}.")

            # ---- Load again with the merged state dict ----
            missing_after, unexpected_after = self.model.load_state_dict(model_state_dict, strict=True)

            print(f"After fallback load. Remaining missing keys: {missing_after or 'None'}. "
                f"Unexpected keys: {unexpected_after or 'None'}.")


#parse arguments
parser = argparse.ArgumentParser(description="Finetune camera head of vggt model.")
parser.add_argument("--config", type=str, default="default", help="Name of the config file (without .yaml extension, default: default)")
parser.add_argument("--num_devices", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)

def main():
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

    #setup data
    datamodule = DynamicDataModule(cfg.data,cfg.seed_value)

    #init logging
    tags = cfg.tags if "tags" in cfg else []
    if cfg.mode == "test":
        tags.append("results")

    wandb_logger = WandbLogger(project=cfg.project_name, name=cfg.exp_name, save_dir=cfg.logging.log_dir, tags=tags)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    #logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    #setup checkpointing
    last_ckpt_save_path = None
    resume_ckpt_path = None
    save_last = None
    if cfg.checkpoint.resume_from_checkpoint:
        last_ckpt_save_dir = f"{os.fspath(cfg.logging.log_dir)}/{cfg.project_name}/_latest_checkpoints"

        if not os.path.exists(last_ckpt_save_dir):
            os.makedirs(last_ckpt_save_dir, exist_ok=True)

        last_ckpt_save_path = f"{last_ckpt_save_dir}/{cfg.exp_name}.ckpt"
        #check if we have a checkpoint to resume from
        if os.path.exists(last_ckpt_save_path):
            resume_ckpt_path = last_ckpt_save_path

        save_last = 'link'
    
    checkpoint_callback = CustomModelCheckpoint(every_n_train_steps = cfg.checkpoint.save_freq, save_on_train_epoch_end=False, last_ckpt_path=last_ckpt_save_path, save_last=save_last)#, )#, #save_last=True, save_top_k=1, monitor="step", mode="max", 
    
    #init model
    model = LitModel(cfg)

    #setup trainer
    #deterministic=True, DDPStrategy(find_unused_parameters=True)
    trainer = L.Trainer(devices=args.num_devices, num_nodes=args.num_nodes, strategy="ddp", use_distributed_sampler=False, max_steps=cfg.max_steps, logger=wandb_logger, log_every_n_steps=cfg.logging.log_freq, gradient_clip_val=cfg.optim.gradient_clip.max_norm, callbacks=[lr_monitor,checkpoint_callback,StepProgressBar()], check_val_every_n_epoch=None, val_check_interval=cfg.val_epoch_freq, limit_test_batches=1, limit_val_batches=1, reload_dataloaders_every_n_epochs=1, accumulate_grad_batches=cfg.accum_steps, precision=cfg.optim.amp.amp_dtype)
    
    #set seed
    seed_value = (cfg.seed_value + trainer.global_rank) * cfg.max_steps
    L.seed_everything(seed_value,verbose=True)

    if cfg.mode == "train":
        #train model
        trainer.fit(model=model,datamodule=datamodule,ckpt_path=resume_ckpt_path)
    elif cfg.mode == "validate":
        #validate model
        trainer.validate(model=model,datamodule=datamodule)
    elif cfg.mode == "test":
        #test model
        trainer.test(model=model,datamodule=datamodule)
    else:
        raise Exception("Unknown mode")
    #trainer.save_checkpoint("models/baseVGGT.ckpt")

    wandb.finish()
    
if __name__ == "__main__":
    main()