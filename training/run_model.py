import os
import numpy as np
import argparse
from typing import Any, Dict

from hydra import initialize, compose
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

import torch
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint,TQDMProgressBar
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.loggers import CSVLogger

from vggt.training.train_utils.freeze import freeze_modules
#from vggt.training.train_utils.normalization import normalize_camera_extrinsics_and_points_batch

from aligned_vggt.utils.data import alignAndConvertOutputs, generate_chunks, chunk_batch, normalize_camera_extrinsics_and_points_batch

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
            self.model = instantiate({"_target_": cfg.model._target_}, _recursive_=False).from_pretrained(cfg.checkpoint.from_pretrained,cache_dir=cfg.checkpoint.save_dir) #VGGT.from_pretrained("facebook/VGGT-1B" ,cache_dir="models")
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
        valid_chunk_widths = (S / rev_chunk_widths) > 1
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
            predictions = self.model(chunked_batch['images'][i],random_overlap,last_chunk_outputs,gt_poses=gt_poses)

            last_chunk_outputs = predictions
        
        #perform alignment if necessary and convert to tensors
        alignAndConvertOutputs(predictions,batch,chunked_batch,self.hparams.cfg.gt_alignment_type, self.train_chunk_width, 0 if self.training else random_overlap)

        return predictions
        

    def configure_optimizers(self):
        #only optimize camera head parameters
        max_lr = self.optim_cfg.options.lr.max_value
        min_lr = self.optim_cfg.options.lr.min_value
        linear_steps_percent = self.optim_cfg.options.lr.linear_steps

        named_parameters = dict(self.model.named_parameters())
        optimizer = instantiate(self.optim_cfg.optimizer, named_parameters.values())
        
        #Compute total number of iterations. Adapted from here https://github.com/Lightning-AI/pytorch-lightning/issues/5449
        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)

        if self.trainer.max_steps > -1:
            iterations = self.trainer.max_steps
        else:
            iterations = dataset_size * self.trainer.max_epochs // self.trainer.accumulate_grad_batches

        warmup_iterations = int(max(linear_steps_percent * iterations,1.0))
        
        #Also employ cosine lr scheduler with warmup of 8k iterations (5% of total)
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (min((step + 1) / warmup_iterations, 1.0) * (max_lr - min_lr) + min_lr) / max_lr)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=min_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iterations])
        
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
    
    def on_before_optimizer_step(self, optimizer):
        # log camerahead gradients
        norm_type = 2.0
        norms = grad_norm(self.model, norm_type=norm_type)
        if f"grad_{norm_type}_norm_total" in norms:
            if self.trainer.is_global_zero:
                self.log(f"grad_{norm_type}_norm_total",norms[f"grad_{norm_type}_norm_total"],rank_zero_only=True)

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

    logger = CSVLogger(save_dir=os.fspath(cfg.logging.log_dir), name=cfg.exp_name, version=None)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    #setup checkpointing
    last_ckpt_save_path = None
    resume_ckpt_path = None
    save_last = None
    if cfg.checkpoint.resume_from_checkpoint:
        last_ckpt_save_dir = f"{os.fspath(cfg.checkpoint.save_dir)}/_latest_checkpoints"

        if not os.path.exists(last_ckpt_save_dir):
            os.makedirs(last_ckpt_save_dir, exist_ok=True)

        last_ckpt_save_path = f"{last_ckpt_save_dir}/{cfg.exp_name}.ckpt"
        #check if we have a checkpoint to resume from
        if os.path.exists(last_ckpt_save_path):
            resume_ckpt_path = last_ckpt_save_path

        save_last = 'link'

    checkpoint_callback = CustomModelCheckpoint(dirpath= f"{os.fspath(cfg.checkpoint.save_dir)}/{cfg.exp_name}", every_n_train_steps = cfg.checkpoint.save_freq, save_on_train_epoch_end=False, last_ckpt_path=last_ckpt_save_path, save_last=save_last)
    
    #init model
    model = LitModel(cfg)

    #workaround so that we can use VGGT's worker_fn, since we do not use enviromment variables for saving rank info
    os.environ['RANK'] = "0"

    #setup trainer
    trainer = L.Trainer(devices=args.num_devices, num_nodes=args.num_nodes, strategy="ddp", use_distributed_sampler=False, max_steps=cfg.max_steps, logger=logger, log_every_n_steps=cfg.logging.log_freq, gradient_clip_val=cfg.optim.gradient_clip.max_norm, callbacks=[lr_monitor,checkpoint_callback,StepProgressBar()], check_val_every_n_epoch=None, val_check_interval=cfg.val_epoch_freq, limit_test_batches=1, limit_val_batches=1, reload_dataloaders_every_n_epochs=1, accumulate_grad_batches=cfg.accum_steps, precision=cfg.optim.amp.amp_dtype)
    
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
    
if __name__ == "__main__":
    main()