import os.path as osp
import logging
import random
import glob

import pandas as pd
import cv2
import numpy as np

from vggt.training.data.dataset_util import *
from vggt.training.data.base_dataset import BaseDataset
from vggt.vggt.utils.geometry import closed_form_inverse_se3
from skimage.measure import label,regionprops

SCENES = [
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
]

class KITTIOdometryDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        KITTIOD_DIR: str = None,
        sequence_ids: list = None,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the kitti odometry dataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            KITTIOD_DIR (str): Directory path to kitti data.
            sequence_ids (list): List of specific sequence IDs to load.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        
        Raises:
            ValueError: If KITTIOD_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.overlapping = common_conf.overlapping
        self.fix_seq_img_num = common_conf.fix_seq_img_num
        self.subsampling_step = common_conf.subsampling_step
        self.chunk_subsampling = common_conf.augs.chunk_subsampling

        # --- Optional Fixed Settings (useful for debugging) ---
        # force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_conf.fix_img_num
        # force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_conf.fix_aspect_ratio

        if KITTIOD_DIR is None:
            raise ValueError("KITTIOD_DIR must be specified.")

        self.KITTIOD_DIR = KITTIOD_DIR

        logging.info(f"KittiOd_DIR is {KITTIOD_DIR}")

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # load or generate sequence list
        if sequence_ids is not None:
            sequence_list = sorted([f"sequences/{seq}/image_2" for seq in sequence_ids])
        else:
            # generate sequence list           
            sequence_list = sorted([f"sequences/{seq}/image_2" for seq in SCENES])

        # count number of images in each sequence
        self.seq_frame_num = []
        for seq in sequence_list:
            frame_num = len(glob.glob(osp.join(self.KITTIOD_DIR, seq, "*.jpg")))

            # adjust for subsampling step if subsampling is enabled
            if self.subsampling_step > 1:
                frame_num = int(np.ceil(frame_num / self.subsampling_step))

            if self.fix_seq_img_num > 0 and self.fix_seq_img_num < frame_num:
                frame_num = self.fix_seq_img_num

            self.seq_frame_num.append(frame_num)

        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Kitti odometry Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Kitti odometry Data length of training set: {len(self)}")

    def get_seq_name(self, seq_index: int) -> str:
        """
        Get the sequence name for a given index.
        Args:
            seq_index (int): Index of the sequence.
        Returns:
            str: Sequence name.
        """
        return self.sequence_list[seq_index].split("/")[1]
    
    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """

        if self.inside_random and ids is None:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        seq_idx_string = seq_name.split("/")[1]

        # load camera parameters
        try:
            camera_parameters = pd.read_csv(osp.join(self.KITTIOD_DIR,"poses",f"{seq_idx_string}.txt"), header=None, sep='\s+')
            camera_parameters = camera_parameters.to_numpy().reshape(-1, 3, 4)
            camera_parameters = np.pad(camera_parameters, ((0, 0), (0, 1), (0, 0)), mode="constant")
            camera_parameters[:,3,3] = 1.0
            camera_parameters = closed_form_inverse_se3(camera_parameters) #invert values to w->c
            camera_parameters = camera_parameters[:,:3,:4] #only take first 3 rows

            calib_data = pd.read_csv(osp.join(self.KITTIOD_DIR,"/".join(seq_name.split("/")[:-1]),"calib.txt"), header=None, sep='\s+',index_col=0)
            projectionMatrix = calib_data.loc["P2:"].to_numpy().reshape(3, 4)
            camera_intrinsic,_,_,_,_,_,_= cv2.decomposeProjectionMatrix(projectionMatrix)

        except Exception as e:
            logging.error(f"Error loading camera parameters for {seq_name}: {e}")
            raise

        frame_num = self.seq_frame_num[seq_index]

        # just to be sure to get correct number of images and aspect ratio if directly accessing dataset
        if self.fixed_num_images > 0:
            img_per_seq = self.fixed_num_images

        if self.fixed_aspect_ratio > 0:
            aspect_ratio = self.fixed_aspect_ratio

        if ids is None:
            if self.debug:
                # sample chunk of first x images
                ids = np.arange(img_per_seq)
            else:
                if self.overlapping:
                    
                    # compute max subsampling step, so that we can extract a subtrajectory
                    rev_subsampling_steps = np.arange(self.chunk_subsampling[1],self.chunk_subsampling[0]-1, -1)
                    valid_subsampling_steps = np.ceil(frame_num / rev_subsampling_steps) >= img_per_seq
                    max_subsampling_step = rev_subsampling_steps[np.argmax(valid_subsampling_steps)]
                    
                    # sample subsampling step
                    random_subsampling_step = np.random.randint(self.chunk_subsampling[0],max_subsampling_step+1)

                    if random_subsampling_step > 1:
                        frame_num = np.ceil(frame_num / random_subsampling_step)

                    # sample one random chunk of length img_per_seq instead
                    last_possible_index = (frame_num-img_per_seq)
                    start_idx = np.random.randint(0,last_possible_index+1)
                    ids = np.arange(start_idx, start_idx + img_per_seq)

                    if random_subsampling_step > 1:
                        # map to real indices
                        ids = ids * random_subsampling_step
                        
                else:
                    if self.fixed_num_images > 0:
                        # sample a random chunk, so that we always have non-overlapping chunks
                        start_ids = np.arange(0, frame_num - self.fixed_num_images + 1,self.fixed_num_images)
                        if len(start_ids) * self.fixed_num_images < frame_num:
                            # add additional chunk that includes last values
                            start_ids = np.append(start_ids, frame_num-self.fixed_num_images)

                        start_idx = np.random.choice(start_ids)
                        ids = np.arange(start_idx, start_idx + img_per_seq)
                    else:
                        raise ValueError("Sampling non overlapping chunks requires fixed size chunks")

        # map subsampled ids to real ids
        if self.subsampling_step > 1:
            ids = ids * self.subsampling_step

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for image_idx in ids:
            # process camera matrices
            extri_opencv = camera_parameters[image_idx]
            intri_opencv = camera_intrinsic

            # load image and depth map
            image_filepath = osp.join(self.KITTIOD_DIR, seq_name, f"{image_idx:06d}.jpg")
            image = read_image_cv2(image_filepath)
            original_size = np.array(image.shape[:2])

            depth_map = np.ones(original_size, dtype=np.float32) # we do not have depth maps for kitti odometry

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"
            
            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_filepath,
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.error(f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_filepath)
            original_sizes.append(original_size)

        set_name = "kittiOd"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }

        return batch