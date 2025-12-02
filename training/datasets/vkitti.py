import os.path as osp
import logging
import random
import glob

import cv2
import numpy as np

from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset

SCENES = [
    "01",
    "02",
    "06",
    "18",
    "20",
]

class VKittiDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        VKitti_DIR: str = None,
        min_num_images: int = 24,
        sequence_ids: list = None,
        settings: list = ["15-deg-left","15-deg-right","30-deg-left","30-deg-right","clone","fog","morning","overcast","rain","sunset"],
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the VKittiDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            VKitti_DIR (str): Directory path to VKitti data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_range (int): Range for expanding nearby image selection.
            get_nearby_thres (int): Threshold for nearby image selection.
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
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_conf.fix_img_num
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_conf.fix_aspect_ratio

        if VKitti_DIR is None:
            raise ValueError("VKitti_DIR must be specified.")

        self.VKitti_DIR = VKitti_DIR
        #self.min_num_images = min_num_images

        #logging.info(f"VKitti_DIR is {self.VKitti_DIR}")

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test

        # Load or generate sequence list
        if sequence_ids is not None:
            sequence_list = []
            for seq_id in sequence_ids:
                for setting in settings:
                    sequences = glob.glob(osp.join(self.VKitti_DIR, f"Scene{seq_id}/{setting}/*/rgb/*"))
                    sequences = [file_path.split(self.VKitti_DIR)[-1].lstrip('/') for file_path in sequences]
                    sequence_list.extend(sequences)

            sequence_list = sorted(sequence_list)

        else:
            sequence_list = []
            for setting in settings:
                    sequences = glob.glob(osp.join(self.VKitti_DIR, f"*/{setting}/*/rgb/*"))
                    sequences = [file_path.split(self.VKitti_DIR)[-1].lstrip('/') for file_path in sequences]
                    sequence_list.extend(sequences)

            sequence_list = sorted(sequence_list)
        
        self.seq_frame_num = []
        for seq in sequence_list:
            #count number of images in each sequence
            frame_num = len(glob.glob(osp.join(self.VKitti_DIR, seq, "rgb_*.jpg")))

            if self.subsampling_step > 1:
                #adjust for subsampling step
                frame_num = int(np.ceil(frame_num / self.subsampling_step))

            if self.fix_seq_img_num > 0 and self.fix_seq_img_num < frame_num:
                frame_num = self.fix_seq_img_num

            self.seq_frame_num.append(frame_num)
            

        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        #self.len_train = len(sequence_list)

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: VKitti Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: VKitti Data dataset length: {len(self)}")
        #print(f"{status}: VKitti Real Data size: {self.sequence_list_len}")
        #print(f"{status}: VKitti Data dataset length: {len(self)}")

    def get_seq_name(self, seq_index):
        return "_".join(self.sequence_list[seq_index].split("/")[:2])

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

        camera_id = int(seq_name[-1])

        # Load camera parameters
        try:
            camera_parameters = np.loadtxt(
                osp.join(self.VKitti_DIR, "/".join(seq_name.split("/")[:2]), "extrinsic.txt"), 
                delimiter=" ", 
                skiprows=1
            )
            camera_parameters = camera_parameters[camera_parameters[:, 1] == camera_id]

            camera_intrinsic = np.loadtxt(
                osp.join(self.VKitti_DIR, "/".join(seq_name.split("/")[:2]), "intrinsic.txt"), 
                delimiter=" ", 
                skiprows=1
            )
            camera_intrinsic = camera_intrinsic[camera_intrinsic[:, 1] == camera_id]
        except Exception as e:
            logging.error(f"Error loading camera parameters for {seq_name}: {e}")
            raise

        
        frame_num = self.seq_frame_num[seq_index]


        #just to be sure to get correct number of images and aspect ratio if directly accessing dataset
        if self.fixed_num_images > 0:
            img_per_seq = self.fixed_num_images

        if self.fixed_aspect_ratio > 0:
            aspect_ratio = self.fixed_aspect_ratio

        if ids is None:
            if self.debug:
                #sample chunk of first x images
                ids = np.arange(img_per_seq)
            else:
                if self.overlapping:

                    #compute max subsampling step, so that we can extract a subtrajectory
                    rev_subsampling_steps = np.arange(self.chunk_subsampling[1],self.chunk_subsampling[0]-1, -1)
                    valid_subsampling_steps = np.ceil(frame_num / rev_subsampling_steps) >= img_per_seq
                    max_subsampling_step = rev_subsampling_steps[np.argmax(valid_subsampling_steps)]

                    #sample subsampling step
                    random_subsampling_step = np.random.randint(self.chunk_subsampling[0],max_subsampling_step+1)

                    if random_subsampling_step > 1:
                        frame_num = np.ceil(frame_num / random_subsampling_step)

                    #sample one random chunk of length img_per_seq instead
                    last_possible_index = (frame_num-img_per_seq)
                    start_idx = np.random.randint(0,last_possible_index+1)#np.random.default_rng().integers(last_possible_index,endpoint=True)
                    ids = np.arange(start_idx, start_idx + img_per_seq)

                    if random_subsampling_step > 1:
                        #map to real indices
                        ids = ids * random_subsampling_step

                else:
                    if self.fixed_num_images > 0:
                        #sample a random chunk, so that we always have non-overlapping chunks
                        start_ids = np.arange(0, frame_num - self.fixed_num_images + 1,self.fixed_num_images)
                        if len(start_ids) * self.fixed_num_images < frame_num:
                            #add additional chunk that includes last values
                            start_ids = np.append(start_ids, frame_num-self.fixed_num_images)

                        start_idx = np.random.choice(start_ids)
                        ids = np.arange(start_idx, start_idx + img_per_seq)
                    else:
                        raise ValueError("Sampling non overlapping chunks requires fixed size chunks")

        #map subsampled ids to real ids
        if self.subsampling_step > 1:
            ids = ids * self.subsampling_step

        #print(f"\nSampled ids: {ids}")

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for image_idx in ids:
            image_filepath = osp.join(self.VKitti_DIR, seq_name, f"rgb_{image_idx:05d}.jpg")
            depth_filepath = osp.join(self.VKitti_DIR, seq_name, f"depth_{image_idx:05d}.png").replace("/rgb", "/depth")

            image = read_image_cv2(image_filepath)
            depth_map = cv2.imread(depth_filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_map = depth_map / 100
            depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            # Process camera matrices
            extri_opencv = camera_parameters[image_idx][2:].reshape(4, 4)
            extri_opencv = extri_opencv[:3]

            intri_opencv = np.eye(3)
            intri_opencv[0, 0] = camera_intrinsic[image_idx][-4]
            intri_opencv[1, 1] = camera_intrinsic[image_idx][-3]
            intri_opencv[0, 2] = camera_intrinsic[image_idx][-2]
            intri_opencv[1, 2] = camera_intrinsic[image_idx][-1]

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
            original_sizes.append(original_size)

        set_name = "vkitti"
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