import os.path as osp
import logging
import random
import glob

import numpy as np

from vggt.training.data.dataset_util import *
from vggt.training.data.base_dataset import BaseDataset

class WaymoDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        Waymo_DIR: str = None,
        sequence_ids: list = None,
        exclude_ids: bool = True,
        cameras: list = ["cam_01","cam_02","cam_03","cam_04","cam_05"],
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the VKittiDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            Waymo_DIR (str): Directory path to Waymo data.
            sequence_ids (list): List of specific sequence IDs to include.
            exclude_ids (bool): Whether to exclude the specified sequence IDs.
            cameras (list): List of camera names to include.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
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

        if Waymo_DIR is None:
            raise ValueError("Waymo_DIR must be specified.")

        self.Waymo_DIR = Waymo_DIR

        logging.info(f"Waymo_DIR is {self.Waymo_DIR}")

        if split == "train":
            split_str = "training"
            self.len_train = len_train
        elif split == "val":
            split_str = "validation"
            self.len_train = len_test
        elif split == "test":
            split_str = "testing"
            self.len_train = len_test

        # load or generate sequence list
        if sequence_ids is not None:
            ids_list = []
            # gather ids if specific ids are provided
            for id in sequence_ids:
                for camera in cameras:
                    ids = glob.glob(osp.join(self.Waymo_DIR, f"{split_str}/{id}*/frames/{camera}"))
                    ids = [file_path.split(self.Waymo_DIR)[-1].lstrip('/') for file_path in ids]
                    ids_list.extend(ids)

            ids_list = sorted(ids_list)
            
            if exclude_ids:
                sequence_list = []
                for camera in cameras:
                    sequences = glob.glob(osp.join(self.Waymo_DIR, f"{split_str}/*/frames/{camera}"))
                    sequences = [file_path.split(self.Waymo_DIR)[-1].lstrip('/') for file_path in sequences if file_path.split(self.Waymo_DIR)[-1].lstrip('/') not in ids_list]
                    sequence_list.extend(sequences)
            else:
                sequence_list = ids_list                
        else:
            sequence_list = []
            for camera in cameras:
                sequences = glob.glob(osp.join(self.Waymo_DIR, f"{split_str}/*/frames/{camera}"))
                sequences = [file_path.split(self.Waymo_DIR)[-1].lstrip('/') for file_path in sequences]
                sequence_list.extend(sequences)

        sequence_list = sorted(sequence_list)
        
        # count number of images in each sequence
        self.seq_frame_num = []
        for seq in sequence_list:
            frame_num = len(glob.glob(osp.join(self.Waymo_DIR, seq, "*.jpg")))

            # adjust for subsampling step if subsampling is enabled
            if self.subsampling_step > 1:
                frame_num = int(np.ceil(frame_num / self.subsampling_step))

            if self.fix_seq_img_num > 0 and self.fix_seq_img_num < frame_num:
                frame_num = self.fix_seq_img_num

            self.seq_frame_num.append(frame_num)
            
        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)
        self.depth_max = 80
        
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Waymo Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Waymo Data dataset length: {len(self)}")

    def get_seq_name(self, seq_index: int) -> str:
        """
        Get the sequence name for a given index.
        Args:
            seq_index (int): Index of the sequence.
        Returns:
            str: Sequence name.
        """
        sequence = self.sequence_list[seq_index].split("/")[1].split("_")[0]
        camera = "".join(self.sequence_list[seq_index].split("/")[3].split("_"))
        return "_".join((sequence,camera))

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
        
        # load camera parameters
        try:
            car_poses = np.load(osp.join(self.Waymo_DIR, "/".join(seq_name.split("/")[:2]), "poses.npy"))

            calibration_dict = np.load(osp.join(self.Waymo_DIR, "/".join(seq_name.split("/")[:2]), "calibration.pkl"),allow_pickle=True)
            image_size = calibration_dict["dims"]

            # +z forward, +y down, +x right -> +z is up, +y is left, +x is forward (model->waymo)
            model_axis_to_waymo_axis = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

            camera_poses = (model_axis_to_waymo_axis.T @ car_poses @ model_axis_to_waymo_axis) @ (model_axis_to_waymo_axis.T @ calibration_dict["extrinsics"][camera_id])
            
            # convert c2w to w2c
            camera_extr_full = np.linalg.inv(camera_poses)

            camera_extr = np.linalg.inv(calibration_dict["extrinsics"][camera_id])[:3,:4]
            
            camera_intr = calibration_dict["proj_mats"][camera_id]
            camera_intr[0,2] += (image_size[1] / 2)
            camera_intr[1,2] += (image_size[0] / 2)
            camera_intr[0,0] *= (image_size[1] / 2)
            camera_intr[1,1] *= (image_size[0] / 2)
            
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
        original_sizes = []

        for image_idx in ids:
            # process camera matrices
            extri_opencv = camera_extr_full[image_idx][:3,:4]
            intri_opencv = camera_intr

            # load image and depth map
            image_filepath = osp.join(self.Waymo_DIR, seq_name, f"{image_idx:010d}.jpg")
            image = read_image_cv2(image_filepath)
            original_size = np.array(image.shape[:2])

            lidar_filepath = osp.join(self.Waymo_DIR, "/".join(seq_name.split("/")[:3]).replace("/frames", "/lidar"), f"{image_idx:010d}.npy")
            lidar_data = np.load(lidar_filepath)
            points = np.concatenate((lidar_data,np.ones((lidar_data.shape[0],1))),axis=-1).T
            depth_map = self.lidar_to_depth(points, intri_opencv, camera_extr, image_size)
            depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)

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
            original_sizes.append(original_size)

        set_name = "waymo"
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
    

    def lidar_to_depth(self, points: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, image_size: tuple, eps: float = 0.05) -> np.ndarray:
        """
        Convert LiDAR points to a depth map using camera intrinsics and extrinsics.
        Args:
            points (np.ndarray): LiDAR points in homogeneous coordinates (4xN).
            intrinsics (np.ndarray): Camera intrinsic matrix (3x3).
            extrinsics (np.ndarray): Camera extrinsic matrix (3x4).
            image_size (tuple): Size of the image (height, width).
            eps (float): Small value to handle depth comparisons.
        Returns:
            np.ndarray: Depth map of shape (H, W).
        """
        H,W = image_size
        
        lidar_cam_points = (intrinsics @ (extrinsics @ points)).T

        # filter out points behind cam
        lidar_cam_points = lidar_cam_points[lidar_cam_points[:,2]>0]

        pixels = lidar_cam_points[:,:2] / lidar_cam_points[:,2:]
        valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] >= 0) & (pixels[:, 1] < H)

        pixels = pixels[valid]
        depths = lidar_cam_points[valid][:,2]

        depth_map = np.zeros((H, W), dtype=np.float32)
        weight_map = np.zeros((H, W), dtype=np.float32)
        zbuf = np.full((H, W), np.inf, dtype=np.float32)

        # floor and fractional
        j = np.floor(pixels[:,0]).astype(int)  # pixel x index (col)
        i = np.floor(pixels[:,1]).astype(int)  # pixel y index (row)
        du = pixels[:,0] - j
        dv = pixels[:,1] - i

        # neighbor offsets and weights
        neigh = [(0,0, (1-du)*(1-dv)),
                (0,1, du*(1-dv)),
                (1,0, (1-du)*dv),
                (1,1, du*dv)]

        for di, dj, w in neigh:
            rows = i + di
            cols = j + dj
            # valid neighbor indices
            mask = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & (w > 0)
            if not np.any(mask):
                continue
            rows_m = rows[mask]; cols_m = cols[mask]; w_m = w[mask]; z_m = depths[mask]

            # iterate contributions
            for r, c, wm, zm in zip(rows_m, cols_m, w_m, z_m):
                cur_z = zbuf[r, c]
                if zm < cur_z - eps:
                    zbuf[r, c] = zm
                    depth_map[r, c] = zm
                    weight_map[r, c] = wm
                elif abs(zm - cur_z) <= eps:
                    # weighted average of near-equal depths
                    prev_w = weight_map[r, c]
                    if prev_w == 0:
                        depth_map[r, c] = zm
                        weight_map[r, c] = wm
                    else:
                        depth_map[r, c] = (depth_map[r, c] * prev_w + zm * wm) / (prev_w + wm)
                        weight_map[r, c] = prev_w + wm
                # else ignore contribution (point behind)
        
        return depth_map
        