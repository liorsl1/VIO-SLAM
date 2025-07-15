import numpy as np
import traceback
import cv2
import torch
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import os
import pandas as pd
import yaml
import plotly.graph_objects as go
from imu_ekf import IMUEKF, update_with_ground_truth
from imu_integrator import integrate_imu_vicon2gt_style
from lightglue import LightGlue, SuperPoint
from scipy.spatial.transform import Rotation as R
import plotly.io as pio

pio.renderers.default = "browser"
from imu_utils import transform_visual_pose_to_z_gravity_frame
from pyivsta_vis import RealTimeSLAMPyVistaVisualizer


class vSLAM:
    def __init__(
        self,
        matcher_type="loftr",
        device="cpu",
        baseline=None,
        intrinsics=None,
        dist_coeffs=None,
    ):
        self.matcher_type = matcher_type
        self.device = device
        self.disk = None
        self.lg_matcher = None
        self.matcher = None
        self.superpoint = None
        self.baseline = baseline
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.disparity_map = None
        self.pts_map3d = []
        self.image_w = None
        self.image_h = None
        self.left_features = []
        self.positions = []
        self.trajectory = [np.eye(4)]
        self.left_feature_indices = []
        self.cam_positions = []
        self.global_map_points = []
        self.extrinsics = np.array([0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0]).reshape(4,4)
        self.T_SB = np.linalg.inv(self.extrinsics)  # Stereo camera to body frame
        self.T_BS = self.extrinsics
        self.R_align = np.eye(3)  # Will be set from DataManager
        self.curr_kpts_left = None
        self.prev_kpts_left = None

    def initialize(self):
        if self.matcher_type == "disk":
            print("Initializing DISK and LightGlueMatcher...")
            self.disk = KF.DISK.from_pretrained("depth").to(self.device).eval()
            self.lg_matcher = KF.LightGlueMatcher("disk").to(self.device).eval()
        elif self.matcher_type == "loftr":
            print("Initializing LoFTR...")
            self.matcher = KF.LoFTR(pretrained="indoor_new").to(self.device).eval()
        elif self.matcher_type == "superpoint":
            print("Initializing SuperPoint and LightGlue...")
            self.superpoint = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.lg_matcher = LightGlue(features="superpoint").eval().to(self.device)
        else:
            raise ValueError(
                "Invalid matcher_type. Choose 'disk', 'loftr', or 'superpoint'."
            )

        if self.intrinsics is not None:
            if isinstance(self.intrinsics, (list, tuple)):
                self.fx, self.fy, self.cx, self.cy = self.intrinsics
            elif isinstance(self.intrinsics, np.ndarray) and self.intrinsics.shape == (
                3,
                3,
            ):
                self.fx = self.intrinsics[0, 0]
                self.fy = self.intrinsics[1, 1]
                self.cx = self.intrinsics[0, 2]
                self.cy = self.intrinsics[1, 2]
            else:
                raise ValueError(
                    "Invalid intrinsics format. Must be a list/tuple or a 3x3 numpy array."
                )

    def preprocess_for_matching(self, img):
        """
        Preprocess image for feature matching.
        For 'disk': expects (H, W, 3) RGB, normalized to [0,1], shape (1, 3, H, W).
        For 'loftr' and 'superpoint': expects (H, W) grayscale, normalized to [0,1], shape (1, 1, H, W).
        """
        if self.matcher_type == "disk":
            if img.ndim == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None]  # (1, 3, H, W)
        elif self.matcher_type in ["loftr", "superpoint"]:
            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img)[None, None]  # (1, 1, H, W)
        else:
            raise ValueError("Unknown matcher_type for preprocessing.")
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def extract_features(self, image, num_features=2048):
        """Extract features from an image using the selected model."""
        if self.matcher_type == "disk":
            return self._extract_features_disk(image, num_features)
        elif self.matcher_type == "superpoint":
            return self._extract_features_superpoint(image)
        else:
            raise NotImplementedError(
                "Feature extraction not implemented for this matcher_type."
            )

    def _extract_features_disk(self, image, num_features):
        with torch.inference_mode():
            features = self.disk(image, num_features, pad_if_not_divisible=True)
        return features

    def _extract_features_superpoint(self, image):
        with torch.inference_mode():
            feats = self.superpoint({"image": image})
        return feats

    def _match_features_loftr(self, image0, image1, num_features):
        with torch.inference_mode():
            input_dict = {"image0": image0, "image1": image1}
            features = self.matcher(input_dict)
        return features

    def match_features(self, features1, features2, img_shape1, img_shape2, device):
        """Compute tentative matches between features using LightGlueMatcher."""
        if self.matcher_type == "superpoint":
            with torch.inference_mode():
                matches = self.lg_matcher({"image0": features1, "image1": features2})
            matches01 = matches["matches"][0]
            dists = matches["scores"][0]
            return matches01, dists
        else:
            hw1 = torch.tensor(img_shape1, device=device)
            hw2 = torch.tensor(img_shape2, device=device)
            kps1 = features1.keypoints
            kps2 = features2.keypoints
            lafs1 = KF.laf_from_center_scale_ori(
                kps1[None], torch.ones(1, kps1.shape[0], 1, 1, device=device)
            )
            lafs2 = KF.laf_from_center_scale_ori(
                kps2[None], torch.ones(1, kps2.shape[0], 1, 1, device=device)
            )
            with torch.inference_mode():
                dists, idxs = self.lg_matcher(
                    features1.descriptors,
                    features2.descriptors,
                    lafs1,
                    lafs2,
                    hw1=hw1,
                    hw2=hw2,
                )
            return idxs, dists

    def compute_disparity_map(self, mkpts_left, mkpts_right):
        """Compute disparity map from matched keypoints."""
        if len(mkpts_left) < 10 or len(mkpts_right) < 10:
            print("Not enough matched keypoints to compute disparity map.")
            return None

        # Convert keypoints to numpy arrays
        mkpts_left = np.array(mkpts_left)
        mkpts_right = np.array(mkpts_right)

        # Compute disparities
        disparities = (
            mkpts_left[:, 0] - mkpts_right[:, 0]
        )  # Assuming horizontal disparity

        # Create a disparity map
        disparity_map = []
        ids = []
        for i in range(len(mkpts_left)):
            x, y = int(mkpts_left[i, 0]), int(mkpts_left[i, 1])
            
            keypoint_idx = self.current_feature_mask[i]
            if disparities[i] < 0:
                # print(
                #     f"Negative disparity at index {i}: {disparities[i]}. Skipping this point."
                # )
                continue
            if x < 0 or y < 0:
                # print(
                #     f"Invalid coordinates at index {i}: ({x}, {y}). Skipping this point."
                # )
                continue
            # Ensure x and y are within image bounds
            if x >= self.image_w or y >= self.image_h:
                # print(
                #     f"Coordinates out of bounds at index {i}: ({x}, {y}). Skipping this point."
                # )
                continue
            if disparities[i] < 5:
                # print(f"small disparity at index {i}. Skipping this point.")
                continue
            if disparities[i] > self.image_w // 2:
                # print(f"large disparity at index {i}. Skipping this point.")
                continue
            
            # Triangulation step - calculate depth
            z = self.fx * self.baseline / disparities[i] if disparities[i] != 0 else 0
            x_cam = (x - self.cx) * z / self.fx
            y_cam = (y - self.cy) * z / self.fy
            # # Store the mapping: keypoint_index -> (X, Y, Z)
            # self.point_cloud_map_curr[keypoint_idx] = np.array([x_cam, y_cam, z])
            disparity_map.append([x, y, z])
            ids.append(i)

        self.curr_rkpts_left = np.copy(ids)
        # self.left_keypoints.append(mkpts_left[ids])
        self.left_feature_indices.append(self.current_feature_mask[ids])
        return np.array(disparity_map, dtype=np.float32)

    def stereo_matching(self, i_left, i_right, confidence_threshold=0.5):
        """Perform stereo matching to compute disparity map."""

        if i_left is None or i_right is None:
            print("Error: One or both images are None.")
            return None
        self.image_h = i_left.shape[0]
        self.image_w = i_left.shape[1]
        left_tensor = self.preprocess_for_matching(i_left)
        right_tensor = self.preprocess_for_matching(i_right)

        if self.matcher_type == "disk":
            features_left = self.extract_features(left_tensor)[0]
            features_right = self.extract_features(right_tensor)[0]
            idxs, confidences = self.match_features(
                features_left,
                features_right,
                i_left.shape[1:],
                i_right.shape[1:],
                self.device,
            )
            mask = confidences < confidence_threshold  # Lower distances are better
            idxs = idxs[mask.squeeze()]
            mkpts_left = features_left.keypoints[idxs[:, 0]].cpu().numpy()
            mkpts_right = features_right.keypoints[idxs[:, 1]].cpu().numpy()

        elif self.matcher_type == "loftr":
            matches = self._match_features_loftr(
                left_tensor, right_tensor, num_features=2048
            )
            features_left = matches["keypoints0"].cpu().numpy()
            features_right = matches["keypoints1"].cpu().numpy()
            confidences = matches["confidence"].cpu().numpy()
            mask = confidences < confidence_threshold  # Lower distances are better
            mkpts_left = features_left[mask]
            mkpts_right = features_right[mask]

        elif self.matcher_type == "superpoint":
            features_left = self.extract_features(left_tensor)
            features_right = self.extract_features(right_tensor)
            matches, scores = self.match_features(
                features_left,
                features_right,
                i_left.shape[1:],
                i_right.shape[1:],
                self.device,
            )
            points0 = features_left["keypoints"][0][matches[:, 0]]  # shape [K, 2]
            points1 = features_right["keypoints"][0][matches[:, 1]]  # shape [K, 2]
            # matches0: indices in left, -1 if no match; matches1: indices in right, -1 if no match
            valid = scores > confidence_threshold
            mkpts_left = points0[valid].cpu().numpy()
            mkpts_right = points1[valid].cpu().numpy()
            self.current_feature_mask = matches[valid, 0].cpu().numpy()
            self.left_features.append(features_left)

        else:
            raise ValueError("Unknown matcher_type for stereo_matching.")

        # filter matches - keep only those around the same y-coordinate (epipolar constraint)
        # y_coords_left = mkpts_left[:, 1]
        # y_coords_right = mkpts_right[:, 1]
        # mask = np.abs(y_coords_left - y_coords_right) < 1.0
        # mkpts_left = mkpts_left[mask]
        # mkpts_right = mkpts_right[mask]

        print(f"Found {len(mkpts_left)} matches after confidence filtering.")
        if len(mkpts_left) < 10:
            print(
                "Warning: Insufficient matches for transformation estimation. Skipping this image."
            )
            return None
        # vslam.visualize_matches(
        #     i_left=i_left,
        #     i_right=i_right,
        #     mkpts_left=mkpts_left,
        #     mkpts_right=mkpts_right,
        # )
        print("Estimating disparity map...")
        # Compute disparity map using the matched keypoints
        disparity_map = self.compute_disparity_map(mkpts_left, mkpts_right)

        self.disparity_map = disparity_map


    def project_to_3d(self):
        """Projects 3D points from camera frame to gravity-aligned world frame using current pose."""
        if self.disparity_map is None:
            raise ValueError("Disparity map not computed. Run stereo_matching first.")

        # --- Step 1: Compute 3D points in camera frame ---
        pts_3d_cam = np.zeros((self.disparity_map.shape[0], 3))
        pts_3d_cam[:, 0] = (self.disparity_map[:, 0] - self.cx) / self.fx * self.disparity_map[:, 2]  # X = (u - cx) * Z / fx
        pts_3d_cam[:, 1] = (self.disparity_map[:, 1] - self.cy) / self.fy * self.disparity_map[:, 2]  # Y = (v - cy) * Z / fy
        pts_3d_cam[:, 2] = self.disparity_map[:, 2]  # Z = depth from disparity

        # --- Step 2: Transform points to IMU/body frame ---
        pts_3d_hom = np.hstack([pts_3d_cam, np.ones((len(pts_3d_cam), 1))]).T  # Convert to homogeneous (4xN)
        pts_3d_imu = self.T_SB @ pts_3d_hom
        pts_3d_imu = pts_3d_imu[:3, :].T  # Convert back to 3D points (N, 3)

        # --- Step 3: Transform points to world frame using current pose ---
        if not self.trajectory:
            T_abs_world = np.eye(4)  # Identity if no pose yet
        else:
            T_abs_world = self.trajectory[-1]  # Latest pose in world frame

        # World frame is gravity-aligned (R_align: IMU → World)
        R_imu_to_world = self.R_align  # Rotation part
        t_imu_to_world = T_abs_world[:3, 3]  # Translation part

        # Apply transformation: p_world = R_imu_to_world @ p_imu + t_imu_to_world
        pts_3d_world = (R_imu_to_world @ pts_3d_imu.T).T + t_imu_to_world

        # Store points
        self.global_map_points.append(pts_3d_world)

    def get_consecutive_keypoints(self):
        """Get consecutive keypoints from the left keypoints."""
        if len(self.left_features) < 2:
            print("Not enough keypoints to get consecutive pairs.")
            return None, None

        features1 = self.left_features[-2]
        features2 = self.left_features[-1]
        used_ids = self.left_feature_indices[-2]

        # Use LightGlue matcher
        with torch.inference_mode():
            matches = self.lg_matcher({"image0": features1, "image1": features2})

        # Extract matches and scores
        matches01 = matches["matches"][0]  # shape [K, 2]
        scores = matches["scores"][0]  # shape [K]

        # Filter valid matches (scores > threshold, optional)
        valid = scores > 0.9  # or set a threshold if you want
        matches01_v = matches01[valid]
        matched_indices = matches01[valid][:, 0].cpu().numpy()
        mask = np.isin(matched_indices, used_ids)
        mask_for_3d = np.isin(used_ids, matched_indices)
        indexes_3d = np.where(mask_for_3d)[0]
        indexes_in_matched = np.where(mask)[0]

        # Get corresponding keypoints to 3D points
        kpts2 = features2["keypoints"][0][matches01_v[:, 1]].cpu().numpy()
        tied_kpts2 = kpts2[indexes_in_matched]
        # Filter keypoints based on used_ids
        if len(indexes_in_matched) < 4:
            print("No valid keypoints found in the last two frames.")
            return None, None
        return tied_kpts2, indexes_3d

    # def pose_estimation_pnp(self, keypoints, indexes_3d):
    #     """Estimate pose using PnP (3D_t -> 2D_t+1)."""
    #     if not len(self.pts_map3d):
    #         print("No 3D points available for pose estimation.")
    #         return

    #     # 3D pt of t, and 2D pt of t+1
    #     pts_3d = self.pts_map3d[-2]
    #     pts_2d = keypoints
    #     pts_3d_tied = pts_3d[indexes_3d]
    #     if len(pts_3d_tied) < 4 or len(pts_2d) < 4:
    #         print(
    #             " -- Not enough points for pose estimation. Need at least 4 points. --"
    #         )
    #         return
    #     # Solve PnP
    #     success, rvec, tvec, inliers = cv2.solvePnPRansac(
    #         pts_3d_tied, pts_2d, self.intrinsics, self.dist_coeffs
    #     )
    #     if not success:
    #         print("PnP failed.")
    #         return

    #     # Convert rvec to rotation matrix
    #     R, _ = cv2.Rodrigues(rvec)
    #     # Extract translation vector
    #     t = tvec.flatten()

    #     # Build relative transformation matrix T_rel
    #     T_rel = np.eye(4)
    #     T_rel[:3, :3] = R
    #     T_rel[:3, 3] = (
    #         t  # NOTE: This is translation in the coordinate frame of the 3D points
    #     )
    #     # project to world frame / IMU
    #     T_rel = self.transform_pose_cam_to_imu_fix(T_rel)  # Transform to IMU frame

    #     # T_rel_inv = np.linalg.inv(T_rel)
    #     # Compose with last global pose
    #     T_prev = self.trajectory[-1]
    #     T_new = T_prev @ T_rel

    #     # Store new global pose
    #     self.trajectory.append(T_new)

    def create_positions(self, trajectories=[], initial_pose=None):
        """
        Create positions and rotations from the trajectory, optionally transforming from an initial pose.
        Also returns full world-frame poses.
        """
        if not trajectories and not self.trajectory:
            print("No trajectory available to create positions.")
            return

        positions = []
        rotations = []
        world_poses = []

        # Select trajectory source
        source = trajectories if trajectories else self.trajectory

        # Apply initial pose if given
        if initial_pose is not None:
            T_init = np.eye(4)
            T_init[:3, 3] = initial_pose  # or use full GT pose if available
            for T in source:
                T_world = T_init @ T
                world_poses.append(T_world)
                positions.append(T_world[:3, 3])
                rotations.append(T_world[:3, :3])
        else:
            for T in source:
                world_poses.append(T)
                positions.append(T[:3, 3])
                rotations.append(T[:3, :3])

        positions = np.array(positions)
        rotations = np.array(rotations)
        world_poses = np.array(world_poses)

        if not trajectories:
            self.positions = positions
            self.rotations = rotations
            self.world_poses = world_poses

        return positions, rotations, world_poses

    def create_positions_dynamic(self, trajectories=None, initial_pose=None):
        """
        Update positions, rotations, and world_poses from the trajectory or a provided list.
        If trajectories is given, append all values to self.v_positions, self.v_rotations, self.v_world_poses.
        Otherwise, update self.positions, self.rotations, self.world_poses from self.trajectory.
        Returns arrays: positions, rotations, world_poses (or visual versions if trajectories is given).
        """
        if trajectories is not None:
            cam_pos = np.zeros((3,))  # Initialize camera position
            # Visual odometry trajectory
            if not hasattr(self, "v_positions") or self.v_positions is None:
                self.v_positions = []
            if not hasattr(self, "v_rotations") or self.v_rotations is None:
                self.v_rotations = []
            if not hasattr(self, "v_world_poses") or self.v_world_poses is None:
                self.v_world_poses = []

            source = trajectories
            start_idx = len(self.v_world_poses)
            for T in source[start_idx:]:
                if initial_pose is not None and start_idx == 0:
                    T_init = np.eye(4)
                    T_init[:3, 3] = initial_pose
                    T_world = T_init @ T
                else:
                    T_world = T
                self.v_world_poses.append(T_world)
                self.v_positions.append(T_world[:3, 3])
                self.v_rotations.append(T_world[:3, :3])
                #cam_pos = cam_pos @ T_world[:3, :3].T + T_world[:3, 3]
                
                self.cam_positions.append(T_world[:3, 3])
                start_idx += 1

            return (
                np.array(self.v_positions),
                np.array(self.v_rotations),
                np.array(self.v_world_poses),
                np.array(self.cam_positions)
            )

        else:
            # Main SLAM trajectory
            if not self.trajectory or len(self.trajectory) == 0:
                print("No trajectory available to create positions.")
                self.positions = np.empty((0, 3))
                self.rotations = np.empty((0, 3, 3))
                self.world_poses = np.empty((0, 4, 4))
                self.cam_positions = np.empty((0, 3))
                return self.positions, self.rotations, self.world_poses, self.cam_positions

            if not hasattr(self, "positions") or self.positions is None:
                self.positions = []
            if not hasattr(self, "rotations") or self.rotations is None:
                self.rotations = []
            if not hasattr(self, "world_poses") or self.world_poses is None:
                self.world_poses = []

            source = self.trajectory
            start_idx = len(self.world_poses)
            for T in source[start_idx:]:
                if initial_pose is not None and start_idx == 0:
                    T_init = np.eye(4)
                    T_init[:3, 3] = initial_pose
                    T_world = T_init @ T
                else:
                    T_world = T
                self.world_poses.append(T_world)
                self.positions.append(T_world[:3, 3])
                self.rotations.append(T_world[:3, :3])
                # calculate actual camera position
                
                self.cam_positions.append(T_world[:3, 3])
                start_idx += 1

            return (
                np.array(self.positions),
                np.array(self.rotations),
                np.array(self.world_poses),
                np.array(self.cam_positions)
            )

    def estimate_visual_odometry(self, img1, img2, K=None, confidence_threshold=0.99):
        """
        Estimate the relative pose (4x4 matrix) between two consecutive images using SuperPoint + LightGlue.
        Args:
            img1: First image (numpy array, grayscale, already loaded and undistorted)
            img2: Second image (numpy array, grayscale, already loaded and undistorted)
            K: camera intrinsics matrix (3x3). If None, uses self.intrinsics.
            confidence_threshold: match confidence threshold for LightGlue.
        Returns:
            T_rel: 4x4 relative pose matrix (from img1 to img2), or None if failed.
        """
        if K is None:
            K = self.intrinsics

        tensor1 = self.preprocess_for_matching(img1)
        tensor2 = self.preprocess_for_matching(img2)
        features1 = self.extract_features(tensor1)
        features2 = self.extract_features(tensor2)

        # Match features
        with torch.inference_mode():
            matches = self.lg_matcher({"image0": features1, "image1": features2})
        matches01 = matches["matches"][0]
        scores = matches["scores"][0]
        valid = scores > confidence_threshold
        if valid.sum() < 8:
            print(f"Not enough matches for visual odometry.")
            return None
        idx1 = matches01[valid][:, 0].cpu().numpy()
        idx2 = matches01[valid][:, 1].cpu().numpy()
        kpts1 = features1["keypoints"][0][idx1].cpu().numpy()
        kpts2 = features2["keypoints"][0][idx2].cpu().numpy()

        # Estimate Essential matrix and recover pose
        E, mask = cv2.findEssentialMat(
            kpts1, kpts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None or mask.sum() < 8:
            print(f"Essential matrix estimation failed for visual odometry.")
            return None
            # Recover pose in camera frame
        _, R_rel_cam, t_rel_cam, _ = cv2.recoverPose(E, kpts1, kpts2, K)
        #t_rel_cam[1] = -t_rel_cam[1]  # Invert y-axis to match world frame
        T_rel_cam = np.eye(4)
        T_rel_cam[:3, :3] = R_rel_cam
        T_rel_cam[:3, 3] = t_rel_cam.flatten()
        #t_rel_cam[1] = - t_rel_cam[1]  # Invert y-axis to match world frame

        # Convert to IMU frame
        #T_rel_imu = self.T_SB @ T_rel_cam @ self.T_BS  # Transform to IMU frame
        T_rel_imu = self.transform_pose_cam_to_imu_fix(T_rel_cam)
        # Align to gravity frame
        R_gravity_to_world = self.R_align
        T_rel_world = np.eye(4)
        T_rel_world[:3, :3] = R_gravity_to_world @ T_rel_imu[:3, :3] @ R_gravity_to_world.T
        T_rel_world[:3, 3] = R_gravity_to_world @ T_rel_imu[:3, 3]
        # insert scale
        #scale = self.calculate_scale_from_stereo(kp1, kp2, matches, depth_map)
        T_rel_world[:3, 3] *= 0.3

        T_abs_world = self.trajectory[-1] @ T_rel_world
        self.trajectory.append(T_abs_world)
        # 4. Return the relative motion in the gravity-aligned world frame
        return T_abs_world, T_rel_imu
    
    
    def calculate_scale_from_stereo(self, kp1, kp2, matches, depth_map=None):
        """
        Calculate scale factor from stereo information
        
        Args:
            kp1: Keypoints from previous frame
            kp2: Keypoints from current frame
            matches: Matches between frames
            depth_map: Depth map if available
            
        Returns:
            scale_factor: Scale factor for visual odometry
        """
        if depth_map is not None:
            # Use depth map to calculate scale
            depths = []
            for m in matches:
                x, y = int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])
                if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                    depth = depth_map[y, x]
                    if depth > 0:
                        depths.append(depth)
            
            if len(depths) > 0:
                # Median depth gives scale
                median_depth = np.median(depths)
                return 1.0 / median_depth
        
        # If no depth map or failed, use baseline
        if self.baseline is not None:
            return self.baseline
        
        # Default scale (can be calibrated)
        return 1.0

    def visualize_matches(self, i_left, i_right, mkpts_left, mkpts_right):
        """Visualize matched keypoints."""
        if len(mkpts_left) == 0 or len(mkpts_right) == 0:
            print("No matched keypoints to visualize.")
            return

            # Ensure keypoints are numpy arrays
        mkpts_left = np.array(mkpts_left)
        mkpts_right = np.array(mkpts_right)

        # Create a canvas with the two images side by side
        h1, w1 = i_left.shape
        h2, w2 = i_right.shape
        canvas = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
        canvas[:h1, :w1] = i_left
        canvas[:h2, w1 : w1 + w2] = i_right

        # Plot using matplotlib
        plt.figure(figsize=(12, 6))
        plt.imshow(canvas, cmap="gray")
        for (x1, y1), (x2, y2) in zip(mkpts_left, mkpts_right):
            plt.plot([x1, x2 + w1], [y1, y2], color="lime", linewidth=0.8)
            plt.scatter([x1, x2 + w1], [y1, y2], color="red", s=5)
        plt.axis("off")
        plt.title("Matched Keypoints")
        plt.show()

    def visualize_3d_map(self, ground_truth):
        "Visualize the 3D proccesed pcl vs. ground truth using plotly."
        import plotly.graph_objects as go

        if self.pts_map3d is None:
            raise ValueError("3D points map is not computed. Run project_to_3d first.")
        pts_3d = self.pts_map3d[0]  # Assuming we visualize the first set of 3D points
        if pts_3d.ndim == 1:
            pts_3d = pts_3d.reshape(-1, 3)
        if pts_3d.shape[1] != 3:
            raise ValueError("3D points must be of shape (N, 3)")
        fig = go.Figure()
        # NOTE: Swap y and z axes for visualization to fit IMU coordinates
        fig.add_trace(
            go.Scatter3d(
                x=pts_3d[:, 0],
                y=pts_3d[:, 2],
                z=-pts_3d[:, 1],
                mode="markers",
                marker=dict(size=2, color="blue"),
                name="Processed Points",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=self.positions[:, 0],
                y=self.positions[:, 2],
                z=-self.positions[:, 1],
                mode="lines+markers",
                marker=dict(size=4, color="green"),
                line=dict(color="green", width=2),
                name="Camera Positions",
            )
        )
        if ground_truth is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=ground_truth[:, 0],
                    y=ground_truth[:, 1],
                    z=ground_truth[:, 2],
                    mode="markers",
                    marker=dict(size=2, color="red"),
                    name="Ground Truth",
                )
            )
        fig.update_layout(
            title="3D Points Map",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            width=800,
            height=600,
        )

        fig.show()

    def visualize_3d_map_with_imu(
        self, ground_truth, imu_positions, visual_trajectory=None
    ):
        import plotly.graph_objects as go

        if self.pts_map3d is None or len(self.pts_map3d) == 0:
            raise ValueError("3D points map is not computed. Run project_to_3d first.")
        pts_3d = self.pts_map3d[0]
        if pts_3d.ndim == 1:
            pts_3d = pts_3d.reshape(-1, 3)
        if pts_3d.shape[1] != 3:
            raise ValueError("3D points must be of shape (N, 3)")
        fig = go.Figure()
        imu_positions = imu_positions.astype(np.float32)
        fig.add_trace(
            go.Scatter3d(
                x=pts_3d[:, 0],
                y=pts_3d[:, 2],
                z=-pts_3d[:, 1],
                mode="markers",
                marker=dict(size=2, color="blue"),
                name="Processed Points",
            )
        )
        if visual_trajectory is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=visual_trajectory[:, 0],
                    y=visual_trajectory[:, 1],
                    z=visual_trajectory[:, 2],
                    mode="lines+markers",
                    marker=dict(size=4, color="purple"),
                    line=dict(color="purple", width=2),
                    name="Visual Trajectory",
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=self.positions[:, 0],
                y=self.positions[:, 2],
                z=-self.positions[:, 1],
                mode="lines+markers",
                marker=dict(size=4, color="green"),
                line=dict(color="green", width=2),
                name="Camera Positions",
            )
        )
        if len(imu_positions):
            fig.add_trace(
                go.Scatter3d(
                    x=imu_positions[:, 0],
                    y=imu_positions[:, 1],
                    z=imu_positions[:, 2],
                    mode="lines+markers",
                    marker=dict(size=4, color="orange"),
                    line=dict(color="orange", width=2),
                    name="IMU Dead-Reckoning",
                )
            )
        if ground_truth is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=ground_truth[:, 0],
                    y=ground_truth[:, 1],
                    z=ground_truth[:, 2],
                    mode="markers",
                    marker=dict(size=2, color="red"),
                    name="Ground Truth",
                )
            )
        fig.update_layout(
            title="3D Points Map",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            width=800,
            height=600,
        )
        fig.show()

    def transform_pose_cam_to_imu(self, T_cam):
        """
        Transform a pose from camera frame to IMU/body frame using self.extrinsics (T_BS).
        Args:
            T_cam: 4x4 pose in camera frame
        Returns:
            T_imu: 4x4 pose in IMU/body frame
        """
        T_BS = self.extrinsics # Body to Stereo camera transformation
        T_SB = np.linalg.inv(T_BS) # Stereo camera to Body transformation
        return T_SB @ T_cam

    def transform_pose_to_gravity_frame(self, T_pose):
        """Transform a 4x4 pose matrix to the gravity-aligned frame."""
        T_aligned = np.eye(4)
        T_aligned[:3, :3] = self.R_align @ T_pose[:3, :3] @ self.R_align.T
        T_aligned[:3, 3] = self.R_align @ T_pose[:3, 3]
        return T_aligned
    
    def transform_gravity_frame_to_pose(self, T_gravity):
        """Transform a 4x4 pose matrix from the gravity-aligned frame to the original frame."""
        T_pose = np.eye(4)
        T_pose[:3, :3] = self.R_align.T @ T_gravity[:3, :3] @ self.R_align
        T_pose[:3, 3] = self.R_align.T @ T_gravity[:3, 3]
        return T_pose

    def transform_pose_cam_to_imu_fix(self, T_cam):
        """
        Transform a pose from camera frame to IMU/body frame using self.extrinsics (T_BS).
        
        Args:
            T_cam: 4x4 transformation matrix in camera frame (relative pose from cam1 to cam2)
        Returns:
            T_imu: 4x4 transformation matrix in IMU/body frame (relative pose from imu1 to imu2)
        """
        # Extract rotation and translation from camera pose
        R_cam = T_cam[:3, :3]
        t_cam = T_cam[:3, 3]
        
        # Transform rotation: R_SB @ R_cam @ R_BS
        R_BS = self.T_BS[:3, :3]
        R_SB = self.T_SB[:3, :3]
        R_imu = R_SB @ R_cam @ R_BS
        
        # Transform translation: R_SB @ t_cam
        t_imu = R_SB @ t_cam
        
        # Construct the transformed 4x4 matrix
        T_imu = np.eye(4)
        T_imu[:3, :3] = R_imu
        T_imu[:3, 3] = t_imu

        return T_imu

    def set_alignment_matrix(self, R_align):
        """Set the alignment matrix for transforming poses to gravity-aligned frame."""
        self.R_align = R_align




