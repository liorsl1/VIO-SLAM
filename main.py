from imu_utils import align_ground_truth_to_gravity_world, align_ground_truth_to_z_gravity_frame, create_gravity_aligned_coordinate_system, transform_imu_to_z_gravity_frame, transform_vicon_to_body_frame
from rel_pose_vis import RelativePoseVisualizer
from vslam import *
from factor_graph_vio import VIFusionGraphISAM2
import os
import cv2
import numpy as np
import yaml
import pandas as pd
from scipy.spatial.transform import Rotation as R

from data_manager import DataManager
from vslam import vSLAM
from factor_graph_vio import VIFusionGraphISAM2
from imu_ekf import IMUEKF, update_with_ground_truth
from rel_pose_vis import RelativePoseVisualizer
from pyivsta_vis import RealTimeSLAMPyVistaVisualizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    data_dir = "f:/Code/Dataset/V1_01_e/mav0"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return

    # --- Data Loading and Calibration ---
    data_manager = DataManager(data_dir)
    data_manager.load_data()

    # Camera calibration
    K1, dist_coeffs, new_K1 = data_manager.load_camera_calib("cam0")
    K2, dist_coeffs, new_K2 = data_manager.load_camera_calib("cam1")
    baseline = data_manager.get_baseline(K1, K2)
    print(f"Baseline between cam0 and cam1: {baseline} meters")

    # --- Initial State and Biases ---
    # Get initial position and velocity from ground truth (or set to zero)
    initial_pos, initial_vel, _ = data_manager.get_initial_state_from_gt()

    # Calculate biases and initial orientation from static IMU data
    true_bias_accel, bias_gyro, initial_ori_quat, world_gravity_vec = \
        data_manager.calculate_initial_biases_and_gravity()
    
    print(f"Gravity-aligned initial orientation (quat): {initial_ori_quat}")
    data_manager.compute_alignment_rotation()
    #bias_gyro = np.array(bias_gyro)  # Ensure bias is a numpy array
    #bias_gyro = data_manager.R_align_IMU_to_gravity @ bias_gyro  # Transform gyro bias to gravity-aligned frame
    # --- IMU EKF and Factor Graph Initialization ---
    imu_ekf = IMUEKF(
        init_position=initial_pos,
        init_velocity=initial_vel,
        init_orientation_quat=initial_ori_quat, # Use gravity-aligned orientation
        accel_bias=true_bias_accel,
        gyro_bias=bias_gyro,
    )
    # The EKF's physics model now operates in the ideal world frame
    imu_ekf.g = world_gravity_vec
    #imu_ekf.R_align = data_manager.R_align_IMU_to_gravity  # Set the alignment rotation matrix
    # --- Factor Graph Initialization ---
    fg_vio = VIFusionGraphISAM2(
        initial_pose=np.concatenate([initial_pos, initial_ori_quat]), # Use gravity-aligned orientation
        initial_vel=initial_vel,
        initial_bias_accel=true_bias_accel,
        initial_bias_gyro=bias_gyro,
        # gravity_vector=world_gravity_vec, # Use ideal world gravity
        # R_align=data_manager.R_align_IMU_to_gravity,  # Alignment rotation matrix
    )
    #fg_vio.n_gravity = world_gravity_vec  # Set the gravity vector in the factor graph
    # --- vSLAM Initialization ---
    vslam = vSLAM(
        matcher_type="superpoint",
        baseline=baseline,
        intrinsics=new_K1,
        dist_coeffs=dist_coeffs,
    )
    vslam.initialize()
    vslam.set_alignment_matrix(data_manager.R_align_IMU_to_gravity)
    vslam.trajectory = [np.eye(4)]  # Initialize trajectory as identity matrix
    vslam.trajectory[0][:3, 3] = initial_pos  # Set initial position
    py_vis = RealTimeSLAMPyVistaVisualizer()
    R_uncertainty_ekf = np.eye(3) * 0.5
    t_uncertainty_ekf = np.eye(3) * 0.2
    R_uncertainty = np.ones(3) * 0.05
    t_uncertainty = np.ones(3) * 0.05

    prev_left_image = None
    prev_cam_timestamp = None
    trajectory_ekf = []
    trajectory_fg = []
    vis_trajectories = []
    synced_gt_poses = []
    step = 20
    R_flipx = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])  # Flip X and Y axes to align with gravity world frame
    
    # --- Main Loop ---
    # The rest of your main loop can remain largely the same.
    for idx, (row, left_img, right_img) in enumerate(data_manager.iter_stereo_frames(step=step)):
        # trajectory_fg = []
        current_cam_timestamp = row["timestamp"]

        # Undistort images
        undistorted_left = cv2.undistort(left_img, K1, dist_coeffs, None, new_K1)
        undistorted_right = cv2.undistort(right_img, K2, dist_coeffs, None, new_K2)

        # Stereo matching and 3D projection
        vslam.stereo_matching(
            i_left=undistorted_left,
            i_right=undistorted_right,
            confidence_threshold=0.99,
        )
        vslam.project_to_3d()

        # Visual odometry
        if prev_left_image is not None:
            v_trajectory, T_rel = vslam.estimate_visual_odometry(
                prev_left_image,
                undistorted_left,
                K=new_K1,
                confidence_threshold=0.985,
            )
            if v_trajectory is not None:
                vis_trajectories.append(v_trajectory)

        v_pos, v_rot, v_T, cam_pos = vslam.create_positions_dynamic(
            trajectories=vis_trajectories, initial_pose=initial_pos
        )

        # Keypoint matching for PnP (optional)
        if idx > 0:
            keypts, indexes_3d = vslam.get_consecutive_keypoints()
            if keypts is not None and indexes_3d is not None:
                # vslam.pose_estimation_pnp(keypts, indexes_3d)
                pass

        # --- IMU Integration between frames ---
        if prev_cam_timestamp is not None:
            for imu_row in data_manager.iter_imu_between(prev_cam_timestamp, current_cam_timestamp):
                gyro = imu_row[["w_x", "w_y", "w_z"]].values.astype(float)
                acc = imu_row[["a_x", "a_y", "a_z"]].values.astype(float)
                timestamp = imu_row["timestamp"]
                fg_vio.add_imu_measurement(acc, gyro, timestamp)
                imu_ekf.predict(acc, gyro, timestamp)

        # --- EKF Visual Update ---
        if len(v_pos):
            ekf_pose = imu_ekf.get_pose()
            ekf_velocity = ekf_pose['velocity']
            ekf_orientation = ekf_pose['orientation_quat']
            print(f"EKF state after IMU integration - Velocity: {ekf_velocity}, Orientation: {ekf_orientation}")

            R_rel = T_rel[:3, :3]
            t_rel = T_rel[:3, 3]
            # transform back to IMU frame
            rotvec = R.from_matrix(R_rel).as_rotvec()
            rel_pose = np.concatenate([rotvec, t_rel])

            before_update_P = imu_ekf.P.copy()
            #imu_ekf.update_with_vslam_relative_pose(R_rel, t_rel, R_uncertainty_ekf, t_uncertainty_ekf)
            imu_ekf.measure_update_impact(before_update_P, imu_ekf.P)
            ekf_pose = imu_ekf.get_pose()
            ekf_position = ekf_pose['position']
            trajectory_ekf.append(np.copy(ekf_position))

            # Factor graph visual update (periodically)
            if (idx %  (2)) == 0 and idx >= 4:
                try:
                    fg_vio.add_new_state()
                    pose_uncertainty = np.concatenate([R_uncertainty[:3], t_uncertainty[:3]])
                    #fg_vio.add_visual_measurement([R_rel, t_rel], 200)
                    # fg_vio.debug_graph()
                    positions, orientations = fg_vio.get_full_trajectory()
                    if len(positions) > 0:
                        trajectory_fg = positions
                except Exception as e:
                    print(f"Error getting full trajectory from factor graph: {e}")

        # Visualization
        points_3d = vslam.global_map_points[-1] if vslam.global_map_points else None
        
        # Get the synced and transformed ground truth pose for this timestamp
        gt_pos, gt_quat = data_manager.get_synced_gt_poses(row["timestamp"])

        if gt_pos is not None:
            # Align the GT pose to the gravity world frame
            synced_gt_poses.append(gt_pos)

        ekf_traj = np.array(trajectory_ekf) if trajectory_ekf else None
        fg_traj = np.array(trajectory_fg) if len(trajectory_fg) > 0 else None

        # Create a plottable GT trajectory from the synced list
        gt_traj_for_vis = np.array(synced_gt_poses) if synced_gt_poses else None

        py_vis.enqueue_update(points_3d, traj=cam_pos, gt_traj=gt_traj_for_vis, ekf_traj=ekf_traj, fg_traj=fg_traj)
        prev_cam_timestamp = current_cam_timestamp
        prev_left_image = undistorted_left.copy()
        print(f"Processed frame {idx + 1} at timestamp {current_cam_timestamp:.3f}")

    py_vis.close()

if __name__ == "__main__":
    main()
