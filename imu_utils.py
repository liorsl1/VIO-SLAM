import numpy as np
from scipy.spatial.transform import Rotation as R

def create_gravity_aligned_coordinate_system(accel_data, imu_hz):
    """
    Create a coordinate system transformation that aligns gravity with -Z axis
    
    Args:
        accel_data: Raw accelerometer data
        imu_hz: IMU sampling frequency
        
    Returns:
        R_align: 3x3 rotation matrix to transform from IMU frame to Z-gravity frame
        gravity_magnitude: Magnitude of gravity vector
        gravity_z_frame: Gravity vector in Z-frame [0, 0, -9.81]
    """
    # Calculate gravity vector from static period
    static_accel = accel_data[:2 * imu_hz]  # First 2 seconds
    gravity_imu_frame = static_accel.mean(axis=0)  # Gravity in IMU frame
    gravity_magnitude = np.linalg.norm(gravity_imu_frame)
    
    print(f"Measured gravity in IMU frame: {gravity_imu_frame}")
    print(f"Gravity magnitude: {gravity_magnitude:.3f} m/s²")
    
    # Normalize gravity vector
    gravity_unit = gravity_imu_frame / gravity_magnitude
    
    # Target: gravity should point along -Z axis (standard robotics convention)
    target_gravity = np.array([0, 0, -1])
    
    # Calculate rotation matrix to align current gravity with -Z axis
    if np.allclose(gravity_unit, target_gravity, atol=1e-3):
        R_align = np.eye(3)
    elif np.allclose(gravity_unit, -target_gravity, atol=1e-3):
        R_align = np.diag([1, 1, -1])
    else:
        # Use Rodrigues' rotation formula
        v = np.cross(gravity_unit, target_gravity)
        s = np.linalg.norm(v)
        c = np.dot(gravity_unit, target_gravity)
        
        if s > 1e-6:  # Avoid division by zero
            # Skew-symmetric matrix
            vx = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
        else:
            R_align = np.eye(3)
    
    # Verify transformation
    gravity_transformed = R_align @ gravity_unit
    print(f"Gravity after alignment: {gravity_transformed * gravity_magnitude}")
    
    # Standard gravity vector in Z-frame
    gravity_z_frame = np.array([0, 0, -gravity_magnitude])
    
    return R_align, gravity_magnitude, gravity_z_frame

def transform_imu_to_z_gravity_frame(acc, gyro, R_align, gravity_imu_frame):
    """
    Transform IMU measurements to Z-gravity coordinate system
    
    Args:
        acc: Accelerometer measurement in IMU frame
        gyro: Gyroscope measurement in IMU frame  
        R_align: Alignment rotation matrix
        gravity_imu_frame: Original gravity vector in IMU frame
        
    Returns:
        acc_aligned: Accelerometer in Z-gravity frame (gravity removed)
        gyro_aligned: Gyroscope in Z-gravity frame
    """
    # Transform to Z-gravity frame
    acc_aligned = R_align @ acc
    gyro_aligned = R_align @ gyro
    
    # Remove gravity (now it's [0, 0, -9.81] in aligned frame)
    gravity_aligned = R_align @ gravity_imu_frame
    acc_corrected = acc_aligned - gravity_aligned
    
    return acc_corrected, gyro_aligned

def align_ground_truth_to_z_gravity_frame(initial_pos, initial_vel, initial_ori_quat, R_align):
    """
    Align ground truth data to Z-gravity coordinate system
    
    Args:
        initial_pos: Initial position from ground truth
        initial_vel: Initial velocity from ground truth
        initial_ori_quat: Initial orientation quaternion [x, y, z, w]
        R_align: Alignment rotation matrix
        
    Returns:
        Aligned position, velocity, and orientation
    """
    # Transform position and velocity
    pos_aligned = R_align @ initial_pos
    vel_aligned = R_align @ initial_vel
    
    # Transform orientation quaternion
    # Convert quaternion to rotation matrix
    R_gt = R.from_quat(initial_ori_quat).as_matrix()
    
    # Apply alignment transformation
    R_gt_aligned = R_align @ R_gt
    
    # Convert back to quaternion
    ori_quat_aligned = R.from_matrix(R_gt_aligned).as_quat()
    
    return pos_aligned, vel_aligned, ori_quat_aligned

def transform_visual_pose_to_z_gravity_frame(T_rel_cam, T_cam_imu, R_align):
    """
    Transform visual odometry pose to Z-gravity coordinate system
    
    Args:
        T_rel_cam: Relative pose in camera frame
        T_cam_imu: Camera to IMU transformation (extrinsics)
        R_align: Alignment rotation matrix
        
    Returns:
        T_rel_aligned: Relative pose in Z-gravity frame
    """
    # First transform from camera to IMU frame
    T_rel_imu = T_cam_imu @ T_rel_cam @ np.linalg.inv(T_cam_imu)
    
    # Then transform to Z-gravity frame
    T_rel_aligned = np.eye(4)
    T_rel_aligned[:3, :3] = R_align @ T_rel_imu[:3, :3] @ R_align.T
    T_rel_aligned[:3, 3] = R_align @ T_rel_imu[:3, 3]
    
    return T_rel_aligned

def transform_vicon_to_body_frame(pos_vicon, vel_vicon, quat_vicon, T_vicon_body):
    """
    Transform ground truth data from Vicon frame to body frame
    
    Args:
        pos_vicon: Position in Vicon frame
        vel_vicon: Velocity in Vicon frame  
        quat_vicon: Quaternion in Vicon frame [x, y, z, w]
        T_vicon_body: 4x4 transformation matrix from Vicon to body frame
        
    Returns:
        Transformed position, velocity, and quaternion in body frame
    """
    # Transform position
    T_VB = np.linalg.inv(T_vicon_body)
    pos_homo = np.append(pos_vicon, 1.0)
    pos_body = (T_VB @ pos_homo)[:3]

    # Transform velocity (only rotation part)
    R_vicon_body = T_VB[:3, :3]
    vel_body = R_vicon_body @ vel_vicon
    
    # Transform orientation
    R_vicon = R.from_quat(quat_vicon).as_matrix()
    R_body = R_vicon_body @ R_vicon
    quat_body = R.from_matrix(R_body).as_quat()
    
    return pos_body, vel_body, quat_body

def transform_visual_pose_to_body_frame(T_rel_cam, T_cam_imu):
    """
    Transform visual odometry pose from camera frame to body frame
    
    Args:
        T_rel_cam: Relative pose in camera frame
        T_cam_imu: Camera to IMU transformation (T_BS)
        
    Returns:
        T_rel_body: Relative pose in body frame
    """
    # Transform from camera to body frame using extrinsics
    T_rel_body = T_cam_imu @ T_rel_cam @ np.linalg.inv(T_cam_imu)
    return T_rel_body

def from_two_vectors(v0, v1):
    """
    Calculates the rotation quaternion that transforms vector v0 to align with vector v1.
    Returns a quaternion in [x, y, z, w] format (for scipy).
    """
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    dot_product = np.dot(v0, v1)

    if dot_product > 0.999999:
        # Vectors are already aligned
        return np.array([0.0, 0.0, 0.0, 1.0])
    elif dot_product < -0.999999:
        # Vectors are opposite. Find an arbitrary orthogonal axis.
        axis = np.cross([1.0, 0.0, 0.0], v0)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross([0.0, 1.0, 0.0], v0)
        axis = axis / np.linalg.norm(axis)
        # 180-degree rotation quaternion
        return np.concatenate([axis, [0.0]])
    else:
        # Standard case
        s = np.sqrt(2 * (1 + dot_product))
        inv_s = 1.0 / s
        axis = np.cross(v0, v1) * inv_s
        w = 0.5 * s
        return np.array([axis[0], axis[1], axis[2], w])

def align_ground_truth_to_gravity_world(pos_gt, quat_gt, R_align):
    """
    Aligns ground truth data (position and orientation) from the Vicon world
    to the gravity-aligned world frame using a fixed alignment rotation.

    Args:
        pos_gt (np.array): Position(s) in the Vicon frame.
        quat_gt (np.array): Quaternion(s) in the Vicon frame [x,y,z,w].
        R_align (np.array): The 3x3 rotation matrix that aligns the Vicon world to the gravity world.

    Returns:
        pos_aligned (np.array): Position(s) in the gravity-aligned world.
        quat_aligned (np.array): Quaternion(s) in the gravity-aligned world.
    """
    from scipy.spatial.transform import Rotation as R

    # Ensure inputs are numpy arrays
    pos_gt = np.asarray(pos_gt)
    quat_gt = np.asarray(quat_gt)

    # Rotate position(s)
    # If single position (3,), reshape to (1, 3) for matmul, then flatten back
    is_single_pos = pos_gt.ndim == 1
    if is_single_pos:
        pos_gt = pos_gt.reshape(1, -1)
    
    pos_aligned = (R_align @ pos_gt.T).T
    
    if is_single_pos:
        pos_aligned = pos_aligned.flatten()

    # Rotate orientation(s)
    r_align = R.from_matrix(R_align)
    r_gt = R.from_quat(quat_gt)
    r_aligned = r_align * r_gt
    quat_aligned = r_aligned.as_quat()

    return pos_aligned, quat_aligned