from scipy.spatial.transform import Rotation as R
import numpy as np

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def quaternion_derivative(q, omega):
    # q: [x, y, z, w] (scipy format)
    # omega: angular velocity (rad/s)
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    Omega = np.array([
        [0,     -omega[0], -omega[1], -omega[2]],
        [omega[0],  0,      omega[2], -omega[1]],
        [omega[1], -omega[2],  0,      omega[0]],
        [omega[2], omega[1], -omega[0], 0]
    ])
    q_dot = 0.5 * Omega @ np.array([qw, qx, qy, qz])
    return np.array([q_dot[1], q_dot[2], q_dot[3], q_dot[0]])  # back to [x, y, z, w]

def update_with_ground_truth(ekf, gt_pos, gt_quat):
    # 1. Update position
    H_pos = np.zeros((3, 16))
    H_pos[:, 0:3] = np.eye(3)
    ekf.update(gt_pos, H_pos)

    # 2. Update orientation
    q_est = ekf.x[6:10]
    q_gt = gt_quat

    R_est = R.from_quat(q_est)
    R_gt = R.from_quat(q_gt)
    rotvec_error = (R_gt * R_est.inv()).as_rotvec()  # small angle approx

    H_ori = np.zeros((3, 16))
    H_ori[:, 6:9] = np.eye(3)  # maps to minimal orientation (approx.)
    # Call internal method directly to apply residual as-is
    ekf.update(rotvec_error, H_ori)



class IMUEKF:
    def __init__(self, init_position=None, init_velocity=None,
                 init_orientation_quat=None, accel_bias=None, gyro_bias=None):
        self.n = 16
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 0.1
        # IMU noise parameters
        self.gyro_noise_density = 1.6968e-04    # rad/s/√Hz
        self.gyro_random_walk = 1.9393e-05      # rad/s²/√Hz
        self.accel_noise_density = 2.0000e-3    # m/s²/√Hz
        self.accel_random_walk = 3.0000e-3      # m/s³/√Hz
        self.R_align = None
        
        # Process and measurement noise - will be set during predict()
        self.Q = np.zeros((self.n, self.n))
        self.R = np.eye(3) * 0.01

        #self.g = np.array([0, 0, 0])
        self.last_time = None
        self.cost_history = []
        if init_position is not None:
            self.x[0:3] = init_position
        if init_velocity is not None:
            self.x[3:6] = init_velocity
        if init_orientation_quat is not None:
            q = np.array(init_orientation_quat)
            self.x[6:10] = q / np.linalg.norm(q)
        else:
            self.x[6:10] = np.array([0, 0, 0, 1])
        if accel_bias is not None:
            self.x[10:13] = accel_bias
        if gyro_bias is not None:
            self.x[13:16] = gyro_bias

    def predict(self, a_m, w_m, timestamp):
        if self.last_time is None:
            self.last_time = timestamp
            return

        dt = timestamp - self.last_time
        self.last_time = timestamp
        
        if dt <= 0 or dt > 1.0:  # sanity check
            return

        # Extract state
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]  # [x, y, z, w] for scipy
        ba = self.x[10:13]
        bg = self.x[13:16]

        # Bias-corrected measurements
        # Transform measurements to body frame first:
        acc = a_m - ba
        omega = w_m - bg
        # acc = self.R_align @ acc  # Transform to world frame
        # omega = self.R_align @ omega  # Transform to world frame

        # Rotation matrix from body to world
        Rwb = R.from_quat(q).as_matrix()
        a_world = Rwb @ acc - self.g

        # Integrate position and velocity
        p += v * dt + 0.5 * a_world * dt**2
        v += a_world * dt

        # Integrate orientation using quaternion kinematics
        delta_angle = omega * dt
        delta_q = R.from_rotvec(delta_angle).as_quat()  # [x,y,z,w]
        q = R.from_quat(q) * R.from_quat(delta_q)
        q = q.as_quat()  # Returns [x,y,z,w]
        q = q / np.linalg.norm(q)


        # Update state
        self.x[0:3] = p
        self.x[3:6] = v
        self.x[6:10] = q

        # --- CORRECTED COVARIANCE PROPAGATION ---
        # Build the full state transition matrix (Jacobian F)
        F = np.eye(self.n)
        F[0:3, 3:6] = np.eye(3) * dt  # d(pos)/d(vel)
        
        # d(vel)/d(ori) = -Rwb * skew(acc) * dt
        F[3:6, 6:9] = -Rwb @ skew(acc) * dt
        
        # d(vel)/d(acc_bias) = -Rwb * dt
        F[3:6, 10:13] = -Rwb * dt
        
        # d(ori)/d(gyro_bias) = -Rwb * dt (in tangent space)
        F[6:9, 13:16] = -Rwb * dt

        # Build process noise covariance Q
        Q = np.zeros((12, 12))
        Q[0:3, 0:3] = np.eye(3) * self.accel_noise_density**2
        Q[3:6, 3:6] = np.eye(3) * self.gyro_noise_density**2
        Q[6:9, 6:9] = np.eye(3) * self.accel_random_walk**2
        Q[9:12, 9:12] = np.eye(3) * self.gyro_random_walk**2
        
        # Map noise to state space
        G = np.zeros((self.n, 12))
        G[3:6, 0:3] = -Rwb
        G[6:9, 3:6] = -Rwb
        G[10:13, 6:9] = np.eye(3)
        G[13:16, 9:12] = np.eye(3)
        
        Q_discrete = G @ Q @ G.T * dt

        # Propagate covariance
        self.P = F @ self.P @ F.T + Q_discrete
        #self.P *= inflation_factor  # Apply inflation

    def get_pose(self):
        return {
            "position": self.x[0:3],
            "velocity": self.x[3:6],
            "orientation_quat": self.x[6:10],
            "accel_bias": self.x[10:13],
            "gyro_bias": self.x[13:16]
        }

    def update(self, z, H, R=None):
        if R is None:
            R = self.R
        z_pred = H @ self.x
        y = z - z_pred
        self._measure_update(y, H, R)

    def _measure_update(self, y, H, R):
        if R is None:
            R = self.R
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ H) @ self.P
        
        """ Mahalanobis distance (chi-squared cost)
        It's a distance measure that accounts for uncertainty.
        Takes into account noise, correlation, covariance
        "How surprising is this new measurement, given what I already know about the system's uncertainty?" """
        cost = float(y.T @ np.linalg.inv(S) @ y)
        self.cost_history.append(cost)  # Add to a list in your class
        print(f" ---- Measurement update cost: {cost} -----")


    def _kalman_update(self, y, H, R_meas):
        """Kalman filter update step"""
        # Make sure y is a column vector with correct dimensions
        y = np.atleast_2d(y).reshape(-1, 1)
        
    
        # Calculate innovation covariance
        S = H @ self.P @ H.T + R_meas
        
        # Calculate Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state vector
        self.x = self.x + (K @ y).flatten()
        
        # Update covariance matrix
        I = np.eye(16)
        self.P = (I - K @ H) @ self.P
        

    def update_with_vslam(self, pos_meas, quat_meas, R_pos, R_quat):
        # --- Position update ---
        H_pos = np.zeros((3, 16))
        H_pos[:, 0:3] = np.eye(3)
        
        pos_meas = np.array(pos_meas).flatten()[:3]
        y_pos = pos_meas - self.x[0:3]
        
        # Calculate innovation covariance for position
        S_pos = H_pos @ self.P @ H_pos.T + R_pos
        
        # Chi-square test for position innovation
        NIS_pos = float(y_pos.T @ np.linalg.inv(S_pos) @ y_pos)
        print(f"Position NIS: {NIS_pos:.2f} (should be ~3.0)")
        
        # Check if update is reasonable before applying
        if NIS_pos > 10:  # 99% confidence threshold for 3 DOF
            print(f"WARNING: Large position innovation ({NIS_pos:.2f}), possible outlier!")
            # Increase measurement noise to trust this update less
            R_pos_adjusted = R_pos * (NIS_pos / 3.0)
            self._kalman_update(y_pos, H_pos, R_pos_adjusted)
        else:
            self._kalman_update(y_pos, H_pos, R_pos)

        # --- Orientation update ---
        # Similar approach for orientation with chi-square test
        q_est = self.x[6:10]
        q_meas = np.array(quat_meas).flatten()[:4]  # Ensure it's 4D
        q_meas = q_meas / np.linalg.norm(q_meas)  # Normalize

        # Convert quaternions to rotation matrices and compute error
        R_est = R.from_quat(q_est)
        R_meas = R.from_quat(q_meas)
        rotvec_error = (R_meas * R_est.inv()).as_rotvec()

        H_ori = np.zeros((3, 16))
        H_ori[:, 6:9] = np.eye(3)  # Small-angle approximation
        
        # Calculate innovation covariance for orientation
        S_ori = H_ori @ self.P @ H_ori.T + R_quat
        
        # Chi-square test for orientation innovation
        NIS_ori = float(rotvec_error.T @ np.linalg.inv(S_ori) @ rotvec_error)
        print(f"Orientation NIS: {NIS_ori:.2f}")
        
        # Check if update is reasonable before applying
        if NIS_ori > 20:  # 99% confidence threshold for 3 DOF
            print(f"WARNING: Large orientation innovation ({NIS_ori:.2f}), possible outlier!")
            # Increase measurement noise to trust this update less
            R_quat_adjusted = R_quat * (NIS_ori / 3.0)
            self._kalman_update(rotvec_error, H_ori, R_quat_adjusted)
        else:
            self._kalman_update(rotvec_error, H_ori, R_quat)


    def update_with_vslam_relative_pose(self, R_rel, t_rel, R_uncertainty=None, t_uncertainty=None):
        """Update EKF with relative pose from visual SLAM"""
        if R_uncertainty is None:
            R_uncertainty = np.eye(3) * 0.01
        if t_uncertainty is None:
            t_uncertainty = np.eye(3) * 0.01
        
        # Ensure correct dimensions
        R_rel = np.array(R_rel).reshape(3, 3)
        t_rel = np.array(t_rel).flatten()[:3]  # Ensure it's 3D
        
        # Extract current state
        p_curr = self.x[0:3]
        q_curr = self.x[6:10]  # quaternion [x,y,z,w]
        
        # Convert current quaternion to rotation matrix
        R_curr = R.from_quat(q_curr).as_matrix()
        
        # Compute absolute position from relative translation
        t_world = R_curr @ t_rel.reshape(3, 1)
        p_meas = p_curr + t_world.flatten()
        
        # Compute absolute orientation from relative rotation
        R_meas = R_curr @ R_rel
        q_meas = R.from_matrix(R_meas).as_quat()  # [x,y,z,w]
        
        # Update with absolute measurements
        self.update_with_vslam(p_meas, q_meas, t_uncertainty, R_uncertainty)

    
    def debug_uncertainty(self):
        """Print uncertainty (standard deviation) for each state variable"""
        std_devs = np.sqrt(np.diag(self.P))
        print("\n--- EKF State Uncertainty (1-sigma) ---")
        print(f"Position [m]: {std_devs[0:3]}")
        # print(f"Velocity [m/s]: {std_devs[3:6]}")
        # print(f"Orientation [rad]: {std_devs[6:9]}")
        # print(f"Accel bias: {std_devs[10:13]}")
        # print(f"Gyro bias: {std_devs[13:16]}")
        print("----------------------------------------\n")
        return std_devs
    
    def measure_update_impact(self, P_before, P_after):
        """Measure how much an update reduced uncertainty"""
        trace_before = np.trace(P_before)
        trace_after = np.trace(P_after)
        reduction_pct = 100 * (trace_before - trace_after) / trace_before
        print(f"Uncertainty reduced by: {reduction_pct:.2f}%")
        return reduction_pct


if __name__ == "__main__":
    # Example usage
    imu_ekf = IMUEKF()
    
    # Simulated measurements
    a_m = np.array([0.1, 0.2, 9.7])  # rad/s
    w_m = np.array([0.01, 0.02, 0.03])  # m/s^2
    timestamp = 1.0
    
    # Predict step
    imu_ekf.predict(a_m, w_m, timestamp)
    pose = imu_ekf.get_pose()
    print("Predicted pose:", pose)
    
    # Simulated ground truth
    gt_pos = np.array([1.0, 2.0, 3.0])
    gt_quat = np.array([0.0, 0.0, 0.0, 1.0])
    
    # Update with ground truth
    update_with_ground_truth(imu_ekf, gt_pos, gt_quat)
    pose = imu_ekf.get_pose()
    print("Updated pose with ground truth:", pose)
    
    # Simulated visual SLAM measurement
    vslam_pos_meas = np.array([1.1, 2.1, 3.1])
    vslam_quat_meas = np.array([0.0, 0.0, 0.1, 0.9])
    R_pos = np.eye(3) * 0.1
    R_quat = np.eye(3) * 0.01
    
    # Update with visual SLAM
    imu_ekf.update_with_vslam(vslam_pos_meas, vslam_quat_meas, R_pos, R_quat)
    pose = imu_ekf.get_pose()
    print("Updated pose with visual SLAM:", pose)
    
    # Simulated relative pose from visual SLAM
    R_rel = R.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
    t_rel = np.array([0.5, 0.5, 0.0])
    
    # Update with relative pose
    if R_rel is not None and t_rel is not None and t_rel.size > 0:
        # Make sure t_rel is a 3x1 vector
        t_rel = t_rel.reshape(3, 1) if t_rel.ndim == 1 else t_rel
        imu_ekf.update_with_vslam_relative_pose(R_rel, t_rel, R_uncertainty, t_uncertainty)
        ekf_pose = imu_ekf.get_pose()
        ekf_position = ekf_pose['position']
        ekf_orientation = ekf_pose['orientation_quat']
        trajectory_ekf.append(ekf_position)
    else:
        print("Skipping EKF update - invalid relative pose")
