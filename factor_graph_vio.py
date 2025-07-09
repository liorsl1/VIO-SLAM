import gtsam
import numpy as np
from gtsam.symbol_shorthand import X, V, B

class VIFusionGraphISAM2:
    def __init__(self, initial_pose, initial_vel, initial_bias_accel, initial_bias_gyro, gravity_vector=None, R_align=None):
        # ISAM2 setup
        self.isam = gtsam.ISAM2()
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()

        self.current_state_idx = 0
        self.prev_timestamp = None
        self.gravity_vector = gravity_vector
        self.R_align = R_align if R_align is not None else np.array([1, 0, 0, 0])  # Identity quaternion
        
        # IMU preintegration
        self.imu_params = self.create_imu_params()
        self.bias = gtsam.imuBias.ConstantBias(initial_bias_accel, initial_bias_gyro)
        self.imu_preintegrated = gtsam.PreintegratedImuMeasurements(self.imu_params, self.bias)

        # Add priors
        self.add_initial_factors(initial_pose, initial_vel)

    def create_imu_params(self):
        imu_params = gtsam.PreintegrationParams.MakeSharedU(-9.81)
        if self.gravity_vector is not None:
            imu_params.n_gravity = [0,0,0]
        
        # Realistic noise parameters (tuned for EuRoC)
        imu_params.setAccelerometerCovariance(np.eye(3) * (0.1)**2)  # Increased from original
        imu_params.setGyroscopeCovariance(np.eye(3) * (0.01)**2)     # Increased from original
        imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)
        imu_params.setAccelerometerCovariance(np.eye(3) * (0.03)**2)       # Added bias covariance
        #imu_params.setBiasOmegaCovariance(np.eye(3) * (0.001)**2)    # Added bias covariance
        return imu_params

    def add_initial_factors(self, pose_arr, vel_vec):
        # Convert pose array to GTSAM Pose3 (in IMU frame)
        if self.gravity_vector is not None:
            R_world_to_imu = gtsam.Rot3(self.R_align)
            # rotated_position = R_world_to_imu.rotate(gtsam.Point3(*pose_arr[:3]))
            # pose_arr[:3] = [rotated_position[0], rotated_position[1], rotated_position[2]]

        pose = gtsam.Pose3(
            gtsam.Rot3.Quaternion(pose_arr[6], pose_arr[3], pose_arr[4], pose_arr[5]),
            gtsam.Point3(*pose_arr[:3])
        )

        # Define noise models
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1] * 3 + [0.01] * 3))
        vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-3)

        # Add priors
        self.graph.add(gtsam.PriorFactorPose3(X(0), pose, pose_noise))
        self.graph.add(gtsam.PriorFactorVector(V(0), vel_vec, vel_noise))
        self.graph.add(gtsam.PriorFactorConstantBias(B(0), self.bias, bias_noise))

        # Initial estimates
        self.initial_estimates.insert(X(0), pose)
        self.initial_estimates.insert(V(0), vel_vec)
        self.initial_estimates.insert(B(0), self.bias)
        
        # Initial update
        self.isam.update(self.graph, self.initial_estimates)
        self.graph.resize(0)
        self.initial_estimates.clear()

    def add_imu_measurement(self, accel, gyro, timestamp):
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            return

        dt = timestamp - self.prev_timestamp
        self.prev_timestamp = timestamp
        # Define the transformation from the IMU-centric graph frame to the gravity-aligned world frame
        T_world_from_imu = gtsam.Pose3() # Default to identity
        if self.gravity_vector is not None:
            accel_world = self.R_align @ accel
            gyro_world = self.R_align @ gyro

        if dt <= 0 or dt > 1.0:
            return
        self.imu_preintegrated.integrateMeasurement(accel, gyro, dt)

    def add_new_state(self):
        if self.current_state_idx >= 20:  # Prevent unbounded growth
            self.isam.update()
            self.graph.resize(0)
            self.current_state_idx = 0

        i = self.current_state_idx
        j = i + 1

        try:
            # Get previous state
            current_estimate = self.isam.calculateEstimate()
            if not (current_estimate.exists(X(i)) and current_estimate.exists(V(i))):
                return False
                
            prev_pose = current_estimate.atPose3(X(i))
            prev_vel = current_estimate.atVector(V(i))
            
            # Update bias estimate
            if current_estimate.exists(B(i)):
                self.bias = current_estimate.atConstantBias(B(i))
            
            # Create new preintegration object BEFORE prediction
            new_preintegrated = gtsam.PreintegratedImuMeasurements(self.imu_params, self.bias)
            
            # Predict new state
            predicted_state = self.imu_preintegrated.predict(gtsam.NavState(prev_pose, prev_vel), self.bias)

            # Add IMU factor
            self.graph.add(gtsam.ImuFactor(X(i), V(i), X(j), V(j), B(i), self.imu_preintegrated))
            
            # Add bias evolution factor with higher uncertainty
            bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 0.1)
            self.graph.add(gtsam.BetweenFactorConstantBias(B(i), B(j), gtsam.imuBias.ConstantBias(), bias_noise))

            # Add initial estimates
            self.initial_estimates.insert(X(j), predicted_state.pose())
            self.initial_estimates.insert(V(j), predicted_state.velocity())
            self.initial_estimates.insert(B(j), self.bias)

            # Update ISAM2 and swap preintegration object
            self.isam.update(self.graph, self.initial_estimates)
            self.imu_preintegrated = new_preintegrated
            
            self.graph.resize(0)
            self.initial_estimates.clear()
            self.current_state_idx = j
            return True
            
        except Exception as e:
            print(f"Error in add_new_state: {e}")
            return False

    def add_visual_measurement(self, rel_pose_world, num_matches):
        if self.current_state_idx < 1:
            return False

        try:
            # Convert world-frame relative pose to IMU frame
            current_estimate = self.isam.calculateEstimate()
            T_prev_imu = current_estimate.atPose3(X(self.current_state_idx-1))
            
            # Create relative pose in GTSAM format
            rel_pose_gtsam = gtsam.Pose3(
                gtsam.Rot3.Rodrigues(rel_pose_world[:3]),
                gtsam.Point3(*rel_pose_world[3:])
            )
            
            # Transform to IMU frame
            #rel_pose_imu = T_prev_imu.between(rel_pose_gtsam)

            # Adaptive noise based on match quality
            # uncertainty = 0.1 / max(num_matches, 1)
            # noise = gtsam.noiseModel.Isotropic.Sigma(6, uncertainty)
            # Adaptive noise
            trans_noise = 0.001
            rot_noise = 0.0005
            noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([trans_noise]*3 + [rot_noise]*3)
            )
            
            self.graph.add(gtsam.BetweenFactorPose3(
                X(self.current_state_idx-1),
                X(self.current_state_idx),
                rel_pose_gtsam,
                noise
            ))
            
            # Update ISAM2
            self.isam.update(self.graph, gtsam.Values())
            self.graph.resize(0)
            return True
            
        except Exception as e:
            print(f"Error in add_visual_measurement: {e}")
            return False

    def get_full_trajectory(self):
        try:
            result = self.isam.calculateEstimate()
            positions = []
            orientations = []

            for i in range(self.current_state_idx + 1):
                if result.exists(X(i)):
                    
                    # 1. Get the pose from the graph (this is in the World frame)
                    pose_world = result.atPose3(X(i)) 
                    # 2. Transform the entire pose to the world frame

                    # 3. Extract the world-frame position and orientation
                    pos_world = pose_world.translation()
                    quat_world = pose_world.rotation().toQuaternion()

                    positions.append([pos_world[0], pos_world[1], pos_world[2]])
                    orientations.append([quat_world.x(), quat_world.y(), quat_world.z(), quat_world.w()])
            
            return np.array(positions), np.array(orientations)
            
        except Exception as e:
            print(f"Error in get_full_trajectory: {e}")
            return np.array([]), np.array([])

    def debug_graph(self):
        try:
            result = self.isam.calculateEstimate()
            print("\n--- Factor Graph Debug ---")
            print(f"Current state index: {self.current_state_idx}")
            print(f"Active factors: {self.graph.size()}")
            print(f"Variables in ISAM2: {result.size()}")
            
            if self.current_state_idx > 0:
                print("\nLatest IMU Bias:")
                print(f"Accel bias: {self.bias.accelerometer()}")
                print(f"Gyro bias: {self.bias.gyroscope()}")
                
                print("\nLatest Pose:")
                pose = result.atPose3(X(self.current_state_idx))
                print(f"Position: {pose.translation()}")

                 # Get preintegrated deltas
                delta_R = self.imu_preintegrated.deltaRij()
                delta_v = self.imu_preintegrated.deltaVij()
                delta_p = self.imu_preintegrated.deltaPij()
                
                print("\n--- Preintegrated IMU Measurements ---")
                print(f"Delta Rotation (rad):\n{delta_R.matrix()}")
                print(f"Delta Velocity (m/s):\n{delta_v}")
                print(f"Delta Position (m):\n{delta_p}")
                
                # Predict without initial state
                nav_state = self.imu_preintegrated.predict(
                    gtsam.NavState(gtsam.Pose3(), np.zeros(3)), 
                    self.bias
                )
                
                # Calculate gravity-compensated acceleration manually
                dt = self.imu_preintegrated.deltaTij()
                velocity = nav_state.velocity()
                rotation_matrix = nav_state.pose().rotation().matrix()
                
                # The acceleration vector (including gravity) in the body frame
                # To remove gravity effect, we rotate the velocity delta back to body frame
                rotation_inverse = np.linalg.inv(rotation_matrix)
                accel_with_gravity = rotation_inverse @ (velocity / dt)
                
                print("Gravity-compensated accel:", accel_with_gravity)
                
        except Exception as e:
            print(f"Debug error: {e}")