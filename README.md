# VIO-SLAM
Loosely-coupled Visual Inertial Odometry SLAM implementation, with factor graph (GTSAM) and custom EKF. Also including 3d projection from stereo disparity, creating 3D pointcloud in global frame, based on global poses extracted from the fusion.

Alot of implemented theory in SLAM and Multi-View geometry:
Stereo rectification, Triangulation, epipolar geometry constraints, pinhole camera model, projections, reprojection errors, statistical optimization (EKF and Factor Graphs), Imu preintegration and visual-inertial fusion.
Things to add for improvement :
Loop closure, change to tightly-coupled approach
### A work in progress.

### Visualization
![VIO-SLAM Trajectory Comparison](basic_plot.png)
*Comparison of VIO-SLAM different trajectories, with PCL of frames 3d reconstruction.*
