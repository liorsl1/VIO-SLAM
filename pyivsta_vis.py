import pyvista as pv
import numpy as np
import threading
import queue
import time

class RealTimeSLAMPyVistaVisualizer:
    def __init__(self):
        self.update_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._run_visualization)
        self.thread.start()

    def _run_visualization(self):
        # Create PyVista plotter
        self.plotter = pv.Plotter(window_size=[1024, 768])
        self.plotter.add_axes()
        self.plotter.set_background("white")
        self.point_cloud = None
        self.traj_line = None
        self.ekf_line = None
        self.fg_line = None
        self.gt_line = None
        self.voxel_grid = None
        self.map_points_3d = np.zeros((0, 3))  # Accumulated map points

        # Start interactive plotting in a non-blocking way
        self.plotter.show(auto_close=False, interactive_update=True)

        while self.running:
            updated = False
            try:
                points_3d, traj, ekf_traj, fg_traj, gt_traj = self.update_queue.get_nowait()
                updated = True
            except queue.Empty:
                pass

            if updated:
                # Remove previous actors
                # if self.point_cloud is not None:
                #     self.plotter.remove_actor(self.point_cloud)
                if self.traj_line is not None:
                    self.plotter.remove_actor(self.traj_line)
                if self.ekf_line is not None:
                    self.plotter.remove_actor(self.ekf_line)
                if self.gt_line is not None:
                    self.plotter.remove_actor(self.gt_line)
                if self.point_cloud is not None:
                    self.plotter.remove_actor(self.point_cloud)
                if self.fg_line is not None:
                    self.plotter.remove_actor(self.fg_line)

                # Add point cloud
                # if points_3d is not None and len(points_3d):
                #     self.point_cloud = self.plotter.add_points(
                #         np.asarray(points_3d), color="blue", point_size=5, render_points_as_spheres=True
                #     )
                if points_3d is not None and len(points_3d):
                    # Accumulate points for the map
                    self.map_points_3d = np.vstack([self.map_points_3d, np.asarray(points_3d)])
                # add scalars based heatmap
                if self.map_points_3d.shape[0] > 0:
                    scalars = self.map_points_3d[:, 2]  # Use Y values for heatmap
                    self.point_cloud = self.plotter.add_points(
                        self.map_points_3d, scalars=scalars, point_size=5, render_points_as_spheres=True,
                        cmap="hot"
                    )

                # Add visual trajectory
                if traj is not None and len(traj) > 1:
                    polyline = pv.lines_from_points(np.asarray(traj))
                    self.traj_line = self.plotter.add_mesh(polyline, color="purple", line_width=3)
                    # Add points at each position
                    self.traj_points = self.plotter.add_points(
                        np.asarray(traj), color="purple", point_size=10, render_points_as_spheres=True
                    )

                # Add EKF trajectory
                if ekf_traj is not None and len(ekf_traj) > 1:
                    polyline = pv.lines_from_points(np.asarray(ekf_traj))
                    self.ekf_line = self.plotter.add_mesh(polyline, color="yellow", line_width=3)
                    self.ekf_points = self.plotter.add_points(
                        np.asarray(ekf_traj), color="yellow", point_size=10, render_points_as_spheres=True
                    )

                # Add ground truth trajectory
                if gt_traj is not None and len(gt_traj) > 1:
                    polyline = pv.lines_from_points(np.asarray(gt_traj))
                    self.gt_line = self.plotter.add_mesh(polyline, color="green", line_width=3)
                    self.gt_points = self.plotter.add_points(
                        np.asarray(gt_traj), color="green", point_size=10, render_points_as_spheres=True
    )
                # Add factor graph trajectory
                if fg_traj is not None and len(fg_traj) > 1:
                    polyline = pv.lines_from_points(np.asarray(fg_traj))
                    self.fg_line = self.plotter.add_mesh(polyline, color="black", line_width=3)
                    self.fg_points = self.plotter.add_points(
                        np.asarray(fg_traj), color="black", point_size=10, render_points_as_spheres=True
                    )

                # Fit camera to all data
                #self.plotter.reset_camera()

            self.plotter.update()
            time.sleep(0.01)

        self.plotter.close()

    def enqueue_update(self, points_3d, traj=None, ekf_traj=None, fg_traj=None, gt_traj=None):
        self.update_queue.put((points_3d, traj, ekf_traj, fg_traj, gt_traj))

    def close(self):
        self.running = False
        self.thread.join()