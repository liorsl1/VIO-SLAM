import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_extrinsics_difference(T1, T2):
    """
    Compute the difference between two 4x4 extrinsic matrices.
    
    Args:
        T1: First 4x4 transformation matrix
        T2: Second 4x4 transformation matrix
        
    Returns:
        T_diff: 4x4 transformation matrix representing T1 to T2
        rotation_diff_deg: Rotation difference in degrees
        translation_diff: Translation difference vector
        translation_distance: Euclidean distance between translations
    """
    # Compute relative transformation: T2 = T_diff @ T1
    # Therefore: T_diff = T2 @ T1^(-1)
    T_diff = T2 @ np.linalg.inv(T1)
    
    # Extract rotation and translation components
    R_diff = T_diff[:3, :3]
    t_diff = T_diff[:3, 3]
    
    # Compute rotation difference in degrees
    rotation_diff_deg = rotation_matrix_to_degrees(R_diff)
    
    # Compute translation difference
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    translation_diff = t2 - t1
    translation_distance = np.linalg.norm(translation_diff)
    
    return T_diff, rotation_diff_deg, translation_diff, translation_distance

def rotation_matrix_to_degrees(R_matrix):
    """
    Convert rotation matrix to rotation angle in degrees.
    
    Args:
        R_matrix: 3x3 rotation matrix
        
    Returns:
        angle_deg: Rotation angle in degrees
    """
    # Use scipy to convert to axis-angle representation
    r = R.from_matrix(R_matrix)
    rotvec = r.as_rotvec()
    angle_rad = np.linalg.norm(rotvec)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def compare_extrinsics_detailed(T1, T2, labels=None):
    """
    Detailed comparison of two extrinsics with comprehensive output.
    
    Args:
        T1: First 4x4 transformation matrix
        T2: Second 4x4 transformation matrix
        labels: Optional tuple of labels for the two matrices
        
    Returns:
        Dictionary with detailed comparison results
    """
    if labels is None:
        labels = ("T1", "T2")
    
    # Compute differences
    T_diff, rot_diff_deg, trans_diff, trans_distance = compute_extrinsics_difference(T1, T2)
    
    # Extract individual components
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    
    # Convert rotations to Euler angles for easier interpretation
    euler1 = R.from_matrix(R1).as_euler('xyz', degrees=True)
    euler2 = R.from_matrix(R2).as_euler('xyz', degrees=True)
    euler_diff = euler2 - euler1
    
    # Prepare results
    results = {
        'relative_transformation': T_diff,
        'rotation_difference_deg': rot_diff_deg,
        'translation_difference': trans_diff,
        'translation_distance': trans_distance,
        'euler_angles': {
            labels[0]: euler1,
            labels[1]: euler2,
            'difference': euler_diff
        },
        'translations': {
            labels[0]: t1,
            labels[1]: t2,
            'difference': trans_diff
        }
    }
    
    return results

def print_extrinsics_comparison(T1, T2, labels=None):
    """
    Print a formatted comparison of two extrinsics matrices.
    """
    results = compare_extrinsics_detailed(T1, T2, labels)
    
    if labels is None:
        labels = ("T1", "T2")
    
    print(f"=== Extrinsics Comparison: {labels[0]} vs {labels[1]} ===")
    print(f"Rotation difference: {results['rotation_difference_deg']:.3f} degrees")
    print(f"Translation distance: {results['translation_distance']:.6f} meters")
    print(f"Translation difference: [{results['translation_difference'][0]:.6f}, "
          f"{results['translation_difference'][1]:.6f}, "
          f"{results['translation_difference'][2]:.6f}]")
    
    print(f"\nEuler angles (XYZ, degrees):")
    print(f"  {labels[0]}: [{results['euler_angles'][labels[0]][0]:.2f}, "
          f"{results['euler_angles'][labels[0]][1]:.2f}, "
          f"{results['euler_angles'][labels[0]][2]:.2f}]")
    print(f"  {labels[1]}: [{results['euler_angles'][labels[1]][0]:.2f}, "
          f"{results['euler_angles'][labels[1]][1]:.2f}, "
          f"{results['euler_angles'][labels[1]][2]:.2f}]")
    print(f"  Difference: [{results['euler_angles']['difference'][0]:.2f}, "
          f"{results['euler_angles']['difference'][1]:.2f}, "
          f"{results['euler_angles']['difference'][2]:.2f}]")

# Example usage
if __name__ == "__main__":
    # Example extrinsics matrices
    T1 = np.array([0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0]).reshape(4,4)
    
    T2 = np.array(
         [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
         0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
        -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
         0.0, 0.0, 0.0, 1.0]
    ).reshape(4,4)

    # Print detailed comparison
    print_extrinsics_comparison(T1, T2, ("Camera1", "Camera2"))
    
    # Get detailed results
    results = compare_extrinsics_detailed(T1, T2)
    print(f"\nRelative transformation matrix:")
    print(results['relative_transformation'])

        # Example: Point at origin in body frame
    p_body = np.array([0, 0, 0, 1])

    # Transform to camera frame
    p_cam_left = T1 @ p_body
    p_cam_right = T2 @ p_body

    print(f"Origin in left camera frame: {p_cam_left[:3]}")
    print(f"Origin in right camera frame: {p_cam_right[:3]}")