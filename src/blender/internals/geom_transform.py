import numpy as np


def rotate_vertices_around_center(vertices, angles_degrees):
    """
    Rotate vertices around their center point
    vertices: numpy array of shape (..., 3)
    angles_degrees: (x, y, z) angles in degrees to rotate
    """
    # Calculate center of the mesh
    original_shape = vertices.shape
    vertices_reshaped = vertices.reshape(-1, 3)
    center = np.mean(vertices_reshaped, axis=0)
    
    # Center the vertices
    centered_vertices = vertices_reshaped - center
    
    # Convert angles to radians
    angles = np.radians(angles_degrees)
    
    # X rotation matrix
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])
    
    # Y rotation matrix
    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    
    # Z rotation matrix
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    
    # Apply rotation to centered vertices
    rotated_centered = np.dot(centered_vertices, R.T)
    
    # Move back to original center
    rotated_vertices = rotated_centered + center
    
    return rotated_vertices.reshape(original_shape) 

def rotate_for_side_view(vertices, front_facing_angles, side='right'):
    """
    Rotate vertices for side view based on known front-facing angles
    side: 'right' or 'left'
    """
    # First, apply the front-facing rotation
    front_rotated = rotate_vertices(vertices.copy(), front_facing_angles)
    
    # Then apply a pure Y-axis rotation of ±90° from that position
    angle_y = np.radians(90 if side == 'left' else -90)
    rot_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    # Apply Y rotation
    original_shape = front_rotated.shape
    reshaped = front_rotated.reshape(-1, 3)
    side_view = np.dot(reshaped, rot_y.T)
    
    return side_view.reshape(original_shape)


def rotate_vertices(vertices, angles_degrees):
    """
    Rotate vertices around multiple axes
    vertices: numpy array of shape (..., 3)
    angles_degrees: (x, y, z) angles in degrees to rotate
    """
    angles = np.radians(angles_degrees)
    
    # X rotation matrix
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])
    
    # Y rotation matrix
    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    
    # Z rotation matrix
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    
    # Rotate the vertices
    original_shape = vertices.shape
    vertices_reshaped = vertices.reshape(-1, 3)
    rotated_vertices = np.dot(vertices_reshaped, R.T)
    
    return rotated_vertices.reshape(original_shape)

