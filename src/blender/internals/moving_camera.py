import bpy
from math import radians, cos, sin, atan2
import numpy as np
from mathutils import Vector, Matrix, Euler

class MovingCamera:
    def __init__(self, *, first_root, mode, 
                 is_mesh, 
                 fakeinone=False,
                 camera_locx=None,
                 camera_locy=None,
                 camera_locz=None,
                 camera_fov=None,
                 smooth_factor=5.0,
                 use_constraints=False):
        
        self.camera = bpy.data.objects['Camera']
        self.mode = mode
        self._root = first_root
        self.pivot_location = first_root
        self.angle_increment = radians(45)
        self.smooth_factor = smooth_factor
        self.use_constraints = use_constraints

        # Set initial position
        self.camera.location.x = camera_locx if camera_locx is not None else 5.45
        self.camera.location.y = camera_locy if camera_locy is not None else -4.95
        
        if camera_locz is None:
            self.camera.location.z = 2.1 if is_mesh else 5.2
        else:
            self.camera.location.z = camera_locz

        # Set FOV based on mode
        if mode == "sequence":
            if camera_fov is None:
                if is_mesh:
                    self.camera.data.lens = 75 if fakeinone else 100
                else:
                    self.camera.data.lens = 85
            else:
                self.camera.data.lens = camera_fov
        elif mode == "frame":
            self.camera.data.lens = 130 if is_mesh else 140
        elif mode == "video":
            if camera_fov is None:
                self.camera.data.lens = 65 if is_mesh else 140
            else:
                self.camera.data.lens = camera_fov

        # Adjust position based on root
        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]
        # self.camera.location.z += first_root[2]

        # Set initial rotation
        self.camera.rotation_euler[0] = radians(81.7)
        self.camera.rotation_euler[2] = radians(47.2)

        # self.use_constraints = False
        # self.lock_to_target(bpy.data.objects['Target'], relative_pos=(0.0, -0.0005, 1.5))

        # import ipdb; ipdb.set_trace()

        if use_constraints:
            self._setup_constraints()

    def _setup_constraints(self):
        """Set up camera constraints for smooth tracking"""
        self.camera.constraints.clear()
        
        # Create damper empty if it doesn't exist
        damper_name = f"Camera_Damper_{self.camera.name}"
        damper = bpy.data.objects.get(damper_name)
        if not damper:
            bpy.ops.object.empty_add(type='PLAIN_AXES')
            damper = bpy.context.active_object
            damper.name = damper_name
        
        # Track To constraint
        track = self.camera.constraints.new('TRACK_TO')
        track.target = damper
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'
        
        # Damper constraints
        damp_loc = damper.constraints.new('COPY_LOCATION')
        damp_loc.target = bpy.data.objects.get('Target')  # Assuming target object exists
        damp_loc.influence = 1.0 / self.smooth_factor
        bpy.data.objects["Camera_Damper_Camera"].location[0] = 0.6
        self.damper = damper

    def lock_to_target(self, target_obj, relative_pos=(0, 0, 1.5)):
        """Lock camera to follow a target object"""
        if self.use_constraints:
            if not hasattr(self, 'damper'):
                self._setup_constraints()
            self.damper.constraints["Copy Location"].target = target_obj
        else:
            target_loc = target_obj.location
            self.camera.location = Vector((
                target_loc.x + relative_pos[0],
                target_loc.y + relative_pos[1],
                target_loc.z + relative_pos[2]
            ))
            self.look_at(target_loc)

    def rotate_camera(self):
        """Rotate camera around pivot point"""
        radius = ((self.camera.location.x - self.pivot_location[0]) ** 2 + 
                 (self.camera.location.y - self.pivot_location[1]) ** 2) ** 0.5
        current_angle = atan2(self.camera.location.y - self.pivot_location[1],
                            self.camera.location.x - self.pivot_location[0])
        new_angle = current_angle + self.angle_increment

        new_x = self.pivot_location[0] + radius * cos(new_angle)
        new_y = self.pivot_location[1] + radius * sin(new_angle)

        self.camera.location.x = new_x
        self.camera.location.y = new_y
        self.look_at(self.pivot_location)

    def look_at(self, target, pitch_offset=-15):
        """Points the camera's Z-axis towards the target location"""
        if not self.use_constraints:
            loc_camera = self.camera.location
            direction = Vector(target) - loc_camera
            direction_norm = direction.normalized()
            up_direction = Vector((0, 0, 1))
            
            right = direction_norm.cross(up_direction).normalized()
            up = right.cross(direction_norm).normalized()
            
            rot_mat = Matrix((right, up, direction_norm)).transposed()
            new_euler = rot_mat.to_euler()
            new_euler.x += radians(pitch_offset)
            
            self.camera.rotation_euler = new_euler

    def update(self, newroot):
        """Update camera position based on new root position"""
        delta_root = newroot - self._root

        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]
        if delta_root[2] + self.camera.location.z > 2.1:
            self.camera.location.z += delta_root[2]

        self._root = newroot
        
        if hasattr(self, 'damper'):
            self.damper.location = newroot

    def update_from_dict(self, camera_dict):
        """Update camera based on camera dictionary settings"""
        if camera_dict:
            # Get rotation and apply it
            R_z = self._get_z_rot(np.array(camera_dict['camera_rot']))
            heading = -R_z[:, 1]  # Extract heading direction from rotation matrix
            
            # Set camera position directly from translation
            camera_pos = np.array(camera_dict['camera_transl'])
            self.camera.location = Vector((
                camera_pos[0],
                camera_pos[1],
                self.camera.location.z  # Maintain current z height
            ))


    def update_heading_target(self, camera_dict, z_offset=1.5):
        """
        Update the location of the 'Target' empty based on heading logic,
        so the camera smoothly follows the front of the character.
        """
        # 1) Get the 'Target' empty
        target_obj = bpy.data.objects.get("Target")
        if not target_obj:
            print("No object named 'Target' found in the scene!")
            return

        # 2) Compute the rotation matrix from angle-axis
        #    camera_dict['camera_rot'] is presumably a list or array of shape (3,)
        angle_axis = np.array(camera_dict['rot'][0])  # or [0] if needed
        R_z = self._get_z_rot(angle_axis)
        
        # 3) Extract heading from the second column (negative sign)
        heading = R_z[:, 1]  # typical AITViewer logic
        
        # 4) Compute the "facing" position in XY
        transl = np.array(camera_dict['trans'][0])  # or [0] if needed
        coef = 1.0  # or some default
        xy_facing = transl + heading * coef
        
        # 5) Decide if we 'lock2object' or not
        if 'lock2object' in camera_dict:
            # If locking to a mesh, place camera closer (lower Z offset)
            z_offset = 0.3

        # 6) Update the Target empty’s location
        target_obj.location = (
            xy_facing[0],
            xy_facing[1],
            z_offset
        )

    def _get_z_rot(self, angle_axis):
        """Convert angle-axis rotation to rotation matrix"""
        angle = np.linalg.norm(angle_axis)
        if angle < 1e-8:
            return np.eye(3)
        axis = angle_axis / angle
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.matmul(K, K)
        return R