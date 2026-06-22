import bpy
import numpy as np
import matplotlib
COLORMAPS = [matplotlib.cm.get_cmap('Blues'), matplotlib.cm.get_cmap('Greys'), 
             matplotlib.cm.get_cmap('Purples'), 
             matplotlib.cm.get_cmap('Greens'),
             matplotlib.cm.get_cmap('Reds'),
             matplotlib.cm.get_cmap('Oranges'),
             matplotlib.cm.get_cmap('YlGn')]

class Goals:
    def __init__(self, goal_tensor, *, mode, radius=0.015, 
                 num_of_frames_to_rend=8,
                 fixed_color='yellow',
                 mat=None
                 ):
        self.goal_tensor = goal_tensor
        self.n_frames = goal_tensor.shape[0]
        self.mode = mode
        self.radius = radius
         
        self.fixed_color = fixed_color

        if mat is None:
            self.mat = (0.005, 0.034, 0.089, 1)
            self._set_material_color()
        else:
            self.mat = mat

        self.active_spheres = {}
        self.last_idx = None
        self.num_of_frames_to_rend = num_of_frames_to_rend
        
        
    def _set_material_color(self):
        if self.mode == 'sequence' or self.mode == 'video':
            if self.fixed_color == 'red':
                self.mat = (0.08, 0.008, 0.007, 1)
            elif self.fixed_color == 'green':
                self.mat = (0.012, 0.07, 0.013, 1)
            elif self.fixed_color == 'blue':
                self.mat = (0.003, 0.06, 0.659, 1)
            elif self.fixed_color in ['grey', 'gray']:
                self.mat = (0.204, 0.157, 0.137, 1)
            elif self.fixed_color == 'purple':
                self.mat = (0.071, 0.038, 0.296, 1)
            elif self.fixed_color == 'yellow':
                self.mat = (0.9, 0.9, 0.006, 1)
            else: 
                self.mat = (0.005, 0.034, 0.089, 1)
        elif self.mode == 'frame':
            self.mat = matplotlib.cm.get_cmap('Dark2')(0)
    
    def _create_material(self, name, color):
        """Create a new material that works in both viewport and render"""
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Create Principled BSDF shader
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        # Set base color
        principled.inputs['Base Color'].default_value = color
        principled.inputs['Emission'].default_value = color
        principled.inputs['Metallic'].default_value = 0.0
        principled.inputs['Roughness'].default_value = 0.2
        principled.inputs['Specular'].default_value = 0.5
        
        # Create output node
        output = nodes.new('ShaderNodeOutputMaterial')
        
        # Link nodes
        links.new(principled.outputs[0], output.inputs[0])
        
        # Set viewport display mode
        mat.blend_method = 'OPAQUE'
        mat.shadow_method = 'OPAQUE'
        
        # Make sure color is visible in viewport
        mat.use_nodes = True
        mat.diffuse_color = color
        return mat

    def get_sequence_mat(self, frac=None):


        if self.mode == 'sequence' and frac is not None:
            
            cmap = matplotlib.cm.get_cmap('YlGn')

            begin = 0.60
            end = 0.90
            rgbcolor = list(cmap(begin + (end-begin)*frac))

            if self.fixed_color == 'red':
                rgbcolor[1] *= 0.2
                rgbcolor[2] *= 0.2
                rgbcolor[0] *= 0.45

            elif self.fixed_color == 'green':
                rgbcolor[0] *= 0.2
                rgbcolor[2] *= 0.2
                rgbcolor[1] *= 0.45

            elif self.fixed_color == 'blue':
                rgbcolor[0] *= 0.2
                rgbcolor[1] *= 0.2
                rgbcolor[2] *= 0.45

            elif self.fixed_color == 'purple':
                rgbcolor[0] *= 0.2
                rgbcolor[1] *= 0.2
                rgbcolor[2] *= 0.45

            elif self.fixed_color in ['grey', 'gray']:
                rgbcolor[0] *= 0.2
                rgbcolor[1] *= 0.2
                rgbcolor[2] *= 0.2

            elif self.fixed_color == 'yellow':
                rgbcolor[0] *= 0.2
                rgbcolor[1] *= 0.2
                rgbcolor[2] *= 0.2

            return tuple(rgbcolor)
        else:
            return self.mat
    
    def _create_sphere(self, location, name, color):
        # Create sphere (using ico sphere as example, but works with any method)
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=self.radius,
            location=location
        )
        # import ipdb;ipdb.set_trace()

        sphere = bpy.context.active_object
        sphere.name = name
        sphere.data.name = name
        
        # Create material similar to your plane setup
        material = bpy.data.materials.new(name=f"Material_{name}")
        material.use_nodes = True
        
        # Get the principled BSDF node
        bsdf = material.node_tree.nodes["Principled BSDF"]
        
        # Set the base color directly
        bsdf.inputs['Base Color'].default_value = color
        
        # Apply material to sphere
        sphere.active_material = material
        
        # Smooth shading
        bpy.ops.object.shade_smooth(use_auto_smooth=True)
        sphere.is_shadow_catcher = False
 
        return sphere

    # def _create_sphere(self, location, name, color):
    #     # Create sphere
    #     bpy.ops.mesh.primitive_ico_sphere_add(
    #         subdivisions=3,
    #         radius=self.radius,
    #         location=location
    #     )
    #     sphere = bpy.context.active_object
    #     sphere.name = name
    #     sphere.data.name = name

    #     # Create a new material with nodes
    #     material = bpy.data.materials.new(name=f"Material_{name}")
    #     material.use_nodes = True
    #     nodes = material.node_tree.nodes
    #     links = material.node_tree.links

    #     # Clear all default nodes
    #     nodes.clear()

    #     # Create a Principled BSDF node
    #     principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    #     principled.location = (0, 0)
    #     principled.inputs['Base Color'].default_value = color
    #     principled.inputs['Roughness'].default_value = 0.2

    #     # Create an Emission node for extra brightness
    #     emission = nodes.new(type='ShaderNodeEmission')
    #     emission.location = (0, -200)
    #     emission.inputs['Color'].default_value = color
    #     emission.inputs['Strength'].default_value = 5.0  # Adjust strength as needed

    #     # Create a Mix Shader node to combine BSDF and Emission
    #     mix_shader = nodes.new(type='ShaderNodeMixShader')
    #     mix_shader.location = (200, 0)
    #     mix_shader.inputs['Fac'].default_value = 0.5  # Adjust the mix factor if desired

    #     # Create Material Output node
    #     output = nodes.new(type='ShaderNodeOutputMaterial')
    #     output.location = (400, 0)

    #     # Link the nodes:
    #     # Principled BSDF --> Mix Shader (input 1)
    #     links.new(principled.outputs[0], mix_shader.inputs[1])
    #     # Emission --> Mix Shader (input 2)
    #     links.new(emission.outputs[0], mix_shader.inputs[2])
    #     # Mix Shader --> Material Output
    #     links.new(mix_shader.outputs[0], output.inputs[0])

    #     # Assign material to sphere
    #     sphere.active_material = material

    #     # Apply smooth shading
    #     bpy.ops.object.shade_smooth(use_auto_smooth=True)
    #     sphere.is_shadow_catcher = False

    #     return sphere
    
    def load_in_blender(self, index, use_mat=False):
        # if self.last_idx is not None and self.last_idx != index:
        #     self.clean_up()

        frame_goals = self.goal_tensor[index]
        active_spheres_this_frame = {}

        # Get color for this frame
        if self.mode == 'sequence':
            frac = index / (self.num_of_frames_to_rend-1) if self.num_of_frames_to_rend > 1 else 0.5
            color = self.get_sequence_mat(frac)
        else:
            color = self.get_sequence_mat()
        
        # Ensure color is 4-component (RGBA)
        if len(color) == 3:
            color = (*color, 1.0)

        for i, goal_pos in enumerate(frame_goals):
            if np.allclose(goal_pos, 0):
                continue
            name = f"Goal_{index}_{i}"
            
            if use_mat:
                sphere = self._create_sphere(goal_pos, name, self.mat[i])
            else:
                sphere = self._create_sphere(goal_pos, name, color)

            active_spheres_this_frame[name] = sphere
        
        self.active_spheres = active_spheres_this_frame
        self.last_idx = index
        return list(active_spheres_this_frame.keys())

    def clean_up(self):
        for name, sphere in self.active_spheres.items():
            # Remove material
            if sphere.active_material:
                bpy.data.materials.remove(sphere.active_material)
            # Remove object
            bpy.data.objects.remove(sphere, do_unlink=True)
        self.active_spheres = {}
    
    def __len__(self):
        return self.n_frames