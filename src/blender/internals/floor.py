import bpy
from .materials import colored_material as get_mat, image_plane_mat
from .materials import normal_plane_mat_checker

import math


def get_trajectory(data, is_mesh):
    if is_mesh:
        # mean of the vertices
        trajectory = data[:, :, [0, 1]].mean(1)
    else:
        # get the root joint
        trajectory = data[:, 0, [0, 1]]
    return trajectory

def plot_floor(data, color_alpha=None, texture_path=None, rotation=None,
               scale_of_tiles=None, color1=None, color2=None):
    # Create a floor
    if isinstance(data, list):
        minx, miny, minz = 0.0 ,0.0,0.0
        maxx, maxy = 0.0 ,0.0
    else:
        minx, miny, minz = data.min(axis=(0, 1))
        maxx, maxy, _ = data.max(axis=(0, 1))
    location = ((maxx + minx) / 2, (maxy + miny) / 2, minz-0.01)
    scale = ((maxx - minx) / 2, (maxy - miny) / 2, 1)

    # Create a plane
    bpy.ops.mesh.primitive_plane_add(size=18, enter_editmode=False, align='WORLD', 
                                     location=location, 
                                     scale=(1, 1, 1))

    #bpy.ops.transform.resize(value=[2 * x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    #                         constraint_axis=(False, True, False))

    obj = bpy.data.objects["Plane"]
    obj.name = "BigPlane"
    obj.data.name = "BigPlane"

    # Create a checkerboard material
    material = bpy.data.materials.new(name="CheckerboardMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    checker = material.node_tree.nodes.new('ShaderNodeTexChecker')
    if scale_of_tiles is not None:
        checker.inputs['Scale'].default_value = scale_of_tiles
    else:
        checker.inputs['Scale'].default_value = 22.0  # Adjust scale for smaller checkerboard tiles
    if color1 is not None:
        checker.inputs['Color1'].default_value = color1  # Very dark gray, almost black
    else:
        checker.inputs['Color1'].default_value = (0.0, 0.0, 0.0, 1)  # Very dark gray, almost black

    checker.inputs['Color2'].default_value = (1, 1, 1, 1)  # Very dark gray, almost black

    # Link the checker texture to the base color of the material
    material.node_tree.links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
    if rotation is not None:
        bpy.data.objects["BigPlane"].rotation_euler[2] = math.radians(rotation)
    obj.active_material = material

def plot_floor_wo_checker(data, color_alpha=None, texture_path=None):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    scale = ((maxx - minx)/2, (maxy - miny)/2, 1)

    if False:
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))
        
        bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)
        obj = bpy.data.objects["Plane"]
        obj.name = "SmallPlane"
        obj.data.name = "SmallPlane"
        if color_alpha is not None:
            # obj.active_material = get_mat(0.1, 0.1, 0.1, 1*color_alpha)
            obj.active_material = get_mat(65/255, 105/255, 225/255, 1)
        else:
            obj.active_material = get_mat(0.1, 0.1, 0.1, 1)
            
    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))
    ground_plane = bpy.context.selected_objects[0]
    ground_plane.is_shadow_catcher = True
    floor_size = [200, 200, 1]
    #bpy.ops.transform.resize(value=floor_size, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
     #                        constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
     #                        proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
     #                        use_proportional_projected=False, release_confirm=True)
    bpy.ops.transform.resize(value=floor_size, constraint_axis=(True, True, False))

    obj = bpy.data.objects["Plane"]
    obj.name = "BigPlane"
    obj.data.name = "BigPlane"
    bpy.context.active_object.rotation_euler[2] = math.radians(45)
    if texture_path=='checkerboard':
        obj.active_material = normal_plane_mat_checker()
    elif texture_path is not None:
        obj.active_material = image_plane_mat(texture_path)
    else:
        if color_alpha is not None:
            obj.active_material = get_mat(70/255, 130/255, 180/255, 1)
        else:
            logr = [(0.142, 0.15, 0.152),
                    (0.406, 0.432, 0.439),]
            clr = logr[1]
            obj.active_material = get_mat(clr[0], clr[1], clr[2], 1)


def show_traj(coords):
    # create the Curve Datablock
    curveData = bpy.data.curves.new('myCurve', type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 2

    # map coords to spline
    polyline = curveData.splines.new('POLY')
    polyline.points.add(len(coords)-1)
    for i, coord in enumerate(coords):
        x, y = coord
        polyline.points[i].co = (x, y, 0.001, 1)

    # create Object
    curveOB = bpy.data.objects.new('myCurve', curveData)
    curveData.bevel_depth = 0.01

    bpy.context.collection.objects.link(curveOB)
