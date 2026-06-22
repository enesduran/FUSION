import bpy


def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


def colored_material(r, g, b, a=1, roughness=0.127451):
    materials = bpy.data.materials
    material = materials.new(name="body")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs["Color"].default_value = (r, g, b, a)
    diffuse.inputs["Roughness"].default_value = roughness
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    return material


def image_plane_mat(texture_path):
    materials = bpy.data.materials
    material = materials.new(name="plane")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')

    # Add the Image Texture node
    node_tex = nodes.new('ShaderNodeTexImage')   
    # Assign the image
    node_tex.image = bpy.data.images.load(texture_path)

    links.new(node_tex.outputs["Color"], diffuse.inputs['Color'])
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    
    diffuse.inputs["Roughness"].default_value = 0.127451
    return material

import bpy

def clear_material_cycles(material):
    nodes = material.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

def plane_mat_checker():
    materials = bpy.data.materials
    material = materials.new(name="plane")
    material.use_nodes = True

    # Clear existing nodes
    clear_material_cycles(material)
    
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Create new nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')  # Use Principled BSDF for Cycles
    checker = nodes.new(type="ShaderNodeTexChecker")
    text_coord = nodes.new(type="ShaderNodeTexCoord")
    coord_mapping = nodes.new(type="ShaderNodeMapping")

    # Connect texture coordinates to mapping node
    links.new(text_coord.outputs["Generated"], coord_mapping.inputs['Vector'])

    # Connect mapping node to checker texture
    links.new(coord_mapping.outputs['Vector'], checker.inputs['Vector'])

    # Set up mapping node to adjust the scale
    coord_mapping.inputs[3].default_value[0] = 210  # Translation X
    coord_mapping.inputs[3].default_value[1] = 210  # Translation Y
    coord_mapping.inputs[3].default_value[2] = 0    # Translation Z

    # Configure the checker texture
    checker.inputs["Scale"].default_value = 5
    checker.inputs["Color1"].default_value = (0.025, 0.025, 0.025, 1)  # Dark grey
    checker.inputs["Color2"].default_value = (0.05, 0.05, 0.05, 1)     # Light grey

    # Connect the checker texture to the Principled BSDF shader
    links.new(checker.outputs["Color"], principled.inputs['Base Color'])

    # Connect the Principled BSDF shader to the material output
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # Set additional properties for the Principled BSDF shader
    principled.inputs["Roughness"].default_value = 0.127451  # Match roughness to the original value

    return material

def normal_plane_mat_checker():
    materials = bpy.data.materials
    material = materials.new(name="plane")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    checker = nodes.new(type="ShaderNodeTexChecker")

    text_coord = nodes.new(type="ShaderNodeTexCoord")
    coord_mapping = nodes.new(type="ShaderNodeMapping")

    links.new(text_coord.outputs["Generated"], coord_mapping.inputs['Vector'])
    links.new(coord_mapping.outputs['Vector'], checker.inputs['Vector'])
    bpy.data.materials["plane"].node_tree.nodes["Mapping"].inputs[3].default_value[0] = 210
    bpy.data.materials["plane"].node_tree.nodes["Mapping"].inputs[3].default_value[1] = 210
    bpy.data.materials["plane"].node_tree.nodes["Mapping"].inputs[3].default_value[2] = 0

    checker.inputs["Scale"].default_value = 5
    checker.inputs["Color1"].default_value = (0.025, 0.025, 0.025, 1)
    checker.inputs["Color2"].default_value = (0.05, 0.05, 0.05, 1)
    links.new(checker.outputs["Color"], diffuse.inputs['Color'])
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    diffuse.inputs["Roughness"].default_value = 0.127451
    return material



def add_floor(size): # alternative checkerboard
    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = 'floor'

    floor_mat = bpy.data.materials.new(name="floorMaterial")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
    floor_text = floor_mat.node_tree.nodes.new("ShaderNodeTexChecker")
    floor_text.inputs[3].default_value = 150
    floor_mat.node_tree.links.new(bsdf.inputs['Base Color'], floor_text.outputs['Color'])

    floor.data.materials.append(floor_mat)
    return floor



def plane_mat_uni():
    materials = bpy.data.materials
    material = materials.new(name="plane_uni")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
    diffuse.inputs["Roughness"].default_value = 0.127451
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    return 
