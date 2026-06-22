import bpy
import mathutils
from math import radians

def setup_cycles(cycle=True):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.data.scenes[0].render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    # bpy.context.scene.render.film_transparent = True

    if cycle:
        bpy.context.scene.cycles.use_denoising = True
    # added for python versions >3, I have 3.1.2(BLENDER)
    if bpy.app.version[0] == 3:
        if bpy.context.scene.cycles.device == "GPU":
            bpy.context.scene.cycles.tile_size = 256
        else:
            bpy.context.scene.cycles.tile_size = 32
    else:
        bpy.context.scene.render.tile_x = 256
        bpy.context.scene.render.tile_y = 256

    bpy.context.scene.cycles.samples = 64

    # bpy.context.scene.render.film_transparent = True

    # bpy.context.scene.cycles.denoiser = 'OPTIX'

def setup_eevee():

    for scene in bpy.data.scenes:
        scene.render.engine = 'BLENDER_EEVEE'
        # bpy.context.preferences.addons["eevee"].preferences.compute_device_type = "CUDA"
        # bpy.context.scene.eevee.device = "GPU"
        
        # Enable Eevee features
        scene = bpy.context.scene
        eevee = scene.eevee

        eevee.use_soft_shadows = True

        eevee.use_ssr = True
        eevee.use_ssr_refraction = True

        eevee.use_gtao = True
        eevee.gtao_distance = 1

        eevee.use_volumetric_shadows = True
        eevee.volumetric_tile_size = '2'

        for mat in bpy.data.materials:
            # This needs to be enabled case by case,
            # otherwise we loose SSR and GTAO everywhere.
            # mat.use_screen_refraction = True
            mat.use_sss_translucency = True

        cubemap = None
        grid = None
        # Does not work in edit mode
        invert = False
        try:
            # Simple probe setup
            # bpy.ops.object.lightprobe_add(type='CUBEMAP', location=(0.5, 0, 1.5))
            # cubemap = bpy.context.selected_objects[0]
            # cubemap.scale = (2.5, 2.5, 1.0)
            # cubemap.data.falloff = 0
            # cubemap.data.clip_start = 2.4

            # bpy.ops.object.lightprobe_add(type='GRID', location=(0, 0, 0.25))
            # grid = bpy.context.selected_objects[0]
            # grid.scale = (1.735, 1.735, 1.735)
            # grid.data.grid_resolution_x = 3
            # grid.data.grid_resolution_y = 3
            # grid.data.grid_resolution_z = 2
            pass
        except:
            pass

        try:
            # Try to only include the plane in reflections
            plane = bpy.data.objects['Plane']

            collection = bpy.data.collections.new("Reflection")
            collection.objects.link(plane)
            # Add all lights to light the plane
            if not invert:
                for light in bpy.data.objects:
                    if light.type == 'LIGHT':
                        collection.objects.link(light)

            # Add collection to the scene
            scene.collection.children.link(collection)

            cubemap.data.visibility_collection = collection
        except:
            pass

        eevee.gi_diffuse_bounces = 1
        eevee.gi_cubemap_resolution = '128'
        eevee.gi_visibility_resolution = '16'
        eevee.gi_irradiance_smoothing = 0

        bpy.ops.scene.light_cache_bake()




# Setup scene
def setup_scene(render_eng='CYCLES', res='low',
                sun_color=None, sun_strength=6.5, 
                sun_rotation=None,
                sun_location=None):
    scene = bpy.data.scenes['Scene']
    assert res in ["ultra", "high", "med", "low"]
    if res == "high":
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 1024
    elif res == "med":
        scene.render.resolution_x = 1280//2
        scene.render.resolution_y = 1024//2
    elif res == "low":
        scene.render.resolution_x = 1280//4
        scene.render.resolution_y = 1024//4
    elif res == "ultra":
        scene.render.resolution_x = 1920 # 1280*2
        scene.render.resolution_y = 1080 #1024*2
    # render_eng='CYCLES'

    if render_eng=='CYCLES':
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.data.scenes[0].render.engine = "CYCLES"
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    # bpy.context.scene.render.film_transparent = True

        
        bpy.context.scene.cycles.use_denoising = True
        # added for python versions >3, I have 3.1.2(BLENDER)
        if bpy.app.version[0] == 3:
            if bpy.context.scene.cycles.device == "GPU":
                bpy.context.scene.cycles.tile_size = 256
            else:
                bpy.context.scene.cycles.tile_size = 32
        else:
            bpy.context.scene.render.tile_x = 256
            bpy.context.scene.render.tile_y = 256

        bpy.context.scene.cycles.samples = 64

    world = bpy.data.worlds['World']
    world.use_nodes = True

    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()
    if 'Light' in bpy.data.objects:
        bpy.data.objects['Light'].select_set(True)
        bpy.ops.object.delete()
    image_based_lighting = False
    sky=False
    if image_based_lighting:
        paths_tohdris = ['aristea_wreck_puresky_8k.hdr', 
                         'drackenstein_quarry_puresky_4k.hdr', 
                         'kloofendal_48d_partly_cloudy_puresky_4k.hdr',
                         ]
        
        path_to_hdr = 'skys/'+ paths_tohdris[2]
        # 'skys/skidpan_4k.hdr'
        # skys/kloofendal_48d_partly_cloudy_puresky_4k.hdr

        hdr_image_path = bpy.path.abspath(path_to_hdr) # Source: https://polyhaven.com/a/lilienstein
        ################################################################################
        # Tutorial #2: Use image-based lighting
        hdr_image = bpy.data.images.load(hdr_image_path)
        world = bpy.data.worlds[0]
        nodes = world.node_tree.nodes

        # Create new nodes in World shader
        # We use Mapping node to have the option to rotate the background image
        texture_coordinate_node = nodes.new(type="ShaderNodeTexCoord")
        mapping_node = nodes.new(type="ShaderNodeMapping")

        environment_texture_node = nodes.new(type="ShaderNodeTexEnvironment")
        environment_texture_node.image = hdr_image

        # Connect shader nodes
        links = world.node_tree.links
        links.new(texture_coordinate_node.outputs["Generated"], mapping_node.inputs["Vector"])
        links.new(mapping_node.outputs["Vector"], environment_texture_node.inputs["Vector"])
        links.new(environment_texture_node.outputs["Color"], nodes["Background"].inputs["Color"])

        bpy.ops.object.light_add(type='SUN', align='WORLD',
                                location=(-1.22, -2.5, 3.5), 
                                scale=(1, 1, 1))
        bpy.data.objects["Sun"].data.energy = 10
        bpy.ops.object.light_add(type='SUN', align='WORLD',
                                location=(-1.22, 8.5, 3.5),
                                scale=(1, 1, 1))
        bpy.data.objects["Sun.001"].data.energy = 10
        bpy.data.objects["Sun.001"].use_shadow = False
        
        bpy.context.scene.render.film_transparent = False

        # bg = world.node_tree.nodes['Background']
        # bg.inputs[0].default_value[:3] = (230/255, 253/255, 255/255)
        # bg.inputs[1].default_value = 1.0
    # elif:
    #     pass
    elif sky:
    ################################################################################

    # bpy.context.scene.render.film_transparent = background_transparent
    ########## OLD LIGHT SETUP ##########
    ###
        sky_texture = bpy.context.scene.world.node_tree.nodes.new("ShaderNodeTexSky")
        bg = bpy.context.scene.world.node_tree.nodes["Background"]
        bpy.context.scene.world.node_tree.links.new(bg.inputs["Color"], sky_texture.outputs["Color"])
        #
        sky_texture.sky_type = 'PREETHAM' #'HOSEK_WILKIE' # or PREETHAM
        sky_texture.turbidity = 3.0
        sky_texture.ground_albedo = 0.4
        sky_texture.sun_intensity = 1.0


        sky_texture.sun_direction = mathutils.Vector((1.0, 0.0, 1.0))  # add `import mathutils` at the beginning of the script 
        sky_texture.sun_size = 90
    else:
        world = bpy.data.worlds[0]
        nodes = world.node_tree.nodes


        paths_tohdris = ['aristea_wreck_puresky_8k.hdr', 
                         'drackenstein_quarry_puresky_4k.hdr', 
                         'kloofendal_48d_partly_cloudy_puresky_4k.hdr',
                         ]
        
        path_to_hdr = 'skys/'+ paths_tohdris[2]
        # 'skys/skidpan_4k.hdr'
        # skys/kloofendal_48d_partly_cloudy_puresky_4k.hdr



        ########## static Sky ##########
        # hdr_image_path = bpy.path.abspath(path_to_hdr) # Source: https://polyhaven.com/a/lilienstein
        # ################################################################################
        # # Tutorial #2: Use image-based lighting
        # hdr_image = bpy.data.images.load(hdr_image_path)
        # # Create new nodes in World shader
        # # We use Mapping node to have the option to rotate the background image
        # texture_coordinate_node = nodes.new(type="ShaderNodeTexCoord")
        # mapping_node = nodes.new(type="ShaderNodeMapping")

        # environment_texture_node = nodes.new(type="ShaderNodeTexEnvironment")
        # environment_texture_node.image = hdr_image

        # # Connect shader nodes
        # links = world.node_tree.links
        # links.new(texture_coordinate_node.outputs["Generated"], mapping_node.inputs["Vector"])
        # links.new(mapping_node.outputs["Vector"], environment_texture_node.inputs["Vector"])
        # links.new(environment_texture_node.outputs["Color"], nodes["Background"].inputs["Color"])
        # from math import radians

        # bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[1].default_value[0] = 0.8
        # bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[1].default_value[0] = 0.1
        # bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[1].default_value[0] = 0.2
        
        # bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[0] = radians(26)
        # bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[0] = radians(-13.2)
        # bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[0] = radians(20.8)
                
        ########## static Sky ##########
        original = True
        if original:
            if sun_location is not None:
                bpy.ops.object.light_add(type='SUN', align='WORLD',
                                    location=(-1.7, 1.5, 5), 
                                    scale=(1, 1, 1))
            else:
                bpy.ops.object.light_add(type='SUN', align='WORLD',
                    location=(-1.7, 1.5, 5), 
                    scale=(1, 1, 1))

            if sun_rotation is not None:
                bpy.data.objects["Sun"].rotation_euler[0] = radians(sun_rotation[0])
                bpy.data.objects["Sun"].rotation_euler[1] = radians(sun_rotation[1])
                bpy.data.objects["Sun"].rotation_euler[2] = radians(sun_rotation[2])
            else:
                bpy.data.objects["Sun"].rotation_euler[0] = radians(58)
                bpy.data.objects["Sun"].rotation_euler[1] = radians(2)
                bpy.data.objects["Sun"].rotation_euler[2] = radians(30)

            bpy.data.objects["Sun"].data.energy = sun_strength # 25
            if sun_color is not None:
                bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (sun_color[0],
                                                                                                  sun_color[1], 
                                                                                                  sun_color[2], 1)
                bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 2
            else:
                bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1,1,1,1)
                bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 2

            bpy.context.scene.render.film_transparent = False
        else:
            world = bpy.data.worlds['World']
            world.use_nodes = True
            bg = world.node_tree.nodes['Background']
            bg.inputs[0].default_value[:3] = (1.0, 1.0, 1.0)
            bg.inputs[1].default_value = 1.0
            bpy.ops.object.light_add(type='SUN', align='WORLD',
                             location=(0, 0, 0), scale=(1, 1, 1))
            bpy.data.objects["Sun"].data.energy = 1.5

            # rotate camera
            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
            bpy.ops.transform.resize(value=(10, 10, 10), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                             orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1,
                             use_proportional_connected=False, use_proportional_projected=False)
        #bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1,1,1,1)
        #bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 2
        
        #bpy.context.scene.render.film_transparent = False
    ###
    if render_eng == 'CYCLES':
        setup_cycles(cycle=True)
    else:
        # pass
        setup_eevee()

    return scene
