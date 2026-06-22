import numpy as np

from .materials import colored_material
import bpy
import matplotlib
import numpy as np
import matplotlib.colors as matplcolors
from matplotlib.cm import register_cmap, get_cmap

# green
# GT_SMPL = colored_material(0.009, 0.214, 0.029)

colors_Yl = [(1, 1, 0.8),        # Light yellow start
          (1, 1, 0.7),        # Light yellow
          (1, 0.95, 0.5),     # Medium yellow
          (1, 0.9, 0.3),      # Deeper yellow
          (1, 0.85, 0.2),     # Golden yellow
          (0.95, 0.75, 0.1)]  # Deep gold

# Create the colormap
custom_yellow_cmap = matplcolors.LinearSegmentedColormap.from_list('custom_yellow', 
                                                               colors_Yl)

# Register the colormap with matplotlib
register_cmap(name='custom_yellow', cmap=custom_yellow_cmap)


GT_SMPL = colored_material(0.035, 0.415, 0.122)
COLORMAPS = [matplotlib.cm.get_cmap('Blues'), 
             matplotlib.cm.get_cmap('Greys'), 
             matplotlib.cm.get_cmap('Purples'), 
             matplotlib.cm.get_cmap('Greens'),
             matplotlib.cm.get_cmap('Reds'),
             get_cmap('custom_yellow')]
# blue
# GEN_SMPL = colored_material(0.022, 0.129, 0.439)
# Blues => cmap(0.87)
GEN_SMPL = colored_material(0.035, 0.322, 0.615)


BP_COLORS = [
    colored_material(66/255, 106/255, 90/255),
    colored_material(0.6500, 0.175, 0.0043),
    colored_material(0.4500, 0.0357, 0.0349),
    colored_material(0.018, 0.059, 0.600),
    colored_material(0.032, 0.325, 0.521),
    colored_material(140/255, 94/255, 88/255)
]
clr_dict = [(0.016, 0.052, 0.195, 1),
            (0.097, 0.104, 0.222, 1),
            (0.222, 0.088, 0.137, 1),
            (0.152, 0.097, 0.064, 1), #blueish 
            (0.044, 0.064, 0.152, 1)] #yellowish

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

class ObjectMeshes:
    def __init__(self, data, mode, faces,
                 fixed_color=None,
                 action_id=0, lengths=None, **kwargs):
        # data = prepare_meshes(data, canonicalize=canonicalize, always_on_floor=always_on_floor)
        import numpy as np
        # import ipdb;ipdb.set_trace()

        self.faces = faces
        self.data = data
        self.mode = mode
        self.N = len(data)
        self.lengths = lengths
        self.fixed_color = fixed_color
        self.action_id = action_id
        if mode == 'sequence':
            self.mat = (0.005, 0.034, 0.089, 1) # default
            if fixed_color == 'red':
                self.colormap = COLORMAPS[4%len(COLORMAPS)]
            elif fixed_color == 'green':
                self.colormap = COLORMAPS[3%len(COLORMAPS)]
            elif fixed_color == 'blue':
                self.colormap = COLORMAPS[0%len(COLORMAPS)]
            elif fixed_color == 'grey' or fixed_color == 'gray':
                self.colormap = COLORMAPS[1%len(COLORMAPS)]
            elif fixed_color == 'purple':
                self.colormap = COLORMAPS[2%len(COLORMAPS)]
            elif fixed_color == 'yellow':
                self.colormap = COLORMAPS[5%len(COLORMAPS)]

        elif mode == 'video':
            self.mat = matplotlib.cm.get_cmap('Dark2')(action_id)
            self.mat = (0.044, 0.064, 0.152, 1)
            self.mat = (0.005, 0.034, 0.089, 1) # default
            if fixed_color == 'red':
                self.mat = (0.08, 0.008, 0.007, 1) # reddish
            elif fixed_color == 'green':
                self.mat = (0.012, 0.07, 0.013, 1) # greenish
            elif fixed_color == 'blue': 
                self.mat = (0.003, 0.06, 0.659, 1 ) #blueish
            elif fixed_color == 'grey' or fixed_color == 'gray':
                self.mat = (0.204, 0.157, 0.137, 1 ) #greyish
            elif fixed_color == 'purple':
                self.mat = (0.071, 0.038, 0.296, 1 ) #purplish
            elif fixed_color == 'yellow':
                self.mat = (0.446, 0.373, 0.0, 1 ) #purplish

        elif mode == 'frame':
            self.mat = matplotlib.cm.get_cmap('Dark2')(action_id)


    def get_sequence_mat(self, frac):
        # cmap = matplotlib.cm.get_cmap('Blues')
        if self.mode == 'sequence':
            cmap = self.colormap
            begin = 0.60
            end = 0.90
            rgbcolor = cmap(begin + (end-begin)*frac)
            rgbcolor = list(rgbcolor)
            if self.fixed_color == 'red':
                rgbcolor[1] = rgbcolor[1]*0.2
                rgbcolor[2] = rgbcolor[2]*0.2
                rgbcolor[0] = rgbcolor[0]*0.45
            elif self.fixed_color == 'green':
                rgbcolor[0] = rgbcolor[0]*0.2
                rgbcolor[2] = rgbcolor[2]*0.2
                rgbcolor[1] = rgbcolor[1]*0.45
            elif self.fixed_color == 'blue':
                rgbcolor[0] = rgbcolor[0]*0.2
                rgbcolor[1] = rgbcolor[1]*0.2
                rgbcolor[2] = rgbcolor[2]*0.45
            elif self.fixed_color == 'purple':
                rgbcolor[0] = rgbcolor[0]*0.2
                rgbcolor[1] = rgbcolor[1]*0.2
                rgbcolor[2] = rgbcolor[2]*0.45
            elif self.fixed_color == 'grey' or self.fixed_color == 'gray':
                rgbcolor[0] = rgbcolor[0]*0.2
                rgbcolor[1] = rgbcolor[1]*0.2
                rgbcolor[2] = rgbcolor[2]*0.2
            elif self.fixed_color == 'yellow':
                rgbcolor[0] = rgbcolor[0]*1.1
                rgbcolor[1] = rgbcolor[1]*1.1
                rgbcolor[2] = rgbcolor[2]*0.15

            rgbcolor = tuple(rgbcolor)
            mat = colored_material(*rgbcolor)
        elif self.mode == 'video':
            rgbcolor = self.mat
            mat = colored_material(*rgbcolor)
        elif self.mode == 'frame':
            rgbcolor = self.mat
            mat = colored_material(*rgbcolor)
        return mat

    def load_in_blender(self, index, mat):
        vertices = self.data[index]
        faces = self.faces
        name = f"object_{str(index).zfill(4)}_{self.action_id}"

        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)

        return name

    def __len__(self):
        return self.N
