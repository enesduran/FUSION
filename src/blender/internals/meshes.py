import numpy as np

from .materials import colored_material
import bpy
import matplotlib
import numpy as np
import matplotlib.colors as matplcolors
from matplotlib.cm import register_cmap, get_cmap
######### DEFINE COLORMAPR ################

# Define colors from white to deep pink
colors_PINK = [(1, 1, 1), (0.98, 0.85, 0.9), (0.96, 0.7, 0.8), 
          (0.94, 0.5, 0.6), (0.92, 0.3, 0.4), (0.9, 0.1, 0.2)]

# Create the colormap
custom_pink_cmap = matplcolors.LinearSegmentedColormap.from_list('custom_pink', 
                                                                 colors_PINK)

# Register the colormap with matplotlib
register_cmap(name='custom_pink', cmap=custom_pink_cmap)


# green
# GT_SMPL = colored_material(0.009, 0.214, 0.029)
GT_SMPL = colored_material(0.035, 0.415, 0.122)
COLORMAPS = [matplotlib.cm.get_cmap('Blues'), 
             matplotlib.cm.get_cmap('Greys'), 
             matplotlib.cm.get_cmap('Purples'), 
             matplotlib.cm.get_cmap('Greens'),
             matplotlib.cm.get_cmap('Reds'),
             get_cmap('custom_pink')]
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

def prepare_meshes(data, canonicalize=True, always_on_floor=False):
    # if canonicalize:
    #     print("No canonicalization for now")

    # fix axis
    data[..., 1] = - data[..., 1]
    data[..., 0] = - data[..., 0]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data


class Meshes:
    def __init__(self, data, *, gt, mode, faces, canonicalize, always_on_floor,
                 fixed_color=None,bp=False, action_id=0, lengths=None, **kwargs):
        # data = prepare_meshes(data, canonicalize=canonicalize, always_on_floor=always_on_floor)
        import numpy as np
        self.bp = bp
        self.faces = np.load(faces)
        self.data = data
        self.mode = mode

        self.N = len(data)
        self.trajectory = data[:, :, [0, 1]].mean(1)
        self.lengths = lengths
        self.fixed_color = fixed_color
        self.action_id = action_id
        if lengths is None:
            if gt:
                self.mat = GT_SMPL
            else:
                self.colormap = COLORMAPS[action_id%len(COLORMAPS)]
                # self.mat = colored_material(*matplotlib.cm.get_cmap('Dark2')(0))
        else:
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
                elif fixed_color == 'pink':
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
                elif fixed_color == 'pink':
                    self.mat = (0.91, 0.198, 0.298, 1 ) #purplish

            elif mode == 'frame':
                self.mat = matplotlib.cm.get_cmap('Dark2')(action_id)
        self.temp_data = data.copy()
        self.last_idx = None

        x_traj = self.data.mean(1)[:, 0]
        y_traj = self.data.mean(1)[:, 1]
        z_traj = self.data.mean(1)[:, 2]

        from src.utils.blender_utils import savitzky_golay, plot_line
        molen = z_traj.shape[0]
        if molen % 2 == 0: 
            ws = molen - 1
        else:
            ws = molen
        
        tsteps = np.arange(molen)
        if (z_traj.max() - z_traj.min()) < 0.1 or mode=='sequence':
            self.z_traj_smooth = z_traj
            self.constant_z = True
        else:
            self.z_traj_smooth = savitzky_golay(z_traj,
                                                ws,
                                                3) # window size 51, polynomial order 3

            # az, bz = np.polyfit(tsteps, z_traj, 1)
            # self.z_traj_smooth = az * tsteps + bz
            self.constant_z = False

        self.y_traj_smooth = y_traj
        self.x_traj_smooth = x_traj
        
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

            rgbcolor = tuple(rgbcolor)
            mat = colored_material(*rgbcolor)
        elif self.mode == 'video':
            rgbcolor = self.mat
            mat = colored_material(*rgbcolor)
        elif self.mode == 'frame':
            rgbcolor = self.mat
            mat = colored_material(*rgbcolor)
        return mat

    def get_root(self, index):
        if self.constant_z:
            return np.array([self.x_traj_smooth[index], 
                             self.y_traj_smooth[index],
                             0])

        return np.array([self.x_traj_smooth[index], 
                         self.y_traj_smooth[index], 
                         self.z_traj_smooth[index]])
        
    def get_mean_root(self):
        return self.data.mean((0, 1))

    def load_in_blender(self, index, mat):
        vertices = self.data[index]
        faces = self.faces
        name = f"{str(index).zfill(4)}_{self.action_id}"

        if self.bp:
            from .tools import load_numpy_vertices_into_blender, load_numpy_vertices_into_blender_bp

            cmap = matplotlib.cm.get_cmap('Blues')
            rgbcolor = cmap(0.50)
            mat = colored_material(*rgbcolor)

            load_numpy_vertices_into_blender_bp(vertices, faces, name, self.bp, BP_COLORS, mat)
        else:

            from .tools import load_numpy_vertices_into_blender
            load_numpy_vertices_into_blender(vertices, faces, name, mat)

        return name

    def __len__(self):
        return self.N
