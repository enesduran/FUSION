# compute correlation between right hand and left hand poses. It is important for our 
import os 
import sys
import torch
import dcor
import glob 
import joblib 
import matplotlib
import numpy as np 
from tqdm import tqdm 
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from utils.process_utils import SMPLX_JOINTS
from utils.transforms3d import transform_body_pose

data_cfg = OmegaConf.load('configs/data.yaml')


def so3_log(R):
    # R: (..., 3, 3)
    cos_theta = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    cos_theta = cos_theta.clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)

    w = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)

    half_angle = theta / (2 * torch.sin(theta))
    # Taylor expansion near θ ≈ 0: θ/(2sinθ) ≈ 1/2 + θ²/12
    small = (theta.abs() < 1e-4)
    half_angle = torch.where(small, 0.5 + theta**2 / 12, half_angle)

    return w * half_angle.unsqueeze(-1)


def concatenate_data(data_loaded):

    time_keys = data_loaded.keys()
    feat_keys = data_loaded[list(data_loaded.keys())[0]].keys()
    feat_keys = ['rots']

    data_concatenated = {key: [] for key in feat_keys}

    for feat_key in feat_keys:
        for time_key in time_keys:

            if type(data_loaded[time_key][feat_key]) in [np.ndarray, torch.Tensor]:
                data_concatenated[feat_key].append(data_loaded[time_key][feat_key][None, ...])
            else:
                data_concatenated[feat_key].append(data_loaded[time_key][feat_key])

        data_concatenated[feat_key] = np.vstack(data_concatenated[feat_key])

    return data_concatenated


def parse_rotations(data, convert_to_so3=False):

    N, T = data['rots'].shape[:2]
    
    if convert_to_so3:

        body_rots = so3_log(transform_body_pose(data['rots'][:, :, 3:75], 'aa->rot')).numpy()
        hand_rots_combined = np.concatenate([so3_log(transform_body_pose(data['rots'][:, :, 75:120], 'aa->rot')).numpy(), 
                                        so3_log(transform_body_pose(data['rots'][:, :, 120:165], 'aa->rot')).numpy()], axis=-1)

    else:
        # aa to 6d
        body_rots = transform_body_pose(data['rots'][:, :, 3:75], 'aa->6d').numpy()
        hand_rots_combined = np.concatenate([transform_body_pose(data['rots'][:, :, 75:120], 'aa->6d').numpy(),
                                        transform_body_pose(data['rots'][:, :, 120:165], 'aa->6d').numpy()], axis=-1)


    return body_rots.reshape(N * T, -1), hand_rots_combined.reshape(N * T, -1)
    

def calculate_cca_correlation(X, Y, n_components=1, normalize=True):
    """
    Computes the Canonical Correlation between two sequences via GPU-accelerated
    Cholesky + SVD (torch_cca).
    X, Y: Arrays of shape (T, D) — numpy arrays.
    n_components: Number of canonical components to return.
    """
    if normalize:
        scaler_body = StandardScaler()
        scaler_hand = StandardScaler()
        X = scaler_body.fit_transform(X)
        Y = scaler_hand.fit_transform(Y)

    return torch_cca(X, Y, n_components).cpu().numpy()


def calculate_distance_correlation(X, Y, n_components=0.95, apply_pca=True):
    """
    Computes the Distance Correlation between two sequences.
    X, Y: Arrays of shape (T, J * 6)
    """

    if apply_pca:
        pca_x = PCA(n_components=n_components)
        pca_y = PCA(n_components=n_components)
    
        X = pca_x.fit_transform(X)
        Y = pca_y.fit_transform(Y)
    
    return dcor.distance_correlation(X, Y)
    

def calculate_rv_coefficient_batch(X, Y, batch_size=10000, T_threshold=80000):
    """
    Computes the RV Coefficient between two sequences with batched processing for large T.
    X, Y: Arrays of shape (T, D) where D = J * 6
    
    Args:
        X: Tensor of shape (T, D)
        Y: Tensor of shape (T, D)
        batch_size: Size of batches for processing when T > T_threshold
        T_threshold: Threshold for T above which to use batching
    
    Returns:
        rho: RV coefficient (scalar)
    """
    T, D = X.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Center the data
    X_mean = torch.mean(X, dim=0)
    Y_mean = torch.mean(Y, dim=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    
    sqrt_T = torch.sqrt(torch.tensor(T, dtype=torch.float32))

    X_scaled = X_centered.type(torch.float32) / sqrt_T
    Y_scaled = Y_centered.type(torch.float32) / sqrt_T
    
    # 2. Decide whether to use batched computation
    if T < T_threshold:
        # Small T: process all at once on GPU
        X_scaled = X_scaled.to(device)
        Y_scaled = Y_scaled.to(device)
        sqrt_T = sqrt_T.to(device)
        
        # Compute covariance matrices (D x D)
        Sxx = X_scaled.T @ X_scaled
        Sxy = X_scaled.T @ Y_scaled
        Syy = Y_scaled.T @ Y_scaled
        
        # Compute RV coefficient
        numerator = torch.trace(Sxy @ Sxy.T)
        denominator = torch.sqrt(torch.trace(Sxx @ Sxx)) * torch.sqrt(torch.trace(Syy @ Syy))
        rho = numerator / (denominator + 1e-9)
        
        return min(1.0, rho.item())
    
    else:
        # Large T: use batched computation
         
        # Initialize accumulator matrices on GPU (D x D)
        Sxx = torch.zeros(D, D, dtype=torch.float32, device=device)
        Sxy = torch.zeros(D, D, dtype=torch.float32, device=device)
        Syy = torch.zeros(D, D, dtype=torch.float32, device=device)
        
        # Process in batches
        num_batches = (T + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, T)
            
            # Move batch to GPU
            X_batch = X_scaled[start_idx:end_idx].to(device)
            Y_batch = Y_scaled[start_idx:end_idx].to(device)
            
            # Accumulate covariance contributions
            # Sxx += X_batch.T @ X_batch
            Sxx.addmm_(X_batch.T, X_batch)
            # Sxy += X_batch.T @ Y_batch
            Sxy.addmm_(X_batch.T, Y_batch)
            # Syy += Y_batch.T @ Y_batch
            Syy.addmm_(Y_batch.T, Y_batch)
            
            # Clear GPU cache for the batch
            del X_batch, Y_batch
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Compute RV coefficient
        numerator = torch.trace(Sxy @ Sxy.T)
        denominator = torch.sqrt(torch.trace(Sxx @ Sxx)) * torch.sqrt(torch.trace(Syy @ Syy))
        rho = numerator / (denominator + 1e-9)
        
        return min(1.0, rho.item())



def calculate_rv_coefficient_full(X, Y, rotation_type='6d'):

    dim = 6 if rotation_type == '6d' else 9

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    j1 = int(X.shape[1]/dim)
    j2 = int(Y.shape[1]/dim)

    rv_coefficients = np.zeros((j1, j2))


    for i in range(j1):
        for j in range(j2):
            rv_coefficients[i, j] = calculate_rv_coefficient_batch(X[:, i*dim:(i+1)*dim], Y[:, j*dim:(j+1)*dim])

    return rv_coefficients


def calculate_mutual_information(rhos):
    # gaussian_mi = -0.5 * np.sum(np.log(1.0 - rhos**2 + 1e-12))
    gaussian_mi = -0.5 * np.log(1.0 - rhos**2 + 1e-12)
    return gaussian_mi


def torch_cca(X, Y, n_components, batch_size=10000):
    """
    CCA implementation in PyTorch (works on GPU).
    Uses batched covariance accumulation for large datasets.
    X, Y: numpy arrays of shape (n_samples, n_features)
    """
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = X.shape[0]

    # Center on CPU (cheap, avoids moving full data to GPU)
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    Dx, Dy = X.shape[1], Y.shape[1]
    Cxx = torch.zeros(Dx, Dx, dtype=torch.float32, device=device)
    Cyy = torch.zeros(Dy, Dy, dtype=torch.float32, device=device)
    Cxy = torch.zeros(Dx, Dy, dtype=torch.float32, device=device)

    # Accumulate covariance in batches on GPU
    for i in range(0, n, batch_size):
        Xb = X[i:i+batch_size].to(device)
        Yb = Y[i:i+batch_size].to(device)
        Cxx.addmm_(Xb.T, Xb)
        Cyy.addmm_(Yb.T, Yb)
        Cxy.addmm_(Xb.T, Yb)
        del Xb, Yb

    Cxx /= (n - 1)
    Cyy /= (n - 1)
    Cxy /= (n - 1)

    # Regularize for numerical stability
    reg = 1e-12
    Cxx += reg * torch.eye(Dx, device=device)
    Cyy += reg * torch.eye(Dy, device=device)

    # Cholesky + SVD for canonical correlations
    Lx_inv = torch.linalg.inv(torch.linalg.cholesky(Cxx))
    Ly_inv = torch.linalg.inv(torch.linalg.cholesky(Cyy))

    M = Lx_inv @ Cxy @ Ly_inv.T
    S = torch.linalg.svdvals(M)

    return S[:n_components]


def heatmap(data, title, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", save_path='', vmin=None, vmax=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 9))

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, vmin=vmin, vmax=vmax)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels, rotation=-45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title)

    if save_path != '':
        plt.savefig(save_path)
 
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def load_generated_dataset():
    
    files = sorted(glob.glob('fusion_runs/main/0/vis_res/*.npy'))
    
    data = []
    for file in files:

        data.append(np.load(file, allow_pickle=True).item()['pose'][:, 9:])
 
    return np.concatenate(data, axis=0)
    
def load_dataset(datapath):

    print(f'Loading data from {datapath}...')
    
    data_loaded = {}

    file_list = sorted(glob.glob(os.path.join(datapath, '*.p')))
    
    file_list = np.random.choice(file_list, size=min(100_000, len(file_list)), replace=False)

    for filename in tqdm(file_list):
        data_loaded[int(filename.split('/')[-1].split('.')[0])] = joblib.load(filename)

    return data_loaded



if __name__ == '__main__':

    full_body_names = list(SMPLX_JOINTS.keys())[1:] 
    body_names = list(SMPLX_JOINTS.keys())[1:25]
    hand_names = list(SMPLX_JOINTS.keys())[25:]
 
    data_generated_fullbody = load_generated_dataset()

    cca_setting_generated = None
    mutual_information_setting_generated = None
    try:
        data_generated_body = data_generated_fullbody[:, :144]
        data_generated_hand = data_generated_fullbody[:, 144:]

        cca_setting_generated = calculate_cca_correlation(data_generated_body, data_generated_hand, 5, normalize=False)
        print(f'CCA Setting Generated {cca_setting_generated}')
        mutual_information_setting_generated = calculate_mutual_information(cca_setting_generated)
        print(f'Mutual Information Setting Generated {mutual_information_setting_generated}')

    except:
        print('Failed to calculate rv coefficients for generated dataset')
    

    # load data 
    data = load_dataset('data/motion/precomputed/train')
     
    # setting1: original hand motion, no pose flipping 
    data_hand_raw = {key: data[key] for key in data.keys() if data[key]['body_dataset_name'] in data_cfg.hand_art_dataset_list}
    data_hand_raw = {key: data[key] for key in data_hand_raw.keys() if not data[key]['augment_flag']}

    # setting4: all hand motions, pose flipping, time augmentation 
    data_hand_all_aug = {key: data[key] for key in data.keys()}
    
   
    os.makedirs('fusion_runs/renders/correlation_analysis', exist_ok=True)

    # make a correlation matrix of the rv coefficients 
    setting1_body, setting1_hand = parse_rotations(concatenate_data(data_hand_raw), convert_to_so3=False)
    setting1_fullbody = np.concatenate([setting1_body, setting1_hand], axis=-1)   
    

    rv_setting1_fullbody = calculate_rv_coefficient_full(setting1_fullbody, setting1_fullbody, rotation_type='6d')
    rv_setting1_hand_body = rv_setting1_fullbody[:len(body_names), len(body_names):] 

    im, cbar = heatmap(rv_setting1_fullbody, f'Hand Dataset/Not Augmented. Mean: {rv_setting1_fullbody.mean():.2f}, Max: {rv_setting1_fullbody.max():.2f}', full_body_names, full_body_names, 
                                        save_path='fusion_runs/renders/correlation_analysis/rv_setting1_fullbody.png')

    im, cbar = heatmap(rv_setting1_hand_body, f'Hand Dataset/Not Augmented. Mean: {rv_setting1_hand_body.mean():.2f}, Max: {rv_setting1_hand_body.max():.2f}', body_names, hand_names, 
                                        save_path='fusion_runs/renders/correlation_analysis/rv_setting1_hand_body.png')

    cca_setting1 = calculate_cca_correlation(setting1_body, setting1_hand, 5)
    print(f'CCA Setting 1 {cca_setting1}')
    mutual_information_setting1 = calculate_mutual_information(cca_setting1)
    print(f'Mutual Information Setting 1 {mutual_information_setting1}')

    setting4_body, setting4_hand = parse_rotations(concatenate_data(data_hand_all_aug), convert_to_so3=False)
    setting4_fullbody = np.concatenate([setting4_body, setting4_hand], axis=-1)   
    
    rv_setting4_fullbody = calculate_rv_coefficient_full(setting4_fullbody, setting4_fullbody, rotation_type='6d')
    rv_setting4_hand_body = rv_setting4_fullbody[:len(body_names), len(body_names):] 

    im, cbar = heatmap(rv_setting4_fullbody, f'All Dataset/Time Augmented. Mean: {rv_setting4_fullbody.mean():.2f}, Max: {rv_setting4_fullbody.max():.2f}', full_body_names, full_body_names, 
                                        save_path='fusion_runs/renders/correlation_analysis/rv_setting4_fullbody.png')

    im, cbar = heatmap(rv_setting4_hand_body, f'All Dataset/Time Augmented. Mean: {rv_setting4_hand_body.mean():.2f}, Max: {rv_setting4_hand_body.max():.2f}', body_names, hand_names, 
                                        save_path='fusion_runs/renders/correlation_analysis/rv_setting4_hand_body.png', 
                                        vmin=0, vmax=rv_setting1_hand_body.max())

    sample_idx = np.random.choice(len(setting4_body), len(setting1_body), replace=False)
    
    cca_setting4 = calculate_cca_correlation(setting4_body[sample_idx], setting4_hand[sample_idx], 5)
    print(f'CCA Setting 4 {cca_setting4}')
    mutual_information_setting4 = calculate_mutual_information(cca_setting4)
    print(f'Mutual Information Setting 4 {mutual_information_setting4}')

    
    # cca plot overlayed
    plt.figure()

    plt.plot(cca_setting1, label='Raw Data', marker='o')
    plt.plot(cca_setting4, label='Merged Data', marker='o')

    plt.ylim(0, 1)
    plt.xticks(np.arange(0, len(cca_setting1), 1))
    plt.xlabel('Idx')
    plt.ylabel('Correlation Values')
    plt.legend()
    plt.title('CCA Correlation')
    plt.savefig('fusion_runs/renders/correlation_analysis/cca_overlayed.png')


    plt.figure()

    plt.plot(mutual_information_setting1, label='Raw Data', marker='o')
    plt.plot(mutual_information_setting4, label='Merged Data', marker='o')

    plt.xticks(np.arange(0, len(mutual_information_setting1), 1))
    plt.xlabel('Idx')
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.title('Mutual Information')
    plt.savefig('fusion_runs/renders/correlation_analysis/mi_overlayed.png')
    
    
    with open('correlation_results.txt', 'w') as f:
        f.write(f'CCA Setting 1 {cca_setting1}\n')
        f.write(f'CCA Setting 4 {cca_setting4}\n')
        f.write(f'CCA Setting Generated {cca_setting_generated}\n')
    
        f.write(f'Mutual Information Setting 1 {mutual_information_setting1}\n')
        f.write(f'Mutual Information Setting 4 {mutual_information_setting4}\n')
        f.write(f'Mutual Information Setting Generated {mutual_information_setting_generated}\n')
 
 
# CCA Setting 1 [0.9709625  0.8196825  0.78840995 0.7672519  0.7309942 ]
# Mutual Information Setting 1 [1.4303229  0.55718696 0.48588884 0.44418642 0.38216323]
# CCA Setting 4 [0.94268006 0.79033166 0.7360413  0.70180273 0.6715751 ]
# Mutual Information Setting 4 [1.0975192  0.4899137  0.3901776  0.33915594 0.2998404 ]
