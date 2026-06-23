import os
import torch
import joblib
import trimesh
import numpy as np 
import multiprocessing
from scipy import linalg
from itertools import repeat
from pytorch3d.ops.knn import knn_points
from scipy.ndimage import uniform_filter1d

from src.utils.process_utils import SMPLX_FINGERTIPS, SMPLX_JOINTS
from src.utils.contact_utils import batch_mesh_contains_points, _pre_compute_closest_dist, \
                                        get_sample_intersect_volume, intersect, compute_jerk


THRESH_LIST = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01]


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative dataset set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative dataset set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_skating_ratio(motions):
    # motions [B, J, 3, T]

    thresh_height = 0.05 # 10
    fps = 30.0
    thresh_vel = 0.50 # 20 cm /s 
    avg_window = 5 # frames

    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up

    # motions [bs, 22, 3, max_len]
    if type(motions) == np.ndarray:
        verts_feet = motions[:, [SMPLX_JOINTS["left_foot"], SMPLX_JOINTS["right_foot"]], :, :]  # [bs, 2, 3, max_len]
    else:
        verts_feet = motions[:, [SMPLX_JOINTS["left_foot"], SMPLX_JOINTS["right_foot"]], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
 
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
    
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg
    

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Flag for both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]

    
    # [bs]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]
    
    return skating_ratio, skate_vel

def calculate_traj_loc_error(dist_error, mask, thresh_list, strict=True):
 
    ''' dist_error shape [5]: error for each kps in metre
      Two threshold: 20 cm and 50 cm.
    If mean error in sequence is more then the threshold, fails
    return: traj_fail(0.2), traj_fail(0.5), all_kps_fail(0.2), all_kps_fail(0.5), all_mean_err.
        Every metrics are already averaged.
    '''
 
    traj_fail_list, loc_fail_list = [], []
    
    for i in range(len(thresh_list)):
        
        loc_fail_list.append((dist_error > thresh_list[i]).sum() / mask.sum().item())


        if strict:
            traj_fail_list.append(1.0 - (dist_error <= thresh_list[i]).all())
        else:
            traj_fail_list.append(1.0 - (dist_error <= thresh_list[i]).any())
            
    return  {'traj_fail_list': traj_fail_list, 
            'loc_fail_list': loc_fail_list,
            'avg_error': 100 * (dist_error.sum() / mask.sum()).item()}   


 
       
def calculate_trajectory_diversity(trajectories, lengths):
    ''' Standard diviation of point locations in the trajectories
    Args:
        trajectories: [bs, rep, 196, 2]
        lengths: [bs]
    '''
    # [32, 2, 196, 2 (xz)]
    # mean_trajs = trajectories.mean(1, keepdims=True)
    # dist_to_mean = np.linalg.norm(trajectories - mean_trajs, axis=3)
    def traj_div(traj, length):
        # traj [rep, 196, 2]
        # length (int)
        traj = traj[:, :length, :]
        # point_var = traj.var(axis=0, keepdims=True).mean()
        # point_var = np.sqrt(point_var)
        # return point_var

        mean_traj = traj.mean(axis=0, keepdims=True)
        dist = np.sqrt(((traj - mean_traj)**2).sum(axis=2))
        rms_dist = np.sqrt((dist**2).mean())
        return rms_dist
        
    div = []
    for i in range(len(trajectories)):
        div.append(traj_div(trajectories[i], lengths[i]))
    return np.array(div).mean()


def compute_metrics(res_dict, **kwargs):

    key = kwargs['key']
    thresh_list = kwargs['thresh_list']

    metric_key_dict = {
                       'hmp': 'tlcontrol',
                       'body_only': 'separate',
                       'body_hand': 'compact',
                       'with_beat2': 'compact_with_beat2'
                       }

    joints_of_interest = res_dict[key]['out_dict']['joints'].transpose(0, 2, 3, 1)
    skating_ratio, skate_vel = calculate_skating_ratio(joints_of_interest[:, :, [0, 2, 1]])

    # separate optimization setting
    if key == 'body_only':
        closeness_err = np.concatenate([res_dict['hand']['out_dict']['right']['joints'], 
                                res_dict['hand']['out_dict']['left']['joints']], axis=-2) - res_dict['hand']['target']['joints'].cpu().numpy()    
        target_mask = res_dict['hand']['target_mask']['joints'][..., 0].cpu().numpy()

    if key in ['body_hand', 'with_beat2']:

        closeness_err = res_dict[key]['out_dict']['joints'] - res_dict[key]['target']['joints'].cpu().numpy()
        target_mask = res_dict[key]['target_mask']['joints'][..., 0].cpu().numpy()

    # hmp hand optimization setting
    if key == 'hmp':
        closeness_err = res_dict[key]['out_dict']['hand_joints'] - res_dict[key]['target']['joints'].cpu().numpy()
        target_mask = res_dict[key]['target_mask']['joints'][..., 0].cpu().numpy()
    
    closeness_err = np.linalg.norm(closeness_err, ord=2, axis=-1)
    closeness_err = closeness_err * target_mask

    traj_err = calculate_traj_loc_error(closeness_err, target_mask, thresh_list)

    return {f'{metric_key_dict[key]}_foot_skating_ratio': skating_ratio,
            f'{metric_key_dict[key]}_foot_skating_vel': skate_vel,
            f'{metric_key_dict[key]}_traj_err': traj_err}

    

class ObjectContactMetrics:
    def __init__(self, device, use_multiprocessing=True):
        self.volume = {}
        self.depth = {}
        self.contact_ratio = {}

        self._use_multiprocessing = use_multiprocessing
        self.device = device
        self.pool = multiprocessing.Pool(64)
        return

    def compute_hoi_metrics(self, sbj_vertices, sbj_faces, obj_vertices, obj_faces, obj_poses, start_id, end_id):
        assert (len(sbj_vertices.shape) == 3)
        n_frames = sbj_vertices.shape[0]
        n_sbj_vertices = sbj_vertices.shape[1]
        assert (sbj_vertices.shape[2] == 3)

        volume_list = []
        depth_list = []
        ratio_list = []

        contact_frames = []

    
        if start_id == -1 or start_id == end_id:
            volume_list = [0.0]
            depth_list = [0.0]
            ratio_list = [0.0]
        else:
            
            if self._use_multiprocessing:
                dist_to_closest_point_list = self.pool.starmap(_pre_compute_closest_dist, zip(np.arange(start=start_id, stop=end_id), repeat(obj_faces),
                                                                                     repeat(obj_vertices), repeat(sbj_vertices)))
                self.pool.close()
                self.pool.join()

            for frame in range(start_id, end_id):
                obj_triangles = obj_vertices[:,obj_faces.numpy()]

                exterior = batch_mesh_contains_points(sbj_vertices[None, frame, :].float().to(self.device),
                                                    obj_triangles[None, frame, :, :].float().to(self.device),
                                                    torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745]).to(self.device))
                penetr_mask = ~exterior.squeeze(dim=0)

                if penetr_mask.sum() == 0:
                    max_depth = 0.0
                    volume = 0.0
                    contact_ratio = 0.0
                else:
                    if self._use_multiprocessing:
                        self._dist_to_closets_point_on_obj = dist_to_closest_point_list[frame-start_id]
                    else:
                        self._dist_to_closets_point_on_obj = _pre_compute_closest_dist(frame, obj_faces, obj_vertices, sbj_vertices)

                    max_depth, volume = self.compute_interpenetration_volume_depth_mesh_2_mesh(obj_faces=obj_faces,
                                                                                            obj_vertices=obj_vertices[frame],
                                                                                            sbj_faces=sbj_faces,
                                                                                            sbj_vertices=sbj_vertices[frame],
                                                                                            penetr_mask=penetr_mask.detach().cpu().numpy())

                    contact_ratio = self.compute_contact_ratio(obj_faces=obj_faces, obj_vertices=obj_vertices[frame],
                                                            sbj_vertices=sbj_vertices[frame])
                    contact_frames.append(frame-start_id)

                volume_list += [volume]
                depth_list += [max_depth]
                ratio_list += [contact_ratio]

        if len(contact_frames) == 0:
            contact_frames = [0]
            contact_frames_res = []
        else:
            contact_frames_res = contact_frames
        contact_frames = np.array(contact_frames)

        self.volume = {
            "inter_volume_mean": np.mean(volume_list), "inter_volume_contact": np.mean(np.array(volume_list)[contact_frames]), "inter_volume_max": np.max(volume_list),
            "inter_volume_mean_last_5": np.mean(volume_list[-5:]),
            "inter_volume_last": volume_list[-1],
        }
        self.depth = {
            "inter_depth_mean": np.mean(depth_list), "inter_depth_contact": np.mean(np.array(depth_list)[contact_frames]), "inter_depth_max": np.max(depth_list),
            "inter_depth_mean_last_5": np.mean(depth_list[-5:]),
            "inter_depth_last": depth_list[-1]
        }
        self.contact_ratio = {
            "contact_ratio_mean": np.mean(ratio_list), "contact_ratio_contact": np.mean(np.array(ratio_list)[contact_frames]), "contact_ratio_max": np.max(ratio_list),
            "contact_ratio_mean_last_5": np.mean(ratio_list[-5:]),
            "contact_ratio_last": ratio_list[-1], "contact_frames": contact_frames_res
        }

        try:
            jerk_pos = compute_jerk(obj_poses[:,:3])
            jerk_ang = compute_jerk(obj_poses[:,3:6])

            self.jerk = {"jerk_pos": jerk_pos, "jerk_ang": jerk_ang}
        except:
            self.jerk = {"jerk_pos": 0, "jerk_ang": 0}

        return

    def compute_interpenetration_volume_depth_mesh_2_mesh(self, obj_faces, obj_vertices, sbj_faces, sbj_vertices, penetr_mask):
        """
        Original source: https://github.com/hwjiang1510/GraspTTA/tree/master/metric
        https://github.com/CGAL/cgal-swig-bindings
        """
        #if do_intersect_single_frame_cgal(sbj_vertices, sbj_faces, obj_vertices, obj_faces):
        volume = get_sample_intersect_volume(
            sample_info={
                "sbj_verts": sbj_vertices,
                "obj_verts": obj_vertices,
                "sbj_faces": sbj_faces,
                "obj_faces": obj_faces
            }, mode="voxels"  # voxels
        )
        if volume is None:
            volume = 0.0
            max_depth = 0.0
        else:
            float(volume*1e6)

        max_depth = self.compute_max_depth(obj_faces, obj_vertices, sbj_vertices, penetr_mask)
        # else:
        #     volume = 0.0
        #     max_depth = 0.0

        return max_depth, volume

    def compute_contact_ratio(self, obj_faces, obj_vertices, sbj_vertices, in_contact_threshold=0.005):
        """
        nr. sbj vertices / nr. sbj vertices close to object (below in_contact_threshold)
        """
        n_sbj_vertices = sbj_vertices.shape[0]

        n_verts_in_contact = np.sum(self._dist_to_closets_point_on_obj < in_contact_threshold)

        ratio = n_verts_in_contact / n_sbj_vertices

        return ratio

    def compute_max_depth(self, obj_faces, obj_vertices, sbj_vertices, penetr_mask):
        """
        Original source: https://github.com/hwjiang1510/GraspTTA/tree/master/metric/penetration.py
        """
        obj_mesh = trimesh.Trimesh(vertices=obj_vertices, faces=obj_faces)
        trimesh.repair.fix_normals(obj_mesh)

        if penetr_mask.sum() == 0:
            max_depth = 0.0
        else:
            max_depth = self._dist_to_closets_point_on_obj[penetr_mask == 1].max()

        return max_depth



def save_metrics(gen_motion_dict, affordance_dict, out_dict):
    
    
    k = 0 
    
    metric_dict = {}
    
    min_loss_idx = np.argmin(out_dict['hist'][k]['loss_sum'])
    
 
    object_dict = affordance_dict['object_dict']
    command_list = affordance_dict['command_list']
   
    # closeness metrics
    metric_dict['closeness'] = np.sqrt(out_dict['hist'][k]['closeness_loss'][min_loss_idx].item()) 
    
    # likelihood metric 
    metric_dict['likelihood'] = out_dict['hist'][k]['likelihood_loss'][min_loss_idx].item()
    
    # foot skating metric 
    metric_dict['foot_skating'] = out_dict['hist'][k]['contact_v_loss'][min_loss_idx].item()
     
    # lightweight to initialize
    contact_collision_metric = ObjectContactMetrics(device='cuda')
    
    # object penetration metric 
    if object_dict != {}:

        try: 
            inside_flag_list = []
    
            contact_collision_metric.compute_hoi_metrics(sbj_vertices=torch.tensor(gen_motion_dict['vertices']),
                                                        sbj_faces=gen_motion_dict['faces'],
                                                        obj_vertices=torch.tensor(object_dict['vertices']), 
                                                        obj_faces=torch.tensor(object_dict['faces']),
                                                        obj_poses=torch.cat([object_dict['trans'], object_dict['pose']], dim=1), 
                                                        start_id=0, 
                                                        end_id=object_dict['vertices'].shape[0])

            metric_dict.update(contact_collision_metric.volume)
            metric_dict.update(contact_collision_metric.depth)
            metric_dict.update(contact_collision_metric.contact_ratio)
            metric_dict.update(contact_collision_metric.jerk)

            fingertips = gen_motion_dict['vertices'][:, SMPLX_FINGERTIPS]
        
            for _i_, mesh_v in enumerate(object_dict['vertices']):
                
                obj_mesh = trimesh.Trimesh(vertices=mesh_v, faces=object_dict['faces'])
                inside_flag_list.append(obj_mesh.contains(fingertips[_i_]))
                
            

                inside_flag_list = torch.tensor(np.array(inside_flag_list)).float()
                    
                closest_distances = knn_points(torch.tensor(fingertips).cuda(), 
                                            torch.tensor(object_dict['vertices']).cuda(), 
                                                return_nn=False, K=1, return_sorted=False)[0]
                
                penetration = closest_distances.cpu() * inside_flag_list[..., None]
                metric_dict['penetration_mean'] = penetration.mean()
                metric_dict['self_penetration_loss'] = None
        except:
            metric_dict['penetration_mean'] = None
            metric_dict['self_penetration_loss'] = None
        
    elif command_list != []:
        # self penetration metric
        from src.optim.condition import object_penetration_loss

        self_penetration_loss = object_penetration_loss(mesh3d={'vertices': torch.from_numpy(gen_motion_dict['vertices'])[None], 
                                        'faces': gen_motion_dict['faces'].astype(np.int64)},
                                 
                                obj_dict_list=[{}])


        CONTACT_CLOSENESS_THRESH = 0.025

        hand_penetration_volume, hand_penetration_depth = 0, 0
 
        rhand_verts = gen_motion_dict['vertices'][:, watertight_conversion_dict['basemesh_rhand_vertex_ids']]
        lhand_verts = gen_motion_dict['vertices'][:, watertight_conversion_dict['basemesh_lhand_vertex_ids']]
        rhand_faces = watertight_conversion_dict['watertight_rhand_faces']
        lhand_faces = watertight_conversion_dict['watertight_lhand_faces']

        body_wo_hand_verts = gen_motion_dict['vertices'][:, base2watertight_wo_hand]
        body_wo_hand_faces = watertight_wo_hand_faces

  

        contact_collision_metric.compute_hoi_metrics(sbj_vertices=torch.tensor(body_wo_hand_verts),
                                                    sbj_faces=body_wo_hand_faces,
                                                    obj_vertices=torch.tensor(rhand_verts), 
                                                    obj_faces=torch.tensor(rhand_faces),
                                                    obj_poses=None, 
                                                    start_id=0, 
                                                    end_id=rhand_verts.shape[0])
        
        hand_penetration_volume += contact_collision_metric.volume['inter_volume_mean']
        hand_penetration_depth += contact_collision_metric.depth['inter_depth_max']

        contact_collision_metric.pool = multiprocessing.Pool(64)
        contact_collision_metric.compute_hoi_metrics(sbj_vertices=torch.tensor(body_wo_hand_verts),
                                                    sbj_faces=body_wo_hand_faces,
                                                    obj_vertices=torch.tensor(lhand_verts), 
                                                    obj_faces=torch.tensor(lhand_faces),
                                                    obj_poses=None, 
                                                    start_id=0, 
                                                    end_id=lhand_verts.shape[0])    

        hand_penetration_volume += contact_collision_metric.volume['inter_volume_mean']
        hand_penetration_depth += contact_collision_metric.depth['inter_depth_max']

        metric_dict['hand_penetration_volume'] = hand_penetration_volume
        metric_dict['hand_penetration_depth'] = hand_penetration_depth
        metric_dict['self_penetration_loss'] = self_penetration_loss

        distances_list = []
        successful_contact_pairs, total_contact_pairs = 0, 0

        # compute contact accuracy 
        for keyf_idx, action_vertices in command_list:

            if len(action_vertices) == 0:
                continue

            # 1, T, ACT_VERTICES, 2, 3
            verts_subset = gen_motion_dict['vertices'][keyf_idx[0]:keyf_idx[1], action_vertices] 
            
            verts_diff = verts_subset[:, :, :, None] - verts_subset[:, :, None, :]

            # Shape: (T', ACT_VERTICES, 2, 2)
            distances = np.linalg.norm(verts_diff, axis=-1)[:, :, 0, 1] 
          
            distances_list.append(distances)
            successful_contact_pairs += (distances < CONTACT_CLOSENESS_THRESH).sum().item()
        
            total_contact_pairs += len(action_vertices) * (keyf_idx[1] - keyf_idx[0])
 
        metric_dict['contact_distances_list'] = distances_list
        metric_dict['contact_accuracy'] = successful_contact_pairs / total_contact_pairs

  
    
    return metric_dict



# Resolve data path relative to the repo root (src/utils/metric_utils.py -> repo root),
# so import works regardless of the current working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
watertight_conversion_dict = joblib.load(os.path.join(_REPO_ROOT, 'data/body_models/watertight/conversion_dict.pkl'))
base2watertight_wo_hand = list(watertight_conversion_dict['watertight2base_wo_hand'].values())
watertight_wo_hand_faces = np.array(list(watertight_conversion_dict['watertight_wo_hand_faces']))
