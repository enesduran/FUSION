import glob 
import joblib
import itertools
import numpy as np



def compile_hoi_metrics():

    files = sorted(glob.glob('fusion_runs/main/0/object_guided_quant_res/*_metrics.p'))


    metric_names = ['inter_volume_mean', 'inter_depth_max', 'contact_ratio_contact']
    metrics_mean_dict = {elem: [] for elem in metric_names}


    for file in files:
        metric_file = joblib.load(file)

        for metric in metric_names:
            metrics_mean_dict[metric].append(metric_file[metric])
         
    for metric in metrics_mean_dict:
    
        if metric in ['contact_frames', 'self_penetration_loss']:
            metrics_mean_dict[metric] = None 
        else:
            metrics_mean_dict[metric] = np.mean(metrics_mean_dict[metric])

        print(metric, metrics_mean_dict[metric])

    print(f'contact_ratio_contact: {metrics_mean_dict["contact_ratio_contact"]}')
    print(f'inter_depth_max: {metrics_mean_dict["inter_depth_max"] * 1e3} mm')
    print(f'inter_volume_mean: {metrics_mean_dict["inter_volume_mean"] * 1e6} cm^3')


 

def compute_self_contact_metrics():

    rr = sorted(glob.glob('/is/cluster/fast/eduran2/fusion/fusion_runs/main/0/self_contact_quant_res/*_metrics.p'))
    contact_distances_list, contact_accuracy_list, penetration_volume_list = [], [], []
    max_depth_err_list = []
    
    interaction_dict = joblib.load('data/self_interaction/self_interaction_dict.p')
    interaction_dict_keys = list(interaction_dict.keys())

    for _i_, _rr_ in enumerate(rr):

        metrics = joblib.load(_rr_)

        contact_accuracy_list.append(metrics['contact_accuracy'])

        penetration_volume_list.append(metrics['hand_penetration_volume'] * 1e6)
        max_depth_err_list.append(metrics['hand_penetration_depth'] * 1e3)

        print(_i_, metrics['text_for_vid'], metrics['contact_accuracy'])
 
    penetration_volume_list = np.array(penetration_volume_list)
    max_depth_err_list = np.array(max_depth_err_list)

    contact_accuracy_mean = np.array(contact_accuracy_list).mean()
    idx = np.where(np.array(contact_accuracy_list) > -1)[0]
   
    

    for _i_, _rr_ in enumerate(rr):
        metrics = joblib.load(_rr_)
        
        if _i_ in idx:

            for _dist_ in metrics['contact_distances_list']:
                contact_distances_list.append(_dist_.flatten().tolist())
 
    
    flattened = list(itertools.chain.from_iterable(contact_distances_list))

    print('penetration_volume_mean', penetration_volume_list[:].mean())
    print('max_depth_err_mean', max_depth_err_list[idx].mean()) 

    # Convert to 1D NumPy array
    contact_distances_list = np.array(flattened, dtype=float)
 

    print('contact_distances_mean', contact_distances_list.mean())


    for threshold in [0.1, 0.05, 0.03, 0.01]:
        print('contact_accuracy {} cm:'.format(threshold*100), (contact_distances_list < threshold).sum() / contact_distances_list.shape[0])



def compute_embody_metrics():

    rr = sorted(glob.glob('/is/cluster/fast/eduran2/fusion/fusion_runs/main/0/keypoint_tracking_quant_res/*.p'))

    skating_ratio_list, skate_vel_list, avg_error_list, traj_fail_list, loc_fail_list = [], [], [], [], []
    
    for _i_, _rr_ in enumerate(rr):

        metrics = joblib.load(_rr_)

        skating_ratio_list.append(metrics['skating_ratio'])
        skate_vel_list.append(metrics['skate_vel'])
        avg_error_list.append(metrics['traj_err']['avg_error'])
        traj_fail_list.append(metrics['traj_err']['traj_fail_list'])
        loc_fail_list.append(metrics['traj_err']['loc_fail_list'])

    skating_ratio_list = np.array(skating_ratio_list)
    skate_vel_list = np.array(skate_vel_list)
    avg_error_list = np.array(avg_error_list)
    traj_fail_list = np.array(traj_fail_list)
    loc_fail_list = np.array(loc_fail_list)

    THRESH_LIST = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01]
    print(THRESH_LIST)

    print('skating_ratio_mean', skating_ratio_list.mean() * 100)
    print('skate_vel_mean', skate_vel_list.mean())
    print('avg_error_mean', avg_error_list.mean())
    print('traj_fail_mean', traj_fail_list.mean(0) * 100)
    print('loc_fail_mean', loc_fail_list.mean(0) * 100)


if __name__ == '__main__':
    # compile_hoi_metrics()
    # compute_self_contact_metrics()
    compute_embody_metrics()

