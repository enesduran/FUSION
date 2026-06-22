import torch
import joblib
  
def get_obstacles():
    # Obstacles for obstacle avoidance task. Each one is a circle with radius
    # on the xz plane with center at (x, z)
    obs_list = [((-0.5, 3.5) , 0.7), ((0.2, 1.5) , 0.8)]
 
    return obs_list


def prepare_task(task_info):
    """At this point, we need to have (1) target, (2) target_mask_dict, (3) kframes, 
    (4) whether to start from noise (is_noise_init), 
    (5, optional) initial motion, (6) obs_list
    If is_noise_init is True, the initial motion will not be used.
    return target, target_mask_dict, kframes, is_noise_init, init_motion, obs_list
    """

    bs = task_info['initial_motion'].shape[0]
    J_num = task_info['initial_motion'].shape[2]
    gen_frames = task_info['gen_frames']

    # Prepare empty target and empty mask based on the desire length.
    target_opt_dict = {'joints': torch.zeros([bs, gen_frames, J_num, 3], device=task_info["device"]),
                       'vertices': torch.zeros([bs, gen_frames, 10475, 3], device=task_info["device"])} 
    
    target_mask_dict = {'joints': torch.zeros_like(target_opt_dict['joints'], dtype=torch.bool),
                        'vertices': torch.zeros_like(target_opt_dict['vertices'], dtype=torch.bool)} 

    taskname = task_info["task"]
   
    if taskname in ["keypoint_tracking", "object_guided_optimization", "clutch_optimization"]:
        return task_dense_optimization(task_info, target_opt_dict, target_mask_dict)
    elif taskname == "self_contact_optimization":
        return task_self_contact(task_info, target_opt_dict, target_mask_dict)
    elif taskname == "motion_projection":
        return task_motion_projection(task_info, target_opt_dict, target_mask_dict)
    elif taskname == "motion_blending":
        return task_motion_blending(task_info, target_opt_dict, target_mask_dict)
    elif taskname == "motion_inbetween":
        return task_motion_inbetweening(task_info, target_opt_dict, target_mask_dict)
 
    else:
        raise ValueError(f"Unknown task name: {taskname}")
 

def task_trajectory_editing(task_info, args, target_opt_dict, target_mask_dict):
    """ Trajectory Editing task. The goal is, given an original motion and the editing target,
    we want to optimize the noise to be the one that can be mapped to the motions that satisfy
    the editing target.
    """
    # Get obstacle list
    if "use_obstacles" in args:
        obs_list = get_obstacles()
    else:
        obs_list = []
    
    # selected_index = [62, 90, 110]  # [0] # 
    selected_index = [90]  # [0] # 
    # target_locations = [(0.5, 0.5), (1., 1.), (1.5, 1.5)] #  [(0,0)] # 
    target_locations = [(1.5, 1.5)]

    # Set up the new target based on the selected frames and the target locations
    kframes = [
        (tt, locs) for (tt, locs) in zip(selected_index, target_locations)
    ]
    for tt, locs in zip(selected_index, target_locations):
        print("target at %d = %.1f, %.1f" % (tt, locs[0], locs[1]))
        target_opt_dict[0, tt, 0, [0, 2]] = torch.tensor(
            [locs[0], locs[1]], dtype=torch.float32, device=target_opt_dict.device
        )
        target_mask_dict[0, tt, 0, [0, 2]] = True
    
    is_noise_init = False
    return target_opt_dict, target_mask_dict, kframes, is_noise_init, obs_list


def task_pose_editing(task_info, args, target_opt_dict, target_mask_dict):
    ''' This is a more general version of the trajectory editing task where each joint can be modified.
    The core idea is the same.
    '''
    # List for editing
    # (joint_index, keyframe, edit_dim target(x, y, z))
    # edit_dim: list of dimensions to edit [0, 1, 2] for x, y, z respectively
    # We can edit only some dimensions of the target pose e.g. only y (height of the joint)
    # joint_idx = 21 # Right hand
    
    target_edit_list = [
        # (joint_index, keyframe, edit_dim, target(x, y, z))
        # (21, 90, [1], [1.0]), # Right hand at frame 90th, edit height to 1.0
        (15, 90, [1], [0.6]), # Head at frame 90th, edit height to 1.0

    ]
    kframes = []
    obs_list = []
    for (joint_index, keyframe, edit_dim, target_loc) in target_edit_list:
        target_opt_dict[0, keyframe, joint_index, edit_dim] = torch.tensor(
            target_loc, 
            dtype=torch.float32, device=target_opt_dict.device
        )
        target_mask_dict[0, keyframe, joint_index, edit_dim] = True
        # kframes.append((keyframe, (0, 0)))


    is_noise_init = False
    return target_opt_dict, target_mask_dict, kframes, is_noise_init, obs_list


def task_dense_optimization(task_info, target_opt_dict, target_mask_dict):
    """Dense optimization. This task is only for testing if noise optimization can reconstruct an arbitrary motion.
    The idea is, starting from random noise, we want to steer it to generate a specific motion.
    This task is an alternative way to obtain the corresponding noise for a given motion without DDIM inversion.
    It is also useful for debugging and providing a better idea of what the motion distribution landscape looks like.
    """
    is_noise_init = True
    keyframes = []
    obs_list = []
    # Target is the generated motion
    init_motion_len = task_info["initial_motion"].shape[1]
    
    target_opt_dict['joints'][:, :init_motion_len, :, :] = task_info["initial_motion"].to(target_opt_dict['joints'].device)
    target_mask_dict['joints'][:, :init_motion_len, -10:, :] = True
    
    return target_opt_dict, target_mask_dict, keyframes, is_noise_init, obs_list

def task_self_contact(task_info, target_opt_dict, target_mask_dict, t_pose_flag=False):
    """Dense optimization. This task is only for testing if noise optimization can reconstruct an arbitrary motion.
    The idea is, starting from T pose, we want to steer it to generate a specific motion through llms."""

    from src.utils.llm_utils import SELF_INTERACTION_META_DICT
    from src.llm.gpt_self_interaction import generate_self_interaction_commands

    INSTRUCTION_DICT_LIST = []

    for text_id in task_info['text_id']:

        text = SELF_INTERACTION_META_DICT[text_id]
        print('Text: ', text)  
      
        try:   
            # check if the text description is already in the interaction dict
            if text in task_info['interaction_dict'].keys():
                INSTRUCTION_DICT = task_info['interaction_dict'][text]
                INSTRUCTION_DICT_LIST.append(INSTRUCTION_DICT)
            else:
                INSTRUCTION_DICT = generate_self_interaction_commands(task_info['client'], text)
                task_info['interaction_dict'][text] = INSTRUCTION_DICT
                print(f'Interaction dict updated: {text}')
                INSTRUCTION_DICT_LIST.append(INSTRUCTION_DICT)

        except:
            print(f'PROBLEM WITH {text}')
     
    # Set the initial and end motion to be t_pose
    if t_pose_flag:
        target_mask_dict['joints'][:, [0, -1]] = True
        target_opt_dict['joints'][:, [0, -1], :55] = task_info['t_pose'].to(target_opt_dict['joints'].device)
    
    return target_opt_dict, target_mask_dict, None, True, INSTRUCTION_DICT_LIST
 

def task_motion_projection(task_info, args, target, target_mask_dict):
    """Motion projection (same as motion denoising). Given a set of noisy joints, we want to reconstruct
    a valid motion that is as close as possible to the noisy input.
    Start from random noise and use the noisy input as target. Functionally, this is the same as 
    the dense optimization task about.
    """
    is_noise_init = True
    kframes = []
    obs_list = []
    # Target is the generated motion
    init_motion_len = task_info["initial_motion"].shape[0]
    target[0, :init_motion_len, :, :] = torch.from_numpy(
        task_info["initial_motion"]).to(target.device)
    target_mask_dict[0, :init_motion_len, :, :] = True

    ###### Add noise to target  ######
    # To ensure that there is a valid solution that can be reconstructed from the given noise,
    # we construct the noisy target by adding noise the given motion.
    noise_level = 0.03  # 0.01
    target = target + (torch.randn_like(target)) * noise_level

    return target, target_mask_dict, kframes, is_noise_init, obs_list


def task_motion_inbetweening(task_info, target, target_mask_dict):
    """Motion in-betweening. Select two frames from a given motion to be use a starting frame and
    ending frame. Then, infill the motion in-between these two poses.
    """


    is_noise_init = True
    keyframes = []
    obs_list = []

    init_motion_len = task_info["initial_motion"].shape[1]
  
    target[:, :init_motion_len, :, :] = task_info["initial_motion"].to(target.device)

    # Select two frames to be used as starting and ending frames.
    start_frame = 0
    target_frame = -1
  
    target_mask_dict[:, [start_frame, target_frame], :, :] = True

 
    return target, target_mask_dict, keyframes, is_noise_init, obs_list
 


def task_motion_blending(task_info, args, target, target_mask_dict):
    """Motion blending. Concat two initial motions together. 
    To create target, combine the motion in the representation space such that
    when we concat it, the second motion will start where the first motion ends.
    """
    # is_noise_init = False
    is_noise_init = True
    kframes = []
    obs_list = []

    # No target around the seam
    SEAM_WIDTH = 10 # 15 # 10 # 5 # 3

    # combined_motions[0] shape [1, 22, 3, 196]
    target[0] = task_info["combine_motion"]
    target_mask_dict = torch.ones_like(target, dtype=torch.bool)
    target_mask_dict[0, :, :, :] = True
    target_mask_dict[0, args.gen_frames // 2 - SEAM_WIDTH: args.gen_frames // 2 + SEAM_WIDTH] = False

    return target, target_mask_dict, kframes, is_noise_init, obs_list


 