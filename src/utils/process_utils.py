import numpy as np
import matplotlib.pyplot as plt

from pygit2 import Repository
BRANCH_NAME = Repository('.').head.shorthand  

SMPLX_JOINTS = {"pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3,
    "left_knee": 4, "right_knee": 5, "spine2": 6, "left_ankle": 7, "right_ankle": 8,
    "spine3": 9, "left_foot": 10, "right_foot": 11, "neck": 12, "left_collar": 13,
    "right_collar": 14, "head": 15, "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19, "left_wrist": 20, "right_wrist": 21,
    "jaw": 22, "left_eye_smplhf": 23, "right_eye_smplhf": 24, "left_index1": 25,
    "left_index2": 26, "left_index3": 27, "left_middle1": 28, "left_middle2": 29,
    "left_middle3": 30, "left_pinky1": 31, "left_pinky2": 32, "left_pinky3": 33,
    "left_ring1": 34, "left_ring2": 35, "left_ring3": 36, "left_thumb1": 37, 
    "left_thumb2": 38, "left_thumb3": 39, "right_index1": 40, "right_index2": 41, 
    "right_index3": 42, "right_middle1": 43, "right_middle2": 44, "right_middle3": 45,
    "right_pinky1": 46, "right_pinky2": 47, "right_pinky3": 48, "right_ring1": 49,
    "right_ring2": 50, "right_ring3": 51, "right_thumb1": 52, "right_thumb2": 53, 
    "right_thumb3": 54}
    
SMPL_JOINTS = {"pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3, "left_knee": 4, 
               "right_knee": 5, "spine2": 6, "left_ankle": 7, "right_ankle": 8, 
               "spine3": 9, "left_foot": 10, "right_foot": 11, "neck": 12, "left_collar": 13, 
               "right_collar": 14, "head": 15, "left_shoulder": 16, "right_shoulder": 17, 
               "left_elbow": 18, "right_elbow": 19, "left_wrist": 20, "right_wrist": 21, 
               "left_hand": 22, "right_hand": 23}
 
VIZ_PLOTS = False
FLOOR_VEL_THRESH = 0.005
ROOT_HEIGHT_THRESH = 0.10 # 0.04
CLUSTER_SIZE_THRESH = 1
CONTACT_VEL_THRESH = 0.010 #0.005 
FLOOR_HEIGHT_OFFSET = 0.01
TERRAIN_HEIGHT_THRESH = 0.08 # 0.04
CONTACT_TOE_HEIGHT_THRESH = 0.08
DISCARD_TERRAIN_SEQUENCES = True 
CONTACT_ANKLE_HEIGHT_THRESH = 0.12 # 0.08

RIGHT_WRIST_BASE_LOC = np.array([[0.0957, 0.0064, 0.0062]])
LEFT_WRIST_BASE_LOC = np.array([[-0.0957, 0.0064, 0.0062]])
  
CONTACT_INDICES = np.array([SMPLX_JOINTS['left_ankle'], SMPLX_JOINTS['right_ankle'],
                            SMPLX_JOINTS['left_foot'], SMPLX_JOINTS['right_foot']])

GLOBAL_LEFT_LEG_INDICES = np.array([SMPLX_JOINTS['left_hip'], SMPLX_JOINTS['left_knee'],
                       SMPLX_JOINTS['left_ankle'], SMPLX_JOINTS['left_foot']])

GLOBAL_RIGHT_LEG_INDICES = np.array([SMPLX_JOINTS['right_hip'], SMPLX_JOINTS['right_knee'],
                       SMPLX_JOINTS['right_ankle'], SMPLX_JOINTS['right_foot']])

GLOBAL_FOOT_INDICES = np.concatenate([GLOBAL_LEFT_LEG_INDICES, GLOBAL_RIGHT_LEG_INDICES])

GLOBAL_BODY_INDICES = np.array([SMPLX_JOINTS["spine1"], SMPLX_JOINTS["spine2"],
                                SMPLX_JOINTS["spine3"], SMPLX_JOINTS["neck"],
                                SMPLX_JOINTS["left_collar"], SMPLX_JOINTS["right_collar"],
                                SMPLX_JOINTS["head"], SMPLX_JOINTS["left_shoulder"],
                                SMPLX_JOINTS["right_shoulder"], SMPLX_JOINTS["jaw"], 
                                SMPLX_JOINTS["left_eye_smplhf"], SMPLX_JOINTS["right_eye_smplhf"]])

GLOBAL_WRIST_INDICES =  np.array([SMPLX_JOINTS['left_wrist'], SMPLX_JOINTS['right_wrist']])
  
GLOBAL_RHAND_INDICES = np.array([SMPLX_JOINTS['right_wrist'], SMPLX_JOINTS['right_index1'], 
                                 SMPLX_JOINTS['right_index2'], SMPLX_JOINTS['right_index3'], 
                                 SMPLX_JOINTS['right_middle1'], SMPLX_JOINTS['right_middle2'], 
                                 SMPLX_JOINTS['right_middle3'], SMPLX_JOINTS['right_pinky1'], 
                                 SMPLX_JOINTS['right_pinky2'], SMPLX_JOINTS['right_pinky3'], 
                                 SMPLX_JOINTS['right_ring1'], SMPLX_JOINTS['right_ring2'], 
                                 SMPLX_JOINTS['right_ring3'], SMPLX_JOINTS['right_thumb1'], 
                                 SMPLX_JOINTS['right_thumb2'], SMPLX_JOINTS['right_thumb3']])

GLOBAL_LHAND_INDICES = np.array([SMPLX_JOINTS['left_wrist'], SMPLX_JOINTS['left_index1'], 
                                 SMPLX_JOINTS['left_index2'], SMPLX_JOINTS['left_index3'], 
                                 SMPLX_JOINTS['left_middle1'], SMPLX_JOINTS['left_middle2'], 
                                 SMPLX_JOINTS['left_middle3'], SMPLX_JOINTS['left_pinky1'], 
                                 SMPLX_JOINTS['left_pinky2'], SMPLX_JOINTS['left_pinky3'], 
                                 SMPLX_JOINTS['left_ring1'], SMPLX_JOINTS['left_ring2'], 
                                 SMPLX_JOINTS['left_ring3'], SMPLX_JOINTS['left_thumb1'], 
                                 SMPLX_JOINTS['left_thumb2'], SMPLX_JOINTS['left_thumb3']])

GLOBAL_RHAND_TIPS_INDICES = np.concatenate([GLOBAL_RHAND_INDICES, 
                                            [55, 56, 57, 58, 59]])
GLOBAL_LHAND_TIPS_INDICES = np.concatenate([GLOBAL_LHAND_INDICES, 
                                            [60, 61, 62, 63, 64]])
GLOBAL_HAND_TIPS_INDICES = np.concatenate([GLOBAL_RHAND_TIPS_INDICES, GLOBAL_LHAND_TIPS_INDICES])

GLOBAL_HAND_INDICES = np.concatenate([GLOBAL_RHAND_INDICES, GLOBAL_LHAND_INDICES])

GLOBAL_RHAND_ARM_INDICES = np.concatenate([GLOBAL_RHAND_INDICES, 
                                          np.array([SMPLX_JOINTS['right_shoulder'], SMPLX_JOINTS['right_elbow']])])
GLOBAL_LHAND_ARM_INDICES = np.concatenate([GLOBAL_LHAND_INDICES, 
                                          np.array([SMPLX_JOINTS['left_shoulder'], SMPLX_JOINTS['left_elbow']])])

GLOBAL_HAND_ARM_INDICES = np.concatenate([GLOBAL_RHAND_ARM_INDICES, GLOBAL_LHAND_ARM_INDICES])
 
 
FINGERTIPS_INDICES = np.array([SMPLX_JOINTS['left_index3'], SMPLX_JOINTS['left_middle3'],
                       SMPLX_JOINTS['left_pinky3'], SMPLX_JOINTS['left_ring3'], 
                       SMPLX_JOINTS['left_thumb3'], SMPLX_JOINTS['right_index3'], 
                       SMPLX_JOINTS['right_middle3'], SMPLX_JOINTS['right_ring3'], 
                       SMPLX_JOINTS['right_pinky3'], SMPLX_JOINTS['right_thumb3']])
 
     
SMPLX_FINGERTIP_INDICES = {'nose': 9120,
                    'reye':	9929,
                    'leye':	9448,
                    'rear':	616,
                    'lear':	6,
                    'rthumb':	8079,
                    'rindex':	7669,
                    'rmiddle':	7794,
                    'rring':	7905,
                    'rpinky':	8022,
                    'lthumb':	5361,
                    'lindex':	4933,
                    'lmiddle':	5058,
                    'lring':	5169,
                    'lpinky':	5286,
                    'LBigToe':	5770,
                    'LSmallToe': 5780,
                    'LHeel':	8846,
                    'RBigToe':	8463,
                    'RSmallToe': 8474,
                    'RHeel': 8635}


SELF_INTERACTION_VERTICES = {'left_earlobe': 557,
                             'left_ear_mid': 861, 
                             'left_ear_top': 1488,
                             'right_earlobe': 40,
                             'right_ear_mid': 179,
                             'right_ear_top': 1519,
                             'nose': 9120, 
                             'chin': 8752,
                             'left_shoulder': 4439,
                             'right_shoulder': 7178, 
                             'left_kneecap': 3676,
                             'right_kneecap': 6437,
                             'belly': 5939, 
                             'torso': 5532,
                             'left_haunch': 3865,
                             'right_haunch': 6839, 
                             'neck': 2151,
                             'left_heel': 8846,
                             'right_heel': 8635, 
                             'right_eye': 9929,
                             'left_eye': 9448,
                             'right_thumb':	8079,
                             'right_index':	7669,
                             'right_middle': 7794,
                             'right_ring': 7905,
                             'right_pinky': 8022,
                             'left_thumb': 5361,
                             'left_index': 4933,
                             'left_middle': 5058,
                             'left_ring': 5169,
                             'left_pinky': 5286,
                             'left_big_toe': 5770,
                             'left_small_toe': 5780,
                             'right_big_toe': 8463,
                             'right_small_toe': 8474}


MANO_TIP_IDS = {'thumb': 744,
                'index': 320,
                'middle': 443,
                'ring':	554,
                'pinky': 671}


SMPLX_FINGERTIPS = np.array([SMPLX_FINGERTIP_INDICES['rthumb'], SMPLX_FINGERTIP_INDICES['rindex'], 
                    SMPLX_FINGERTIP_INDICES['rmiddle'], SMPLX_FINGERTIP_INDICES['rring'], 
                    SMPLX_FINGERTIP_INDICES['rpinky'], SMPLX_FINGERTIP_INDICES['lthumb'], 
                    SMPLX_FINGERTIP_INDICES['lindex'], SMPLX_FINGERTIP_INDICES['lmiddle'],
                    SMPLX_FINGERTIP_INDICES['lring'], SMPLX_FINGERTIP_INDICES['lpinky']])
    

def mask_grad(grad, mask):
    return grad * mask


def determine_floor_height_and_contacts(body_joint_seq, fps, for_smplx=True, z_ax_number=2):
    '''
    Input: body_joint_seq N x J x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    ---
    z_ax_number: int, index of the axis representing the vertical direction (default is 2 for z-axis)
    '''

    # lazy import
    from sklearn.cluster import DBSCAN
    
    num_frames = body_joint_seq.shape[0]

    JOINT_DICT = SMPLX_JOINTS if for_smplx else SMPL_JOINTS
        
    # compute toe velocities
    root_seq = body_joint_seq[:, JOINT_DICT['pelvis'], :]
    left_toe_seq = body_joint_seq[:, JOINT_DICT['left_foot'], :]
    right_toe_seq = body_joint_seq[:, JOINT_DICT['right_foot'], :]

    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    
    
    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(num_frames)
        plt.plot(steps, left_toe_vel, '-r', label='left vel')
        plt.plot(steps, right_toe_vel, '-b', label='right vel')
        plt.legend()
        plt.savefig('toe_velocities.png')
        plt.close()

    
    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, z_ax_number]
    right_toe_heights = right_toe_seq[:, z_ax_number]
    root_heights = root_seq[:, z_ax_number]

    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(num_frames)
        plt.plot(steps, left_toe_heights, '-r', label='left toe height')
        plt.plot(steps, right_toe_heights, '-b', label='right toe height')
        plt.plot(steps, root_heights, '-g', label='root height')
        plt.legend()
        plt.savefig("toe_root_heights.png")
        plt.close()
        
        
    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)

    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(left_static_foot_heights.shape[0])
        plt.plot(steps, left_static_foot_heights, '-r', label='left static height')
    
        plt.legend()
        plt.savefig("left_right_static_heights.png")
        plt.close()
  
    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        
        cluster_heights, cluster_root_heights, cluster_sizes = [], [], []
        
        
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.01, min_samples=30).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)

        if VIZ_PLOTS:
            plt.figure()
            
        min_median = min_root_median = float('inf')

        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
            
            if VIZ_PLOTS:
                plt.scatter(cur_clust, np.zeros_like(cur_clust), label='foot %d' % (cur_label))
           
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])


            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)
            
            if VIZ_PLOTS:
                plt.scatter(cur_root_clust, np.zeros_like(cur_root_clust), label='root %d' % (cur_label))

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        if VIZ_PLOTS:
            plt.savefig('clustered_heights.png')
            plt.close()

        floor_height = min_median 
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit

        if DISCARD_TERRAIN_SEQUENCES:
      
            for cluster_root_height, cluster_height, cluster_size in zip (cluster_root_heights, cluster_heights, cluster_sizes):
                
                root_above_thresh = cluster_root_height > (min_root_median + ROOT_HEIGHT_THRESH)
                toe_above_thresh = cluster_height > (min_median + TERRAIN_HEIGHT_THRESH)
                cluster_size_above_thresh = cluster_size > int(CLUSTER_SIZE_THRESH*fps)
                          
                if root_above_thresh and toe_above_thresh and cluster_size_above_thresh:
                    discard_seq = True
                    break
    else:
        floor_height = offset_floor_height = 0.0
    
    # now find contacts (feet are below certain velocity and within certain range of floor)
    # compute heel velocities
    left_heel_seq = body_joint_seq[:, JOINT_DICT['left_ankle'], :]
    right_heel_seq = body_joint_seq[:, JOINT_DICT['right_ankle'], :]
    left_heel_vel = np.linalg.norm(left_heel_seq[1:] - left_heel_seq[:-1], axis=1)
    left_heel_vel = np.append(left_heel_vel, left_heel_vel[-1])
    right_heel_vel = np.linalg.norm(right_heel_seq[1:] - right_heel_seq[:-1], axis=1)
    right_heel_vel = np.append(right_heel_vel, right_heel_vel[-1])

    left_heel_contact = left_heel_vel < CONTACT_VEL_THRESH
    right_heel_contact = right_heel_vel < CONTACT_VEL_THRESH
    left_toe_contact = left_toe_vel < CONTACT_VEL_THRESH
    right_toe_contact = right_toe_vel < CONTACT_VEL_THRESH

    # compute heel heights
    left_heel_heights = left_heel_seq[:, z_ax_number] - floor_height
    right_heel_heights = right_heel_seq[:, z_ax_number] - floor_height
    left_toe_heights =  left_toe_heights - floor_height
    right_toe_heights =  right_toe_heights - floor_height

    left_heel_contact = np.logical_and(left_heel_contact, left_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    right_heel_contact = np.logical_and(right_heel_contact, right_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    left_toe_contact = np.logical_and(left_toe_contact, left_toe_heights < CONTACT_TOE_HEIGHT_THRESH)
    right_toe_contact = np.logical_and(right_toe_contact, right_toe_heights < CONTACT_TOE_HEIGHT_THRESH)

    contacts = np.zeros((num_frames, len(JOINT_DICT)))
    
    contacts[:, JOINT_DICT['left_ankle']] = left_heel_contact
    contacts[:, JOINT_DICT['left_foot']] = left_toe_contact
    contacts[:, JOINT_DICT['right_ankle']] = right_heel_contact
    contacts[:, JOINT_DICT['right_foot']] = right_toe_contact

    # hand contacts
    left_hand_contact = detect_joint_contact(body_joint_seq, 'left_wrist', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_hand_contact = detect_joint_contact(body_joint_seq, 'right_wrist', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:, JOINT_DICT['left_wrist']] = left_hand_contact
    contacts[:, JOINT_DICT['right_wrist']] = right_hand_contact

    # knee contacts
    left_knee_contact = detect_joint_contact(body_joint_seq, 'left_knee', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_knee_contact = detect_joint_contact(body_joint_seq, 'right_knee', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:, JOINT_DICT['left_knee']] = left_knee_contact
    contacts[:, JOINT_DICT['right_knee']] = right_knee_contact

    return offset_floor_height, contacts, discard_seq

def detect_joint_contact(body_joint_seq, joint_name, floor_height, vel_thresh, height_thresh):
    # calc velocity
    joint_seq = body_joint_seq[:, SMPLX_JOINTS[joint_name], :]
    joint_vel = np.linalg.norm(joint_seq[1:] - joint_seq[:-1], axis=1)
    joint_vel = np.append(joint_vel, joint_vel[-1])
    # determine contact by velocity
    joint_contact = joint_vel < vel_thresh
    # compute heights
    joint_heights = joint_seq[:, 2] - floor_height
    # compute contact by vel + height
    joint_contact = np.logical_and(joint_contact, joint_heights < height_thresh)

    return joint_contact