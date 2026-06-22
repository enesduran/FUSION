# Taken from https://github.com/facebookresearch/diffh2o
import numpy as np
from scipy.spatial.distance import pdist, squareform

class EvalNode:

    def __init__(self, obj_model_path, sbj_model_path, mano_model_path):

        self.sbj_model_path = sbj_model_path
        self.obj_model_path = obj_model_path
        self.mano_model_path = mano_model_path

        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.id_dict = {}
      



    def evaluate_seqs(self, samples, seq_names, seq_lens, samples_kf = None, eval_physics=False, eval_grasp_reference=False, num_reps=1, num_downsample_frames=15, fps_data=20):
        inter_volume = []
        inter_depth = []
        inter_depth_max = []
        contact_ratio = []
        jerk_pos = []
        jerk_ang = []
        acc_glob_pos_list = []
        acc_loc_pos_list = []
        acc_glob_rot_list = []
        acc_loc_rot_list = []
        t_vel_list = []
        num_contact_frames = []
        handedness_list = []
        grasp_error_list = []
        wrist_pos = []
        for seq, seq_name in enumerate(seq_names[:samples.shape[0]]):
            seq_len = min(seq_lens[0][seq],150)
           
            seq_id = self.id_dict[seq_name]
            sbj_id = seq_name.split('_')[0]
            obj_name = seq_name.split('_')[1]
            intent_name = seq_name.split('_')[2]
            gender = GENDER_MAP[sbj_id]
            sbj_vtemp_l = trimesh.load(os.path.join(self.sbj_model_path, gender, sbj_id + '_lhand.ply'),process=False)
            sbj_vtemp_r = trimesh.load(os.path.join(self.sbj_model_path, gender, sbj_id + '_rhand.ply'),process=False)

            frames_per_pose_input =  seq_len // num_downsample_frames
            downsampled_idcs = np.arange(0, seq_len, frames_per_pose_input)[:num_downsample_frames]
            feature_vec_full = samples[seq][0]

            feature_vec = feature_vec_full[1:] #remove pre-transition frame

            self.mano_model_l = smplx.create(self.mano_model_path, model_type="mano", use_pca=True,
                                v_template=sbj_vtemp_l.vertices, flat_hand_mean=True,is_rhand=False,
                                batch_size=num_downsample_frames, num_pca_comps=24).to(self.device)
            self.mano_model_r = smplx.create(self.mano_model_path, model_type="mano", use_pca=True,
                                        v_template=sbj_vtemp_r.vertices, flat_hand_mean=True,is_rhand=True,
                                        batch_size=num_downsample_frames, num_pca_comps=24).to(self.device)

            hand_l, hand_r, aa_l, aa_r = self.get_mano_obj_seq(feature_vec[downsampled_idcs], num_downsample_frames)


            wrist_pos.append(np.concatenate((hand_l.joints.detach().cpu().numpy(), hand_r.joints.detach().cpu().numpy()),axis=1))
            if samples_kf is not None and eval_grasp_reference:
                feature_vec_kf = samples_kf[seq][:1,:,-1].repeat(num_downsample_frames, 1)

                hand_l_kf, hand_r_kf, _, _ = self.get_mano_obj_seq(feature_vec_kf, num_downsample_frames)
            
            obj_mesh = trimesh.load(os.path.join(self.obj_model_path,obj_name+'.ply'), process=False)
            obj_verts_num = int(obj_mesh.vertices.shape[0] * 0.01)

            obj_mesh = obj_mesh.simplify_quadratic_decimation(obj_verts_num)
            trimesh.repair.fix_normals(obj_mesh)
            if '_m' == seq_name[-2:]:
                obj_mesh.vertices[...,0] *= -1
                obj_mesh.faces = obj_mesh.faces[:, ::-1]

            acc_glob_pos, acc_loc_pos, acc_glob_rot, acc_loc_rot = self.evaluate_accelerations(hand_l, hand_r, aa_l, aa_r)
            acc_glob_pos_list.append(acc_glob_pos.detach().cpu().numpy())
            acc_loc_pos_list.append(acc_loc_pos.detach().cpu().numpy())
            acc_glob_rot_list.append(acc_glob_rot.detach().cpu().numpy())
            acc_loc_rot_list.append(acc_loc_rot.detach().cpu().numpy())


            if eval_grasp_reference:
                t_vel = self.evaluate_tvel(feature_vec_full)
                handedness, grasp_error = self.evaluate_handedness_and_grasp_error(feature_vec, feature_vec_kf, hand_l, hand_r, hand_l_kf, hand_r_kf, obj_mesh)
                handedness_list.append(handedness)
                grasp_error_list.append(grasp_error.detach().cpu().numpy())
                t_vel_list.append(t_vel*fps_data)
                print('handedness accuracy', np.mean(handedness_list))
                print('grasp_error', np.mean(grasp_error_list))
                print('transition_velocity', np.mean(t_vel_list))
            if eval_physics:
            
                res_l, res_r = self.evaluate_physics(feature_vec, hand_l, hand_r, obj_mesh, num_downsample_frames)


                inter_volume.append(np.mean([res_l['inter_volume_mean']+res_r['inter_volume_mean']])*1e6)
                inter_depth.append(np.mean([res_l['inter_depth_mean']+res_r['inter_depth_mean']])*1e3)
                inter_depth_max.append(np.max([res_l['inter_depth_max'],res_r['inter_depth_mean']])*1e3)
                contact_ratio.append(np.mean([res_l['contact_ratio_contact'],res_r['contact_ratio_contact']]))
                jerk_pos.append(res_l['jerk_pos'])
                jerk_ang.append(res_l['jerk_ang'])

                print('inter volume', np.mean([res_l['inter_volume_mean']+res_r['inter_volume_mean']])*1e6)
                print('inter depth', np.mean([res_l['inter_depth_mean']+res_r['inter_depth_mean']])*1e3)
                print('inter depth max', np.max([res_l['inter_depth_max'],res_r['inter_depth_max']])*1e3)
                print('contact ratio', np.mean([res_l['contact_ratio_contact'],res_r['contact_ratio_contact']]))


            res_dict = {}

        od_metric = self.evaluate_overall_diversity(np.array(wrist_pos), downsampled_idcs.shape[0])

        res_dict['acc_glob_pos'] = np.array(acc_glob_pos_list)
        res_dict['acc_loc_pos'] = np.array(acc_loc_pos_list)
        res_dict['acc_glob_rot'] = np.array(acc_glob_rot_list)
        res_dict['acc_loc_rot'] = np.array(acc_loc_rot_list)
        res_dict['t_vels'] = np.array(t_vel_list)

        res_dict['overall_diversity'] = od_metric

        if num_reps > 1:
            res_dict['sample_diversity'] = od_metric
        if eval_grasp_reference:
            res_dict['handedness'] = np.array(handedness_list)
            res_dict['grasp_error'] = np.array(grasp_error_list)
        if eval_physics:
            res_dict['inter_volume'] = np.array(inter_volume)
            res_dict['inter_depth'] = np.array(inter_depth)
            res_dict['inter_depth_max'] = np.array(inter_depth_max)
            res_dict['contact_ratio'] = np.array(contact_ratio)
            res_dict['jerk_pos'] = np.array(jerk_pos)
            res_dict['jerk_ang'] = np.array(jerk_ang)
            res_dict['num_contact_frames'] = np.array(num_contact_frames)

            res_dict['inter_volume_mean'] = np.mean(inter_volume)
            res_dict['inter_depth_mean'] = np.mean(inter_depth)
            res_dict['contact_ratio_mean'] = np.mean(contact_ratio)
            res_dict['jerk_pos_mean'] = np.mean(jerk_pos)
            res_dict['jerk_ang_mean'] = np.mean(jerk_ang)
            res_dict['num_contact_frames_mean'] = np.mean(num_contact_frames)

        return res_dict

    def get_mano_obj_seq(self, feature_vec, seq_len_eval):

        # get the subject and object vertices over the whole sequence from the sample
        feature_vec = torch.tensor(feature_vec).to(self.device)

        pos_left = feature_vec[:,:3]
        pos_right = feature_vec[:,3:6]

        joint_rotations_l = feature_vec[:,6:30]
        joint_rotations_r = feature_vec[:,36:60]

        aa_l = feature_vec[:,6:30].clone().mm(self.mano_model_l.hand_components).reshape(-1,15,3)
        aa_r = feature_vec[:,36:60].clone().mm(self.mano_model_r.hand_components).reshape(-1,15,3)

        global_orient_l = matrix_to_axis_angle(rotation_6d_to_matrix(feature_vec[:,30:36]).reshape(-1,1,3,3))
        global_orient_r = matrix_to_axis_angle(rotation_6d_to_matrix(feature_vec[:,60:66]).reshape(-1,1,3,3))

        aa_l = torch.cat((global_orient_l, aa_l),axis=1)
        aa_r = torch.cat((global_orient_r, aa_r),axis=1)

        trans_l = pos_left -  self.mano_model_l(hand_pose=torch.zeros((seq_len_eval,24)).to(self.device)).joints[0,0]
        trans_r = pos_right - self.mano_model_r(hand_pose=torch.zeros((seq_len_eval,24)).to(self.device)).joints[0,0]

        hand_seq_l = self.mano_model_l(
            hand_pose=joint_rotations_l,
            global_orient=global_orient_l.squeeze(1),
            transl=trans_l,
        )

        hand_seq_r = self.mano_model_r(
            hand_pose=joint_rotations_r,
            global_orient=global_orient_r.squeeze(1),
            transl=trans_r,
        )

        return hand_seq_l, hand_seq_r, aa_l, aa_r

    def evaluate_overall_diversity(self, wrist_pos, num_downsampled_frames=15):


        pairwise_distances = []

        # Compute the pairwise distances
        pairwise_distances.append(
            np.mean(
                pdist(
                    wrist_pos.reshape(len(wrist_pos), -1),
                    metric="euclidean",
                )
            )
            / num_downsampled_frames
        )

        return np.mean(pairwise_distances)
    
    def evaluate_sample_diversity(self, wrist_pos, num_downsampled_frames=15):
        # Compute the pairwise distances
        pairwise_distance = np.mean(
                pdist(
                    wrist_pos.reshape(wrist_pos.shape[0], -1),
                    metric="euclidean",
                )
            ) / num_downsampled_frames


        return pairwise_distance
    
    def evaluate_handedness_and_grasp_error(self, feature_vec, feature_vec_kf, hand_l, hand_r, hand_l_kf, hand_r_kf, object_mesh, second_hand_thresh=0.05):

        feature_vec = torch.tensor(feature_vec).to(self.device)
        feature_vec_kf = torch.tensor(feature_vec).to(self.device)
        obj_rot = rotation_6d_to_matrix(feature_vec[:1,-6:]).reshape(-1,3,3)
        obj_rot_kf = rotation_6d_to_matrix(feature_vec_kf[:1,-6:]).reshape(-1,3,3)

        obj_verts = torch.matmul(torch.tensor(object_mesh.vertices, dtype=torch.float32).to(self.device),obj_rot)
        obj_verts +=  feature_vec[:1,np.newaxis,-9:-6]

        obj_verts_kf = torch.matmul(torch.tensor(object_mesh.vertices, dtype=torch.float32).to(self.device),obj_rot_kf)
        obj_verts_kf +=  feature_vec_kf[:1,np.newaxis,-9:-6]

        l_nearest = nearest_point(
            torch.tensor(hand_l.vertices[:1]), torch.tensor(obj_verts)
        )[0].min()
        r_nearest = nearest_point(
            torch.tensor(hand_r.vertices[:1]), torch.tensor(obj_verts)
        )[0].min()

        l_nearest_kf = nearest_point(
            torch.tensor(hand_l_kf.vertices[:1]), torch.tensor(obj_verts_kf)
        )[0].min()
        r_nearest_kf = nearest_point(
            torch.tensor(hand_r_kf.vertices[:1]), torch.tensor(obj_verts_kf)
        )[0].min()

        hand_correct = 0
        both_hands = False

        hand_idx = torch.argmin(torch.hstack([l_nearest_kf, r_nearest_kf]))
        hand_idx_pred = torch.argmin(torch.hstack([l_nearest, r_nearest]))
        second_hand_pred = (
            r_nearest
            if hand_idx_pred == 0
            else l_nearest
        )
        both_hands_pred = True if second_hand_pred < second_hand_thresh else False

        if hand_idx == 0:
            hand_joint_ids = np.arange(16)
            if r_nearest_kf < second_hand_thresh:
                hand_joint_ids = np.arange(32)
                both_hands = True
        else:
            hand_joint_ids = np.arange(16, 32)
            if l_nearest_kf < second_hand_thresh:
                hand_joint_ids = np.arange(32)
                both_hands = True

        if (both_hands and both_hands_pred) or (hand_idx_pred == hand_idx):
            hand_correct = 1
        
        hand_joints = torch.cat((hand_l.joints[0], hand_r.joints[0]), dim=0)
        hand_joints_kf = torch.cat((hand_l_kf.joints[0], hand_r_kf.joints[0]), dim=0)


        grasp_error = torch.mean(
                    torch.norm(
                        hand_joints[hand_joint_ids]
                        - hand_joints_kf[hand_joint_ids], dim=-1),
                        axis=-1,
                    )

        return hand_correct, grasp_error


    def evaluate_accelerations(self, hand_l, hand_r, aa_l, aa_r):

        seq_joints = []

        wrist_pos_l = hand_l.joints[:,:1]
        wrist_pos_r = hand_r.joints[:,:1]

        joint_pos_l = hand_l.joints[:,1:]
        joint_pos_r = hand_r.joints[:,1:]

        wrist_rot_l = aa_l[:,:1]
        wrist_rot_r = aa_r[:,:1]

        joint_rot_l = aa_l[:,1:]
        joint_rot_r = aa_r[:,1:]

        wrist_pos = torch.cat((wrist_pos_l,wrist_pos_r), axis=1)
        joints_relative = torch.cat(((joint_pos_l - wrist_pos_l),(joint_pos_r - wrist_pos_r)), axis=1)

        wrist_rot = torch.cat((wrist_rot_l,wrist_rot_r), axis=1)
        joint_rot = torch.cat((joint_rot_l,joint_rot_r), axis=1)
        # Compute velocity by taking the first derivative of the position
        velocity_glob = torch.gradient(wrist_pos, axis=0)[0]
        velocity_loc = torch.gradient(joints_relative, axis=0)[0]
        velocity_glob_rot = torch.gradient(wrist_rot, axis=0)[0]
        velocity_loc_rot = torch.gradient(joint_rot, axis=0)[0]
        # Compute acceleration by taking the first derivative of the velocity
        acc_glob = torch.mean(torch.abs(torch.gradient(velocity_glob, axis=0)[0]))
        acc_loc = torch.mean(torch.abs(torch.gradient(velocity_loc, axis=0)[0]))
        acc_glob_rot = torch.mean(torch.abs(torch.gradient(velocity_glob_rot, axis=0)[0]))
        acc_loc_rot = torch.mean(torch.abs(torch.gradient(velocity_loc_rot, axis=0)[0]))


        return  acc_glob, acc_loc, acc_glob_rot, acc_loc_rot

    def evaluate_tvel(self, feature_vec):
        t_vel = np.linalg.norm(feature_vec[1,:6]-feature_vec[0,:6])

        return t_vel

    def evaluate_physics(self, feature_vec, hand_seq_l, hand_seq_r, obj_mesh, downsampled_seq_len):

        # get the subject and object vertices over the whole sequence from the sample
        obj_verts = obj_mesh.vertices

        feature_vec = torch.tensor(feature_vec).to(self.device)
        obj_rot = rotation_6d_to_matrix(feature_vec[:,-6:]).reshape(-1,3,3)
        obj_verts = torch.matmul(torch.tensor(obj_verts, dtype=torch.float32).to(self.device),obj_rot)
        obj_verts +=  feature_vec[:,np.newaxis,-9:-6]

        obj_rot_vec = R.from_matrix(obj_rot.detach().cpu().numpy()).as_rotvec()
        obj_poses = np.concatenate((feature_vec[:,-9:-6].detach().cpu().numpy(),obj_rot_vec),axis=-1)

        res_dict_l = compute_all_metrics(hand_seq_l.vertices.to(self.device), torch.tensor(self.mano_model_l.faces.astype(np.int32)).to(self.device), torch.tensor(obj_verts).to(self.device),
        torch.tensor(obj_mesh.faces.astype(np.int32)).to(self.device), torch.tensor(obj_poses).to(self.device), 0, downsampled_seq_len, device=self.device)
        res_dict_r = compute_all_metrics(hand_seq_r.vertices.to(self.device), torch.tensor(self.mano_model_r.faces.astype(np.int32)).to(self.device), torch.tensor(obj_verts).to(self.device),
        torch.tensor(obj_mesh.faces.astype(np.int32)).to(self.device), torch.tensor(obj_poses).to(self.device),
        0, downsampled_seq_len, device=self.device)


        return res_dict_l, res_dict_r


    def get_start_end_idcs(self, hand_seq, obj_poses, thresh=0.2):
        hand_obj_dist = np.linalg.norm(hand_seq.joints[:,0].cpu().detach().numpy() - obj_poses[:,:3],axis=-1)
        close_frames = np.where(hand_obj_dist < thresh)[0]
        if close_frames.shape[0] > 0:
            start_idx = close_frames[0]
            end_idx = close_frames[-1]
        else:
            start_idx = -1
            end_idx = -1

        return start_idx, end_idx
