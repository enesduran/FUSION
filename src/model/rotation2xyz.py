import torch
import smplx 
import trimesh

from utils.transforms import quat_fk
from src.utils.viz_utils import pack_to_render
from utils.transforms3d import transform_body_pose
from src.utils.process_utils import SMPLX_FINGERTIPS, SMPLX_JOINTS


class Rotation2xyz:
    def __init__(self, device, smplx_path, num_betas=300):
        
        self.device = device
        self.num_betas = num_betas
        
        self.bm_dict = {'male' : smplx.create(f'{smplx_path}/smplx/SMPLX_MALE.npz',
                                           model_type='smplx',
                                           num_betas=self.num_betas,
                                           gender='male',
                                           flat_hand_mean=True,
                                           use_pca=False).to(device), 
                                           
                        'female' : smplx.create(f'{smplx_path}/smplx/SMPLX_FEMALE.npz',
                                           model_type='smplx',
                                           num_betas=self.num_betas,
                                           gender='female',
                                           flat_hand_mean=True,
                                           use_pca=False).to(device), 
                        
                        'neutral' : smplx.create(f'{smplx_path}/smplx/SMPLX_NEUTRAL.npz',
                                           model_type='smplx',
                                           num_betas=self.num_betas,
                                           gender='neutral',
                                           flat_hand_mean=True,
                                           use_pca=False).to(device)}
        

        self.hand_dict = {'right':  smplx.create(f'{smplx_path}/mano/MANO_RIGHT.pkl',
                                           model_type='mano',
                                           num_betas=self.num_betas,
                                           gender='neutral',
                                           is_rhand=True,
                                           flat_hand_mean=True,
                                           use_pca=False).to(device), 

                          'left':   smplx.create(f'{smplx_path}/mano/MANO_LEFT.pkl',
                                            model_type='mano',
                                            num_betas=self.num_betas,
                                            gender='neutral',
                                            is_rhand=False,
                                            flat_hand_mean=True,
                                            use_pca=False).to(device)}

        self.default_vtemp_dict = {'male' : self.bm_dict['male'].v_template,
                                   'female' : self.bm_dict['female'].v_template,
                                   'neutral' : self.bm_dict['neutral'].v_template}
    
        self.TIPS_IDX = torch.tensor(SMPLX_FINGERTIPS)

        # same with forward method
        self.default_joints = {k: v.J_regressor @ v.v_template for k, v in self.bm_dict.items()}


    def __call__(self, 
                 motion_dict, 
                 betas=None, 
                 cpu_flag=False,
                 body_vtemp=None,
                 add_fingertips=False, 
                 gender_list=[],
                 ):
        
        
        motion = motion_dict['full_motion_unnorm']
        nsamples, time, _ = motion.shape
        
        motion = motion.reshape((nsamples * time, -1))
        
        
        smplx_params = pack_to_render(trans=motion[..., :3].cpu(), 
                                            rots=motion[..., 3:].cpu(),
                                            pose_repr='6d',
                                            device=self.device)
        
        return self.forward_body(smplx_params, 
                 betas, 
                 cpu_flag,
                 body_vtemp,
                 add_fingertips,
                 time=time,
                 gender_list=gender_list)
    
        
    def forward_joints(self, root_offset, pos_offset):
        
        global_joints = [root_offset]
 

        for child, parent in enumerate(self.bm_dict['neutral'].parents.cpu().numpy()):
            
            if parent == -1:
                continue         
                
            global_joints.append(pos_offset[:, child] + global_joints[parent])
    
        return torch.stack(global_joints, dim=1)


    def forward_hand(self, motion_dict, pos_offset, cpu_flag=False, **kwargs):


        hand_dict = {'right': None, 'left': None}
        hand_motion_dict = {'right': {}, 'left': {}}
        

        if 'full_motion_unnorm' in motion_dict.keys():

            motion = motion_dict['full_motion_unnorm']
            nsamples, time = motion.shape[:2]

            n_samples_time = nsamples * time
 
            lpos = torch.repeat_interleave(pos_offset[:, None], 
                                       repeats=motion.shape[1], dim=1).to(self.device)
            
            

            target_gr, target_xyz = quat_fk(lpos=lpos, 
                                    lrot=transform_body_pose(motion[:, :, 3:], '6d->quat'), 
                                    parents=self.bm_dict['neutral'].parents)
            
            target_gr = transform_body_pose(target_gr, 'quat->aa')

            target_xyz += motion[..., :3][:, :, None]
    
          
            hm_params = pack_to_render(trans=motion[..., :3].cpu(), 
                                            rots=motion[..., 3:].cpu(),
                                            pose_repr='6d',
                                            device=self.device)
   
            hand_motion_dict['right']['hand_pose'] = hm_params['right_hand_pose'].reshape(n_samples_time, -1)
            hand_motion_dict['right']['global_orient'] = target_gr[:, :, SMPLX_JOINTS['right_wrist']].reshape(n_samples_time, -1)
            hand_motion_dict['right']['transl'] = target_xyz[:, :, SMPLX_JOINTS['right_wrist']].reshape(n_samples_time, -1)
            hand_motion_dict['right']['betas'] = torch.zeros((n_samples_time, 10)).to(self.device)
            hand_motion_dict['left']['hand_pose'] = hm_params['left_hand_pose'].reshape(n_samples_time, -1)
            hand_motion_dict['left']['global_orient'] = target_gr[:, :, SMPLX_JOINTS['left_wrist']].reshape(n_samples_time, -1)
            hand_motion_dict['left']['transl'] = target_xyz[:, :, SMPLX_JOINTS['left_wrist']].reshape(n_samples_time, -1)
            hand_motion_dict['left']['betas'] = torch.zeros((n_samples_time, 10)).to(self.device)

        else:
            
            hand_motion_dict = motion_dict

            n_samples_time = hand_motion_dict['right']['global_orient'].shape[0]
            time = n_samples_time 
            nsamples = 1

          
        for k in hand_dict.keys():

            hand_out = self.hand_dict[k](**hand_motion_dict[k])
            
            out_joints = hand_out.joints.reshape((nsamples, time, -1, 3))
            out_vertices = hand_out.vertices.reshape((nsamples, time, -1, 3))

            if cpu_flag:
                out_joints = out_joints.detach().cpu().numpy()
                out_vertices = out_vertices.detach().cpu().numpy()

            hand_dict[k] = {"joints": out_joints,
                            "vertices": out_vertices, 
                            "faces": self.hand_dict[k].faces,
                            **hand_motion_dict[k]}
 
        camera_dict = {}

        camera_dict['right'] = {"camera_rot": hand_motion_dict['right']["global_orient"]
                                .reshape((nsamples, time, 3)).detach().cpu(), 
                                "camera_transl": hand_motion_dict['right']["transl"]
                                .reshape((nsamples, time, 3)).detach().cpu()}
       
        return hand_dict, camera_dict
    
    
    def forward_body(self, 
                 motion_dict, 
                 betas=None, 
                 cpu_flag=False,
                 body_vtemp = None,
                 add_fingertips=False, 
                 time=120,
                 gender_list=[]):
        
    
        body_vtemp_list = []
        nsamples = motion_dict['global_orient'].shape[0] // time
        
        if len(gender_list) == 0: 
            gender_list = ['neutral'] * nsamples
            
        if body_vtemp is not None: 
            assert type(body_vtemp) == list 
            
            # at least one non_vtemp 
            if not set(body_vtemp) == {'no_vtemp'}:
                
                for i in range(len(body_vtemp)):
                    if body_vtemp[i] == 'no_vtemp':
                        body_vtemp_i = self.default_vtemp_dict[gender_list[i]]
                    else:
                        body_vtemp_i = torch.tensor(trimesh.load(file_obj=body_vtemp[i]).vertices).to(self.device).float()    
                    
                    body_vtemp_list.append(body_vtemp_i)
         
        if betas is None:
            betas = torch.zeros([nsamples * time, self.num_betas],
                                dtype=motion_dict['global_orient'].dtype).to(self.device)
        else:
 
            betas = betas.reshape((nsamples * time, self.num_betas)).to(self.device)
     


        motion_dict['betas'] = betas
        motion_dict['expression'] = torch.zeros([nsamples*time, 10], dtype=motion_dict['global_orient'].dtype).to(self.device)        
        
        
        # forward it one by one
        if len(body_vtemp_list) > 0:
            
            out_joints, out_vertices = [], []
            
            for i in range(len(body_vtemp_list)):
                
                self.bm_dict[gender_list[i]].v_template = body_vtemp_list[i]               
                motion_dict_i = {k: v[time * i : time * (i + 1)] for k, v in motion_dict.items()}
                    
                out_i = self.bm_dict[gender_list[i]](**motion_dict_i)
 
                out_joints.append(out_i.joints.reshape((1, time, -1, 3)))
                out_vertices.append(out_i.vertices.reshape((1, time, -1, 3)))
            
            
            
            # closing remarks 
            self.bm_dict[gender_list[i]].v_template = self.default_vtemp_dict[gender_list[i]]
             
        # batch fashion forward                     
        else:            

            out_joints, out_vertices = [], []

            for i in range(nsamples):
       
                motion_dict_i = {k: v[time * i : time * (i + 1)] for k, v in motion_dict.items()}

                out_i = self.bm_dict[gender_list[i]](**motion_dict_i)

                out_joints.append(out_i.joints.reshape((1, time, -1, 3)))
                out_vertices.append(out_i.vertices.reshape((1, time, -1, 3)))


        out_joints = torch.cat(out_joints, dim=0)[:, :, :self.bm_dict['neutral'].NUM_JOINTS + 1]
        out_vertices = torch.cat(out_vertices, dim=0)



        if add_fingertips:
            out_joints = torch.cat([out_joints, out_vertices[:, :, self.TIPS_IDX]], dim=2)
            
        
        if cpu_flag:
            out_joints = out_joints.detach().cpu().numpy()
            out_vertices = out_vertices.detach().cpu().numpy()
        
        

        body_dict = {"joints": out_joints,
                     "vertices": out_vertices,
                     "transl": motion_dict["transl"].reshape((nsamples, time, 3)),
                     "faces": self.bm_dict['neutral'].faces}

        camera_dict = {"camera_rot": motion_dict["global_orient"]
                       .reshape((nsamples, time, 3)).detach().cpu(), 
                        "camera_transl": motion_dict["transl"]
                        .reshape((nsamples, time, 3)).detach().cpu()}
        
        return body_dict, camera_dict