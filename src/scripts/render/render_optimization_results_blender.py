import glob
import subprocess



TASK = 'object_guided_quant_res'
# TASK = 'self_contact_quant_res'
# TASK = 'keypoint_tracking_quant_res'


# nikos setting 
#  camera_locx=-4.5,
# camera_locy=-2.3,
# camera_locz=6,
# camera_fov=110,

# default setting if you dont give anything 
# camera_locx=-4.5,
# camera_locy=-2.3,
# camera_locz=3,
# camera_fov=80

if TASK == 'object_guided_quant_res':
    camera_locx = -4.5
    camera_locy = -2.3
    camera_locz = 4.0
    camera_fov = 70.0
    
    floor_flag = True
    sideview = 'half_front'
    num_poses = 4


elif TASK == 'self_contact_quant_res':
    camera_locx = -3.9
    camera_locy = -3.0   
    camera_locz = 5.0  # up
    camera_fov = 110

    floor_flag = False
    sideview = 'self'
    num_poses = 4

    # camera_locx = -2.3
    # camera_locy = -6.0   
    # camera_locz = 4.5  # up
    # camera_fov = 110
elif TASK == 'keypoint_tracking_quant_res':
    camera_locx = -4.5
    camera_locy = -2.3
    camera_locz = 3.0
    camera_fov = 70.0

    floor_flag = True
    sideview = 'front'
    num_poses = 4


camera_string = f'cam.camera_locx={camera_locx} \
                  cam.camera_locy={camera_locy} \
                  cam.camera_locz={camera_locz} \
                  cam.camera_fov={camera_fov}'
 
tarfolder_path = f'/is/cluster/fast/eduran2/fusion/fusion_runs/main/0/{TASK}/*.tar'
vis_folder = tarfolder_path.replace('/*.tar', '').replace('quant_res', 'vis_res_blender')


for tarfile_path in sorted(glob.glob(tarfolder_path)):

    print(tarfile_path)
 
    image_cmd = f'python src/render/render_single_fusion.py num={num_poses} \
        dataset_path={tarfile_path} mode=sequence savedir={vis_folder} view={sideview} \
            floor_flag={floor_flag} {camera_string}'
    
    subprocess.call(image_cmd, shell=True)

    video_cmd = f'python src/render/render_single_fusion.py dataset_path={tarfile_path} mode=video savedir={vis_folder}'
    # subprocess.call(video_cmd, shell=True)

                