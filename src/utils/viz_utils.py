import os
import torch
import subprocess
from src.utils.transforms3d import transform_body_pose

color1 = (160 / 255, 160 / 255, 0 / 255, 1.0)
color2 = (0 / 255, 160 / 255, 160 / 255, 1.0)
color3 = (160 / 255, 0 / 255, 160 / 255, 1.0)

obj_color = (1.0, 0.2, 0.2, 1.0)         


# ensure the dictionary keys are compatible with aitviewer
def pack_to_render(rots, trans, pose_repr='6d', device='cpu'):
    """rots: (B*T, D)"""
    
    
    if pose_repr != 'aa':
        body_pose = transform_body_pose(rots, f"{pose_repr}->aa")
    else:
        body_pose = rots
    if trans is None:
        trans = torch.zeros((rots.shape[0], rots.shape[1], 3),
                             device=rots.device)

    
    # smplx 
    if body_pose.shape[-1] == 165:
        render_d = {'transl': trans.to(device).float(),
                    'global_orient': body_pose[..., :3].to(device).float(),
                    'body_pose': body_pose[..., 3:66].to(device).float(),
                    'jaw_pose': body_pose[..., 66:69].to(device).float(),
                    'leye_pose': body_pose[..., 69:72].to(device).float(),
                    'reye_pose': body_pose[..., 72:75].to(device).float(),
                    'left_hand_pose': body_pose[..., 75:120].to(device).float(),
                    'right_hand_pose': body_pose[..., 120:165].to(device).float()}
    # mano 
    elif body_pose.shape[-1] == 45:
        render_d = {'trans': trans.to(device).float(),
                    'poses_root': body_pose[..., :3].to(device).float(),
                    'poses_body': body_pose[..., 3:].to(device).float()}
    
    # smpl
    elif body_pose.shape[-1] == 72:
        render_d = {'trans': trans.to(device).float(),
                    'global_orient': body_pose[..., :3].to(device).float(),
                    'body_pose': body_pose[..., 3:].to(device).float()}
    # smplx w/o hand
    else:
        render_d = {'trans': trans.to(device).float(),
                    'poses_root': body_pose[..., :3].to(device).float(),
                    'poses_body': body_pose[..., 3:].to(device).float()}
        
    return render_d


def merge_videos(video_list, output_path, info_strs=None, height=900):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    video_list = [os.path.abspath(v) for v in video_list]
    n = len(video_list)
    
    if info_strs is None:
        info_strs = [''] * n  # default to empty strings
        display_labels = False
    else:
        display_labels = True
    
    # Video labels A, B, C, etc.
    video_labels = [chr(65 + i) for i in range(n)]  # A, B, C, D, ...
    
    # Input part
    input_cmd = ' '.join([f"-i {video}" for video in video_list])
    
    # Filter graph parts
    drawtext_filters = []
    scale_filters = []
    border_filters = []
    label_filters = []
    
    for i in range(n):
    
        # Add video label (A, B, C) in top left corner
        label_text = f"{video_labels[i]}: {info_strs[i]}" if info_strs[i] else video_labels[i] if display_labels else ''

        draw = f"[{i}]drawtext=text='{label_text}':fontfile=/usr/share/fonts/truetype/msttcorefonts/arial.ttf:fontcolor=white:fontsize=w/20:x=10:y=10:box=1:boxcolor=black@0.5:boxborderw=5[v{i}]"
        
        # Scale video
        scale = f"[v{i}]scale=-1:{height}[sv{i}]"
        
        # Add border (except for the last video to avoid extra border on the right)
        if i < n - 1:
            border = f"[sv{i}]pad=iw+20:ih:0:0:color=black[bv{i}]"
            border_filters.append(border)
            label_filters.append(f"[bv{i}]")
        else:
            label_filters.append(f"[sv{i}]")
        
        drawtext_filters.append(draw)
        scale_filters.append(scale)
    
    # Combine filters
    filter_complex = '; '.join(drawtext_filters + scale_filters + border_filters)
    filter_complex += f"; {''.join(label_filters)}hstack=inputs={n}[stackout]"
    
    # Final command
    cmd = f"/usr/bin/ffmpeg -y {input_cmd} -filter_complex \"{filter_complex}\" -map \"[stackout]\" \"{output_path}\""
    subprocess.call(cmd, shell=True)

 