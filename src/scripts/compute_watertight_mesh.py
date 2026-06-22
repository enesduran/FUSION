# This script is used to compute watertight mesh faces based on the normal mesh. 
import os 
import torch
import joblib
import trimesh 
import numpy as np 

basemesh2hand = joblib.load('data/body_models/MANO_SMPLX_vertex_ids.pkl')

rhand_watertight = trimesh.load_mesh(f'{os.getcwd()}/data/body_models/watertight/rhand_watertight.ply')
lhand_watertight = trimesh.load_mesh(f'{os.getcwd()}/data/body_models/watertight/lhand_watertight.ply')

print("rhand_watertight", rhand_watertight.is_watertight, "lhand_watertight", lhand_watertight.is_watertight)
 
base_mesh = trimesh.load_mesh(f'{os.getcwd()}/data/body_models/watertight/base_mesh.ply')

# everthing is ok in this case. 
watertight_mesh_w_hand = trimesh.load_mesh(f'{os.getcwd()}/data/body_models/watertight/headless_feetless.ply')
watertight_mesh_wo_hand = trimesh.load_mesh(f'{os.getcwd()}/data/body_models/watertight/headless_feetless_handless.ply')

print('base_mesh', base_mesh.is_watertight, 'watertight_mesh', watertight_mesh_w_hand.is_watertight, 'handless', watertight_mesh_wo_hand.is_watertight) 

 
base_vertices = torch.tensor(base_mesh.vertices)
base_faces = torch.tensor(base_mesh.faces)

watertight_w_hand_vertices = torch.tensor(watertight_mesh_w_hand.vertices)
watertight_w_hand_faces = torch.tensor(watertight_mesh_w_hand.faces)
watertight_wo_hand_vertices = torch.tensor(watertight_mesh_wo_hand.vertices)
watertight_wo_hand_faces = torch.tensor(watertight_mesh_wo_hand.faces)


# make sure that subsamples mesh is a subset 
assert np.array([v in base_vertices for v in watertight_w_hand_vertices]).all(), 'Problem'
assert np.array([v in base_vertices for v in watertight_wo_hand_vertices]).all(), 'Problem'

base2watertight_w_hand, watertight2base_w_hand = {}, {}
base2watertight_wo_hand, watertight2base_wo_hand = {}, {}

for i in range(watertight_w_hand_vertices.shape[0]):

    idx_w_hand = torch.all(watertight_w_hand_vertices[i] == base_vertices, dim=1).nonzero().item()

    watertight2base_w_hand[i] = idx_w_hand 
    base2watertight_w_hand[idx_w_hand] = i 


for i in range(watertight_wo_hand_vertices.shape[0]):

    idx_wo_hand = torch.all(watertight_wo_hand_vertices[i] == base_vertices, dim=1).nonzero().item()

    watertight2base_wo_hand[i] = idx_wo_hand 
    base2watertight_wo_hand[idx_wo_hand] = i 


watertight_rhand_vertex_ids = np.array([base2watertight_w_hand[elem] for elem in basemesh2hand['right_hand']])
watertight_lhand_vertex_ids = np.array([base2watertight_w_hand[elem] for elem in basemesh2hand['left_hand']])

# Find the face idxs (if both 3 values in the face in the watertight)
watertight_w_hand_lhand_face_ids, watertight_w_hand_rhand_face_ids = [], []


for i, watertight_face_i in enumerate(watertight_w_hand_faces):

    is_in_rhand = [vertex_i.item() in watertight_rhand_vertex_ids for vertex_i in watertight_face_i]
    is_in_lhand = [vertex_j.item() in watertight_lhand_vertex_ids for vertex_j in watertight_face_i]

    
    if all(is_in_rhand):
        watertight_w_hand_rhand_face_ids.append(i)
    elif all(is_in_lhand):
        watertight_w_hand_lhand_face_ids.append(i)

assert len(watertight_w_hand_lhand_face_ids) == len(watertight_w_hand_rhand_face_ids) == 1538, 'Problem with the number of faces'


base_lhand_face_ids, base_rhand_face_ids = [], []
base2watertight_faces_dict = {}

for i, base_face_i in enumerate(base_faces):

    is_in_rhand = [vertex_k.item() in basemesh2hand['right_hand'] for vertex_k in base_face_i]
    is_in_lhand = [vertex_l.item() in basemesh2hand['left_hand'] for vertex_l in base_face_i]

    # is_in_watertight_body = [vertex_m.item() in np.array(list(base2watertight_w_hand.keys())) for vertex_m in base_face_i]


    if all(is_in_rhand):
        base_rhand_face_ids.append(i)
    elif all(is_in_lhand):
        base_lhand_face_ids.append(i)

    
    for j, watertight_face_i in enumerate(watertight_w_hand_faces):

        if base_face_i[0].item() in base2watertight_w_hand.keys() and \
            base_face_i[1].item() in base2watertight_w_hand.keys() and \
                base_face_i[2].item() in base2watertight_w_hand.keys():

            v1 = base2watertight_w_hand[base_face_i[0].item()]
            v2 = base2watertight_w_hand[base_face_i[1].item()]
            v3 = base2watertight_w_hand[base_face_i[2].item()]

            if np.array_equal(np.array([v1, v2, v3]), watertight_face_i):
                base2watertight_faces_dict[i] = j 
                break

assert len(base_lhand_face_ids) == len(base_rhand_face_ids) == 1538, 'Problem with the number of faces'
 

joblib.dump({'watertight2base_w_hand': watertight2base_w_hand, 
             'base2watertight_w_hand': base2watertight_w_hand,
             'base2watertight_wo_hand': base2watertight_wo_hand,
             'watertight2base_wo_hand': watertight2base_wo_hand,
             'base2watertight_faces_dict': base2watertight_faces_dict,
             'basemesh_rhand_vertex_ids': basemesh2hand['left_hand'],
             'basemesh_lhand_vertex_ids': basemesh2hand['right_hand'],
             'watertight_w_hand_rhand_vertex_ids': watertight_rhand_vertex_ids,
             'watertight_w_hand_lhand_vertex_ids': watertight_lhand_vertex_ids,
             'base_rhand_face_ids': base_rhand_face_ids,
             'base_lhand_face_ids': base_lhand_face_ids,
             'watertight_w_hand_rhand_face_ids': watertight_w_hand_rhand_face_ids,
             'watertight_w_hand_lhand_face_ids': watertight_w_hand_lhand_face_ids,
             'watertight_rhand_faces': rhand_watertight.faces,
             'watertight_lhand_faces': lhand_watertight.faces,
             'watertight_w_hand_faces': watertight_w_hand_faces, 
             'watertight_wo_hand_faces': watertight_wo_hand_faces},
            f'{os.getcwd()}/data/body_models/watertight/conversion_dict.pkl')
 
