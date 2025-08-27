import os
import os.path as osp
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import time
import random
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.utils.utils_mesh import flip_thetas_batch
from lib.data.dataset_wild import WildDetDataset
from lib.data.dataset_mesh import MotionSMPL
# from lib.model.loss import *
from lib.model.model_mesh import MeshRegressor
from lib.utils.vismo import render_and_save, motion2video_mesh
from lib.utils.utils_smpl import *
from scipy.optimize import least_squares

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mesh/MB_ft_pw3d.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--ref_3d_motion_path', type=str, default=None, help='3D motion path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts

def err(p, x, y):
    return np.linalg.norm(p[0] * x + np.array([p[1], p[2], p[3]]) - y, axis=-1).mean()

def solve_scale(x, y):
    print('Estimating camera transformation.')
    best_res = 100000
    best_scale = None
    for init_scale in tqdm(range(0,2000,5)):
        p0 = [init_scale, 0.0, 0.0, 0.0]
        est = least_squares(err, p0, args = (x.reshape(-1,3), y.reshape(-1,3)))
        if est['fun'] < best_res:
            best_res = est['fun']
            best_scale = est['x'][0]
    print('Pose matching error = %.2f mm.' % best_res)
    return best_scale

opts = parse_args()
args = get_config(opts.config)

# root_rel
# args.rootrel = True

smpl = SMPL(args.data_root, batch_size=1).cuda()
J_regressor = smpl.J_regressor_h36m

end = time.time()
model_backbone = load_backbone(args)
print(f'init backbone time: {(time.time()-end):02f}s')
end = time.time()
model = MeshRegressor(args, backbone=model_backbone, dim_rep=args.dim_rep, hidden_dim=args.hidden_dim, dropout_ratio=args.dropout)
print(f'init whole model time: {(time.time()-end):02f}s')

if torch.cuda.is_available():
    model = nn.DataParallel(model)
    model = model.cuda()

chk_filename = opts.evaluate if opts.evaluate else opts.resume
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()

testloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': True,
        #'prefetch_factor': 4,
        #'persistent_workers': True,
        'drop_last': False
}

vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
fps_in = vid.get_meta_data()['fps']
vid_size = vid.get_meta_data()['size']
os.makedirs(opts.out_path, exist_ok=True)

if opts.pixel:
    # Keep relative scale with pixel coornidates
    wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
    
    #wild_dataset =MotionSMPL(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
else:
    # Scale to [-1,1]
    wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)


test_loader = DataLoader(wild_dataset, **testloader_params)

verts_all = []
reg3d_all = []
with torch.no_grad():
    for batch_input in tqdm(test_loader):
        batch_size, clip_frames = batch_input.shape[:2]
        if torch.cuda.is_available():
            batch_input = batch_input.cuda().float()
        output = model(batch_input)   
        batch_input_flip = flip_data(batch_input)
        output_flip = model(batch_input_flip)
        output_flip_pose = output_flip[0]['theta'][:, :, :72]
        output_flip_shape = output_flip[0]['theta'][:, :, 72:]
        output_flip_pose = flip_thetas_batch(output_flip_pose)
        output_flip_pose = output_flip_pose.reshape(-1, 72)
        output_flip_shape = output_flip_shape.reshape(-1, 10)
        output_flip_smpl = smpl(
            betas=output_flip_shape,
            body_pose=output_flip_pose[:, 3:],
            global_orient=output_flip_pose[:, :3],
            pose2rot=True
        )
        output_flip_verts = output_flip_smpl.vertices.detach()
        J_regressor_batch = J_regressor[None, :].expand(output_flip_verts.shape[0], -1, -1).to(output_flip_verts.device)
        output_flip_kp3d = torch.matmul(J_regressor_batch, output_flip_verts)  # (NT,17,3) 
        output_flip_back = [{
            'verts': output_flip_verts.reshape(batch_size, clip_frames, -1, 3) * 1000.0,
            'kp_3d': output_flip_kp3d.reshape(batch_size, clip_frames, -1, 3),
        }]
        output_final = [{}]
        for k, v in output_flip_back[0].items():
            output_final[0][k] = (output[0][k] + output_flip_back[0][k]) / 2.0
        output = output_final
        verts_all.append(output[0]['verts'].cpu().numpy())
        reg3d_all.append(output[0]['kp_3d'].cpu().numpy())

# verts_all = np.hstack(verts_all)
# verts_all = np.concatenate(verts_all)
# reg3d_all = np.hstack(reg3d_all)
# reg3d_all = np.concatenate(reg3d_all)

# import copy
# motion = copy.deepcopy(verts_all)
# print(motion.shape[1])
# if opts.ref_3d_motion_path:
#     ref_pose = np.load(opts.ref_3d_motion_path)
#     x = ref_pose - ref_pose[:, :1]
#     y = reg3d_all - reg3d_all[:, :1]
#     scale = solve_scale(x, y)
#     root_cam = ref_pose[:, :1] * scale
#     verts_all = verts_all - reg3d_all[:,:1] + root_cam

# # render_and_save(verts_all,f"{opts.out_path}\\mesh.mp4")
# render_and_save(verts_all, f"{opts.out_path}\\mesh.mp4", keep_imgs=False, fps=fps_in, draw_face=True)

# print("saved output")

verts_all = np.hstack(verts_all)
verts_all = np.concatenate(verts_all)
reg3d_all = np.hstack(reg3d_all)
reg3d_all = np.concatenate(reg3d_all)
print("verts_all shape:", verts_all.shape)
print("reg3d_all shape:", reg3d_all.shape)
print("verts_all sample:", verts_all[0, :5])
print("verts_all min/max:", verts_all.min(), verts_all.max())
print("NaN in verts_all:", np.isnan(verts_all).any())
print("Inf in verts_all:", np.isinf(verts_all).any())

# Extract x, y, z coordinates for 17 keypoints from reg3d_all
x_coords = reg3d_all[:, :, 0]  # Shape: (331, 17)
y_coords = reg3d_all[:, :, 1]
z_coords = reg3d_all[:, :, 2]
print("X coords sample (first frame, first 5 keypoints):", x_coords[0, :5])
print("Y coords sample (first frame, first 5 keypoints):", y_coords[0, :5])
print("Z coords sample (first frame, first 5 keypoints):", z_coords[0, :5])

# Save keypoint coordinates to NumPy files
np.save(osp.join(opts.out_path, 'keypoints_x_coords.npy'), x_coords)
np.save(osp.join(opts.out_path, 'keypoints_y_coords.npy'), y_coords)
np.save(osp.join(opts.out_path, 'keypoints_z_coords.npy'), z_coords)
print("Saved keypoint x, y, z coordinates to .npy files")

# Save keypoint coordinates to CSV
import pandas as pd
all_coords = reg3d_all.reshape(-1, 3)
df = pd.DataFrame(all_coords, columns=['x', 'y', 'z'])
df['frame'] = np.repeat(np.arange(331), 17)
df['keypoint'] = np.tile(np.arange(17), 331)
df.to_csv(osp.join(opts.out_path, 'keypoints_coords.csv'), index=False)
print("Saved keypoint coordinates to keypoints_coords.csv")

if opts.ref_3d_motion_path:
    ref_pose = np.load(opts.ref_3d_motion_path)
    x = ref_pose - ref_pose[:, :1]
    y = reg3d_all - reg3d_all[:, :1]
    scale = solve_scale(x, y)
    root_cam = ref_pose[:, :1] * scale
    verts_all = verts_all - reg3d_all[:, :1] + root_cam
    print("Scaled verts_all min/max:", verts_all.min(), verts_all.max())

render_and_save(verts_all, osp.join(opts.out_path, 'mesh.mp4'), keep_imgs=True, fps=fps_in, draw_face=True)
print("saved output")