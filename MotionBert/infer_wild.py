# import os
# import numpy as np
# import argparse
# from tqdm import tqdm
# import imageio
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from lib.utils.tools import *
# from lib.utils.learning import *
# from lib.utils.utils_data import flip_data
# from lib.data.dataset_wild import WildDetDataset
# from lib.utils.vismo import render_and_save

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
#     parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
#     parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
#     parser.add_argument('-v', '--vid_path', type=str, help='video path')
#     parser.add_argument('-o', '--out_path', type=str, help='output path')
#     parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
#     parser.add_argument('--focus', type=int, default=None, help='target person id')
#     parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
#     opts = parser.parse_args()
#     return opts

# opts = parse_args()
# args = get_config(opts.config)

# model_backbone = load_backbone(args)
# if torch.cuda.is_available():
#     model_backbone = nn.DataParallel(model_backbone)
#     model_backbone = model_backbone.cuda()

# print('Loading checkpoint', opts.evaluate)
# checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
# model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
# model_pos = model_backbone
# model_pos.eval()
# testloader_params = {
#           'batch_size': 1,
#           'shuffle': False,
#           'num_workers': 8,
#           'pin_memory': True,
#           'prefetch_factor': 4,
#           'persistent_workers': True,
#           'drop_last': False
# }

# vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
# fps_in = vid.get_meta_data()['fps']
# vid_size = vid.get_meta_data()['size']
# os.makedirs(opts.out_path, exist_ok=True)

# if opts.pixel:
#     # Keep relative scale with pixel coornidates
#     wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
# else:
#     # Scale to [-1,1]
#     wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

# test_loader = DataLoader(wild_dataset, **testloader_params)

# results_all = []
# with torch.no_grad():
#     for batch_input in tqdm(test_loader):
#         N, T = batch_input.shape[:2]
#         if torch.cuda.is_available():
#             batch_input = batch_input.cuda()
#         if args.no_conf:
#             batch_input = batch_input[:, :, :, :2]
#         if args.flip:    
#             batch_input_flip = flip_data(batch_input)
#             predicted_3d_pos_1 = model_pos(batch_input)
#             predicted_3d_pos_flip = model_pos(batch_input_flip)
#             predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
#             predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
#         else:
#             predicted_3d_pos = model_pos(batch_input)
#         if args.rootrel:
#             predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
#         else:
#             predicted_3d_pos[:,0,0,2]=0
#             pass
#         if args.gt_2d:
#             predicted_3d_pos[...,:2] = batch_input[...,:2]
#         results_all.append(predicted_3d_pos.cpu().numpy())

# results_all = np.hstack(results_all)
# results_all = np.concatenate(results_all)
# render_and_save(results_all, '%s/X3D.mp4' % (opts.out_path), keep_imgs=False, fps=fps_in)
# if opts.pixel:
#     # Convert to pixel coordinates
#     results_all = results_all * (min(vid_size) / 2.0)
#     results_all[:,:,:2] = results_all[:,:,:2] + np.array(vid_size) / 2.0
# np.save('%s/X3D.npy' % (opts.out_path), results_all)
#--working------

import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save

def parse_args():
    parser = argparse.ArgumentParser(description="3D Pose Estimation with MotionBERT using AlphaPose JSON")
    parser.add_argument("--config", type=str, default="configs\\pose3d\\MB_ft_h36m_global_lite.yaml", help="Path to the config file")
    parser.add_argument('-e', '--evaluate', default=r"C:\Users\akhileshsing2024\MotionBERT\checkpoint\pose3d\FT_MB_lite_MB_ft_h36m_global_lite\best_epoch.bin", type=str, help="Checkpoint to evaluate")
    parser.add_argument('-j', '--json_path', type=str, required=True, help="AlphaPose detection result JSON path")
    parser.add_argument('-v', '--vid_path', type=str, required=True, help="Video path")
    parser.add_argument('-o', '--out_path', type=str, required=True, help="Output path")
    parser.add_argument('--pixel', action='store_true', help="Align with pixel coordinates")
    parser.add_argument('--focus', type=int, default=None, help="Target person ID")
    parser.add_argument('--clip_len', type=int, default=243, help="Clip length for network input")
    return parser.parse_args()

# Parse arguments
opts = parse_args()
args = get_config(opts.config)

# Load MotionBERT model
model_backbone = load_backbone(args)
if torch.cuda.is_available():
    model_backbone = nn.DataParallel(model_backbone)
    model_backbone = model_backbone.cuda()

# Load checkpoint and handle module prefix
print(f"Loading checkpoint: {opts.evaluate}")
if not os.path.exists(opts.evaluate):
    raise FileNotFoundError(f"Checkpoint file {opts.evaluate} not found")
checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)

# Strip 'module.' prefix from state_dict if present
state_dict = checkpoint['model_pos']
# if any(key.startswith('module.') for key in state_dict.keys()):
#     print("Stripping 'module.' prefix from state_dict")
#     state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

try:
    model_backbone.load_state_dict(state_dict, strict=True)
except RuntimeError as e:
    print(f"Error loading state_dict: {e}")
    print("Check if the model architecture matches the checkpoint or if config file is correct")
    exit(1)

model_pos = model_backbone
model_pos.eval()

# Load video and metadata
try:
    vid = imageio.get_reader(opts.vid_path, 'ffmpeg')
except FileNotFoundError:
    raise FileNotFoundError(f"Video file {opts.vid_path} not found")
fps_in = vid.get_meta_data()['fps']
vid_size = vid.get_meta_data()['size']
os.makedirs(opts.out_path, exist_ok=True)

# Load AlphaPose JSON and create dataset
if not os.path.exists(opts.json_path):
    raise FileNotFoundError(f"JSON file {opts.json_path} not found")
if opts.pixel:
    # Keep relative scale with pixel coordinates
    wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
else:
    # Scale to [-1,1]
    wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)


# Create DataLoader
testloader_params = {
    'batch_size': 1,
    'shuffle': False,
    'num_workers': 0,  # Set to 0 for Windows compatibility
    'pin_memory': True,
    'drop_last': False
}
test_loader = DataLoader(wild_dataset, **testloader_params)
# for clip in test_loader:
#     clip = clip.to(torch.float32)  # Shape: [1, F, 17, 3]
#     with torch.no_grad():
#         output = model_pos(clip)  # Shape: [1, F, 17, 3] (x, y, z in meters)
#     print("Sample 3D keypoint (first frame, first keypoint):", output[ :])
#     break
# # Process video and predict 3D poses
results_all = []
with torch.no_grad():
    for batch_input in tqdm(test_loader, desc="Processing frames"):
      
        N, T = batch_input.shape[:2]
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        if args.no_conf:
            batch_input = batch_input[:, :, :, :2]
        if args.flip:
            batch_input_flip = flip_data(batch_input)
            predicted_3d_pos_1 = model_pos(batch_input)
            #print(predicted_3d_pos_1)
            predicted_3d_pos_flip = model_pos(batch_input_flip)
            #print(predicted_3d_pos_2)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)
            predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
        else:
            predicted_3d_pos = model_pos(batch_input)
        if args.rootrel:
            predicted_3d_pos[:,:,0,:]=0  # Root-relative 3D coordinates

        else:
            predicted_3d_pos[:,0,0,2]=0
        if args.gt_2d:
            predicted_3d_pos[...,:2] = batch_input[...,:2]
            #print(predicted_3d_pos)
        results_all.append(predicted_3d_pos.cpu().numpy())
import pandas as pd
# Combine results
# results_all = np.concatenate(results_all, axis=0)  # Shape: [720, 17, 3]
# results_all = np.transpose(results_all, (1, 2, 0))  # Shape: [17, 3, 720] for vismo.py

results_all = np.hstack(results_all)
results_all = np.concatenate(results_all)
#print(results_all[0])
# Save 3D video and NumPy file
render_and_save(results_all, f"{opts.out_path}\\X3D.mp4", keep_imgs=False, fps=fps_in)
if opts.pixel:
    # Convert to pixel coordinates
    print("Raw 3D coordinates:", results_all)
    results_all = results_all * (min(vid_size) / 2.0)
    print("After scaling:", results_all)
    results_all[:,:,:2] = results_all[:,:,:2] + np.array(vid_size) / 2.0
    print("After offset:", results_all)
np.save(f"{opts.out_path}\\X3D.npy", results_all)
print(f"Saved 3D video to {opts.out_path}\\X3D.mp4 and 3D keypoints to {opts.out_path}\\X3D.npy")


