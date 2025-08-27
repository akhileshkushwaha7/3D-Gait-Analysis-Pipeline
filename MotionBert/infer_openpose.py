# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# import json
# import imageio
# from tqdm import tqdm
# import argparse
# from collections import OrderedDict
# from lib.utils.tools import get_config
# from lib.utils.vismo import render_and_save

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
#     parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, help='Checkpoint to evaluate')
#     parser.add_argument('-j', '--json_path', type=str, required=True, help='OpenPose JSON folder path')
#     parser.add_argument('-v', '--vid_path', type=str, required=True, help='Video path')
#     parser.add_argument('-o', '--out_path', type=str, required=True, help='Output directory for 3D poses')
#     parser.add_argument('--pixel', action='store_true', help='Align with pixel coordinates')
#     parser.add_argument('--focus', type=int, default=None, help='Target person ID (default: first person)')
#     parser.add_argument('--clip_len', type=int, default=243, help='Clip length for network input')
#     parser.add_argument('--no-conf', action='store_true', help='Ignore confidence scores')
#     parser.add_argument('--flip', action='store_true', help='Use flip augmentation')
#     parser.add_argument('--rootrel', action='store_true', help='Root-relative prediction')
#     parser.add_argument('--gt_2d', action='store_true', help='Use ground-truth 2D coordinates')
#     parser.add_argument('--num_joints', type=int, default=17, choices=[17, 25], help='Number of joints (17 for H36M, 25 for COCO)')
#     return parser.parse_args()

# def set_random_seed(seed=0):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

# def trunc_normal_(tensor, mean=0., std=1.):
#     import torch.nn.init as init
#     with torch.no_grad():
#         return init.trunc_normal_(tensor, mean=mean, std=std, a=-2*std, b=2*std)

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, st_mode="stage_st"):
#         super().__init__()
#         self.norm1_s = norm_layer(dim)
#         self.norm1_t = norm_layer(dim)
#         self.st_mode = st_mode
#         self.attn_s = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         self.attn_t = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         self.norm2_s = norm_layer(dim)
#         self.norm2_t = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp_s = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Dropout(drop),
#             nn.Linear(mlp_hidden_dim, dim),
#             nn.Dropout(drop)
#         )
#         self.mlp_t = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Dropout(drop),
#             nn.Linear(mlp_hidden_dim, dim),
#             nn.Dropout(drop)
#         )
#         self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)

#     def forward(self, x, F):
#         B, J, C = x.shape
#         x = x.reshape(-1, F, J, C)
#         if self.st_mode == "stage_st":
#             x = x.permute(0, 2, 1, 3).reshape(-1, F, C)
#             x = x + self.drop_path(self.attn_s(self.norm1_s(x)))
#             x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
#             x = x.reshape(-1, J, F, C).permute(0, 2, 1, 3).reshape(-1, J, C)
#         else:
#             x = x.reshape(-1, F, C)
#             x = x + self.drop_path(self.attn_t(self.norm1_t(x)))
#             x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
#             x = x.reshape(-1, J, F, C).permute(0, 2, 1, 3).reshape(-1, J, C)
#         return x

# class DSTformer(nn.Module):
#     def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
#                  depth=5, num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
#                  qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
#                  norm_layer=nn.LayerNorm, att_fuse=True):
#         super().__init__()
#         self.dim_out = dim_out
#         self.dim_feat = dim_feat
#         self.num_joints = num_joints
#         self.joints_embed = nn.Linear(dim_in, dim_feat)
#         self.pos_drop = nn.Dropout(p=drop_rate)
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.blocks_st = nn.ModuleList([
#             Block(
#                 dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
#                 st_mode="stage_st")
#             for i in range(depth)])
#         self.blocks_ts = nn.ModuleList([
#             Block(
#                 dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
#                 st_mode="stage_ts")
#             for i in range(depth)])
#         self.norm = norm_layer(dim_feat)
#         self.pre_logits = nn.Sequential(OrderedDict([
#             ('fc', nn.Linear(dim_feat, dim_rep)),
#             ('act', nn.Tanh())
#         ]))
#         self.head = nn.Linear(dim_rep, dim_out)
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
#         trunc_normal_(self.pos_embed, std=.02)
#         self.att_fuse = att_fuse
#         if self.att_fuse:
#             self.ts_attn = nn.ModuleList([nn.Linear(dim_feat*2, 2) for _ in range(depth)])
#             for i in range(depth):
#                 self.ts_attn[i].weight.data.fill_(0)
#                 self.ts_attn[i].bias.data.fill_(0.5)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x, return_rep=False):
#         B, F, J, C = x.shape
#         x = x.reshape(-1, J, C)
#         x = self.joints_embed(x)
#         x = x + self.pos_embed
#         _, J, C = x.shape
#         x = x.reshape(-1, F, J, C)
#         x = self.pos_drop(x)
#         for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
#             x_st = blk_st(x.reshape(-1, J, C), F)
#             x_ts = blk_ts(x.reshape(-1, J, C), F)
#             if self.att_fuse:
#                 att = self.ts_attn[idx]
#                 alpha = torch.cat([x_st, x_ts], dim=-1)
#                 alpha = att(alpha).softmax(dim=-1)
#                 x = x_st * alpha[:,:,0:1] + x_ts * alpha[:,:,1:2]
#             else:
#                 x = (x_st + x_ts) * 0.5
#         x = self.norm(x)
#         x = x.reshape(B, F, J, -1)
#         x = self.pre_logits(x)
#         if return_rep:
#             return x
#         x = self.head(x)
#         return x

# def flip_data(data, num_joints):
#     joint_order = [
#         'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
#         'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye',
#         'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel'
#     ] if num_joints == 25 else [
#         'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
#         'Neck', 'Nose', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist'
#     ]
#     left_joints = ['LShoulder', 'LElbow', 'LWrist', 'LHip', 'LKnee', 'LAnkle'] + (['LEye', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel'] if num_joints == 25 else [])
#     right_joints = ['RShoulder', 'RElbow', 'RWrist', 'RHip', 'RKnee', 'RAnkle'] + (['REye', 'REar', 'RBigToe', 'RSmallToe', 'RHeel'] if num_joints == 25 else [])
    
#     flipped_data = data.clone()
#     flipped_data[:, :, :, 0] = -flipped_data[:, :, :, 0]
#     for left, right in zip(left_joints, right_joints):
#         left_idx = joint_order.index(left)
#         right_idx = joint_order.index(right)
#         flipped_data[:, :, [left_idx, right_idx]] = flipped_data[:, :, [right_idx, left_idx]]
#     return flipped_data

# class OpenPoseDataset(Dataset):
#     def __init__(self, json_path, clip_len=243, vid_size=(1920, 1080), scale_range=None, focus=None, num_joints=17):
#         self.json_path = json_path
#         self.clip_len = clip_len
#         self.vid_size = vid_size
#         self.scale_range = scale_range
#         self.focus = focus
#         self.num_joints = num_joints
#         self.joint_order = [
#             'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
#             'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye',
#             'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel'
#         ] if num_joints == 25 else [
#             'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
#             'Neck', 'Nose', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist'
#         ]
#         self.data = self.load_openpose_json()
#         self.clips = self.prepare_clips()
    
#     def load_openpose_json(self):
#         json_files = sorted([f for f in os.listdir(self.json_path) if f.endswith('.json')])
#         data = []
#         for json_file in json_files:
#             with open(os.path.join(self.json_path, json_file), 'r') as f:
#                 json_data = json.load(f)
#                 if not json_data['people']:
#                     keypoints = np.zeros((self.num_joints, 3))
#                 else:
#                     person_idx = self.focus if self.focus is not None else 0
#                     if person_idx >= len(json_data['people']):
#                         keypoints = np.zeros((self.num_joints, 3))
#                     else:
#                         keypoints = json_data['people'][person_idx]['pose_keypoints_2d']
#                         keypoints = np.array(keypoints).reshape(-1, 3)
#                         if self.num_joints == 17:
#                             filtered_keypoints = np.zeros((17, 3))
#                             for i, joint in enumerate(self.joint_order):
#                                 idx = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
#                                        'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle'].index(joint)
#                                 filtered_keypoints[i] = keypoints[idx]
#                             keypoints = filtered_keypoints
#                 data.append(keypoints)
#         data = np.array(data)
#         root_idx = self.joint_order.index('MidHip')
#         data[:, :, :2] -= data[:, [root_idx], :2]
#         if self.scale_range:
#             max_dist = np.max(np.abs(data[:, :, :2]))
#             if max_dist > 0:
#                 data[:, :, :2] /= max_dist
#         else:
#             data[:, :, :2] /= min(self.vid_size)
#         return data
    
#     def prepare_clips(self):
#         clips = []
#         for i in range(0, len(self.data), self.clip_len):
#             clip = self.data[i:i + self.clip_len]
#             if len(clip) < self.clip_len:
#                 clip = np.pad(clip, ((0, self.clip_len - len(clip)), (0, 0), (0, 0)), mode='edge')
#             clips.append(clip)
#         return np.array(clips)
    
#     def __len__(self):
#         return len(self.clips)
    
#     def __getitem__(self, idx):
#         return torch.tensor(self.clips[idx], dtype=torch.float32)

# def main():
#     opts = parse_args()
#     set_random_seed()
#     args = get_config(opts.config)
    
#     os.makedirs(opts.out_path, exist_ok=True)
    
#     vid = imageio.get_reader(opts.vid_path, 'ffmpeg')
#     fps_in = vid.get_meta_data()['fps']
#     vid_size = vid.get_meta_data()['size']
    
#     dataset = OpenPoseDataset(
#         json_path=opts.json_path,
#         clip_len=opts.clip_len,
#         vid_size=vid_size,
#         scale_range=[1, 1] if not opts.pixel else None,
#         focus=opts.focus,
#         num_joints=opts.num_joints
#     )
    
#     test_loader = DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=8,
#         pin_memory=True,
#         prefetch_factor=4,
#         persistent_workers=True,
#         drop_last=False
#     )
    
#     model_pos = DSTformer(
#         dim_in=3,
#         dim_out=3,
#         dim_feat=args.dim_feat if hasattr(args, 'dim_feat') else 256,
#         dim_rep=args.dim_rep if hasattr(args, 'dim_rep') else 512,
#         depth=args.depth if hasattr(args, 'depth') else 5,
#         num_heads=args.num_heads if hasattr(args, 'num_heads') else 8,
#         mlp_ratio=args.mlp_ratio if hasattr(args, 'mlp_ratio') else 4,
#         num_joints=opts.num_joints,
#         maxlen=opts.clip_len,
#         qkv_bias=True,
#         drop_rate=args.drop_rate if hasattr(args, 'drop_rate') else 0.,
#         attn_drop_rate=args.attn_drop_rate if hasattr(args, 'attn_drop_rate') else 0.,
#         drop_path_rate=args.drop_path_rate if hasattr(args, 'drop_path_rate') else 0.
#     )
    
#     print(f'Loading checkpoint {opts.evaluate}')
#     checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
    
#     # Strip 'module.' prefix from state_dict keys
#     state_dict = checkpoint['model_pos'] if 'model_pos' in checkpoint else checkpoint
#     new_state_dict = OrderedDict()
#     for key, value in state_dict.items():
#         new_key = key.replace('module.', '') if key.startswith('module.') else key
#         new_state_dict[new_key] = value
    
#     try:
#         model_pos.load_state_dict(new_state_dict, strict=True)
#     except RuntimeError as e:
#         print(f"Warning: {e}. Loading with strict=False.")
#         model_pos.load_state_dict(new_state_dict, strict=False)
    
#     if torch.cuda.is_available():
#         model_pos = nn.DataParallel(model_pos)
#         model_pos = model_pos.cuda()
    
#     model_pos.eval()
#     results_all = []
#     with torch.no_grad():
#         for batch_input in tqdm(test_loader):
#             N, T = batch_input.shape[:2]
#             if torch.cuda.is_available():
#                 batch_input = batch_input.cuda()
#             if args.no_conf:
#                 batch_input = batch_input[:, :, :, :2]
#             if args.flip:
#                 batch_input_flip = flip_data(batch_input, opts.num_joints)
#                 predicted_3d_pos_1 = model_pos(batch_input)
#                 predicted_3d_pos_flip = model_pos(batch_input_flip)
#                 predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip, opts.num_joints)
#                 predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
#             else:
#                 predicted_3d_pos = model_pos(batch_input)
#             if args.rootrel:
#                 predicted_3d_pos[:, :, 0, :] = 0
#             else:
#                 predicted_3d_pos[:, 0, 0, 2] = 0
#             if args.gt_2d:
#                 predicted_3d_pos[..., :2] = batch_input[..., :2]
#             results_all.append(predicted_3d_pos.cpu().numpy())
    
#     results_all = np.concatenate(results_all, axis=1)
    
#     joint_order = [
#         'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
#         'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye',
#         'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel'
#     ] if opts.num_joints == 25 else [
#         'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
#         'Neck', 'Nose', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist'
#     ]
#     output_data = []
#     for frame_idx in range(results_all.shape[1]):
#         for joint_idx, joint_name in enumerate(joint_order):
#             output_data.append({
#                 'Frame': frame_idx,
#                 'Keypoint': joint_name,
#                 'X_3D': results_all[0, frame_idx, joint_idx, 0],
#                 'Y_3D': results_all[0, frame_idx, joint_idx, 1],
#                 'Z_3D': results_all[0, frame_idx, joint_idx, 2]
#             })
#     output_df = pd.DataFrame(output_data)
#     output_df.to_excel(os.path.join(opts.out_path, "predicted_3d_poses.xlsx"), index=False)
    
#     if opts.pixel:
#         results_all = results_all * (min(vid_size) / 2.0)
#         results_all[:, :, :, :2] = results_all[:, :, :, :2] + np.array(vid_size) / 2.0
    
#     np.save(os.path.join(opts.out_path, 'X3D.npy'), results_all)
#     render_and_save(results_all, os.path.join(opts.out_path, 'X3D.mp4'), keep_imgs=False, fps=fps_in)

# if __name__ == "__main__":
#     main()

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import imageio
from tqdm import tqdm
import argparse
from collections import OrderedDict
from lib.utils.tools import get_config
from lib.utils.vismo import render_and_save

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, help='Checkpoint to evaluate')
    parser.add_argument('-j', '--json_path', type=str, required=True, help='OpenPose JSON folder path')
    parser.add_argument('-v', '--vid_path', type=str, required=True, help='Video path')
    parser.add_argument('-o', '--out_path', type=str, required=True, help='Output directory for 3D poses')
    parser.add_argument('--pixel', action='store_true', help='Align with pixel coordinates')
    parser.add_argument('--focus', type=int, default=None, help='Target person ID (default: first person)')
    parser.add_argument('--clip_len', type=int, default=243, help='Clip length for network input')
    parser.add_argument('--no-conf', action='store_true', help='Ignore confidence scores')
    parser.add_argument('--flip', action='store_true', help='Use flip augmentation')
    parser.add_argument('--rootrel', action='store_true', help='Root-relative prediction')
    parser.add_argument('--gt_2d', action='store_true', help='Use ground-truth 2D coordinates')
    parser.add_argument('--num_joints', type=int, default=17, choices=[17, 25], help='Number of joints (17 for H36M, 25 for COCO)')
    return parser.parse_args()

def set_random_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def trunc_normal_(tensor, mean=0., std=1.):
    import torch.nn.init as init
    with torch.no_grad():
        return init.trunc_normal_(tensor, mean=mean, std=std, a=-2*std, b=2*std)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, st_mode="stage_st"):
        super().__init__()
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.st_mode = st_mode
        self.attn_s = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn_t = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_s = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        self.mlp_t = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)

    def forward(self, x, F):
        B, J, C = x.shape
        x = x.reshape(-1, F, J, C)
        if self.st_mode == "stage_st":
            x = x.permute(0, 2, 1, 3).reshape(-1, F, C)
            x = x + self.drop_path(self.attn_s(self.norm1_s(x)))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x.reshape(-1, J, F, C).permute(0, 2, 1, 3).reshape(-1, J, C)
        else:
            x = x.reshape(-1, F, C)
            x = x + self.drop_path(self.attn_t(self.norm1_t(x)))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x.reshape(-1, J, F, C).permute(0, 2, 1, 3).reshape(-1, J, C)
        return x

class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, att_fuse=True):
        super().__init__()
        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.num_joints = num_joints
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks_st = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode="stage_st")
            for i in range(depth)])
        self.blocks_ts = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode="stage_ts")
            for i in range(depth)])
        self.norm = norm_layer(dim_feat)
        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))
        self.head = nn.Linear(dim_rep, dim_out)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.pos_embed, std=.02)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat*2, 2) for _ in range(depth)])
            for i in range(depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_rep=False):
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)
        x = self.joints_embed(x)
        x = x + self.pos_embed
        _, J, C = x.shape
        x = x.reshape(-1, F, J, C)
        x = self.pos_drop(x)
        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x.reshape(-1, J, C), F)
            x_ts = blk_ts(x.reshape(-1, J, C), F)
            if self.att_fuse:
                att = self.ts_attn[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                alpha = att(alpha).softmax(dim=-1)
                x = x_st * alpha[:,:,0:1] + x_ts * alpha[:,:,1:2]
            else:
                x = (x_st + x_ts) * 0.5
        x = self.norm(x)
        x = x.reshape(B, F, J, -1)
        x = self.pre_logits(x)
        if return_rep:
            return x
        x = self.head(x)
        return x

def flip_data(data, num_joints):
    joint_order = [
        'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
        'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye',
        'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel'
    ] if num_joints == 25 else [
        'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Neck', 'Nose', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist'
    ]
    left_joints = ['LShoulder', 'LElbow', 'LWrist', 'LHip', 'LKnee', 'LAnkle'] + (['LEye', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel'] if num_joints == 25 else [])
    right_joints = ['RShoulder', 'RElbow', 'RWrist', 'RHip', 'RKnee', 'RAnkle'] + (['REye', 'REar', 'RBigToe', 'RSmallToe', 'RHeel'] if num_joints == 25 else [])
    
    flipped_data = data.clone()
    flipped_data[:, :, :, 0] = -flipped_data[:, :, :, 0]
    for left, right in zip(left_joints, right_joints):
        left_idx = joint_order.index(left)
        right_idx = joint_order.index(right)
        flipped_data[:, :, [left_idx, right_idx]] = flipped_data[:, :, [right_idx, left_idx]]
    return flipped_data

class OpenPoseDataset(Dataset):
    def __init__(self, json_path, clip_len=243, vid_size=(1920, 1080), scale_range=None, focus=None, num_joints=17):
        self.json_path = json_path
        self.clip_len = clip_len
        self.vid_size = vid_size
        self.scale_range = scale_range
        self.focus = focus
        self.num_joints = num_joints
        self.joint_order = [
            'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
            'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye',
            'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel'
        ] if num_joints == 25 else [
            'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
            'Neck', 'Nose', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist'
        ]
        self.data = self.load_openpose_json()
        self.clips = self.prepare_clips()
    
    def load_openpose_json(self):
        json_files = sorted([f for f in os.listdir(self.json_path) if f.endswith('.json')])
        data = []
        for json_file in json_files:
            with open(os.path.join(self.json_path, json_file), 'r') as f:
                json_data = json.load(f)
                if not json_data['people']:
                    keypoints = np.zeros((self.num_joints, 3))
                else:
                    person_idx = self.focus if self.focus is not None else 0
                    if person_idx >= len(json_data['people']):
                        keypoints = np.zeros((self.num_joints, 3))
                    else:
                        keypoints = json_data['people'][person_idx]['pose_keypoints_2d']
                        keypoints = np.array(keypoints).reshape(-1, 3)
                        if self.num_joints == 17:
                            filtered_keypoints = np.zeros((17, 3))
                            for i, joint in enumerate(self.joint_order):
                                idx = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
                                       'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle'].index(joint)
                                filtered_keypoints[i] = keypoints[idx]
                            keypoints = filtered_keypoints
                data.append(keypoints)
        data = np.array(data)
        root_idx = self.joint_order.index('MidHip')
        data[:, :, :2] -= data[:, [root_idx], :2]
        if self.scale_range:
            max_dist = np.max(np.abs(data[:, :, :2]))
            if max_dist > 0:
                data[:, :, :2] /= max_dist
        else:
            data[:, :, :2] /= min(self.vid_size)
        return data
    
    def prepare_clips(self):
        clips = []
        for i in range(0, len(self.data), self.clip_len):
            clip = self.data[i:i + self.clip_len]
            if len(clip) < self.clip_len:
                clip = np.pad(clip, ((0, self.clip_len - len(clip)), (0, 0), (0, 0)), mode='edge')
            clips.append(clip)
        return np.array(clips)
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        return torch.tensor(self.clips[idx], dtype=torch.float32)

def main():
    opts = parse_args()
    set_random_seed()
    args = get_config(opts.config)
    
    os.makedirs(opts.out_path, exist_ok=True)
    
    vid = imageio.get_reader(opts.vid_path, 'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    
    dataset = OpenPoseDataset(
        json_path=opts.json_path,
        clip_len=opts.clip_len,
        vid_size=vid_size,
        scale_range=[1, 1] if not opts.pixel else None,
        focus=opts.focus,
        num_joints=opts.num_joints
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=False
    )
    
    model_pos = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=args.dim_feat if hasattr(args, 'dim_feat') else 256,
        dim_rep=args.dim_rep if hasattr(args, 'dim_rep') else 512,
        depth=args.depth if hasattr(args, 'depth') else 5,
        num_heads=args.num_heads if hasattr(args, 'num_heads') else 8,
        mlp_ratio=args.mlp_ratio if hasattr(args, 'mlp_ratio') else 4,
        num_joints=opts.num_joints,
        maxlen=opts.clip_len,
        qkv_bias=True,
        drop_rate=args.drop_rate if hasattr(args, 'drop_rate') else 0.,
        attn_drop_rate=args.attn_drop_rate if hasattr(args, 'attn_drop_rate') else 0.,
        drop_path_rate=args.drop_path_rate if hasattr(args, 'drop_path_rate') else 0.
    )
    
    print(f'Loading checkpoint {opts.evaluate}')
    checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
    
    # Strip 'module.' prefix from state_dict keys
    state_dict = checkpoint['model_pos'] if 'model_pos' in checkpoint else checkpoint
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        # Rename MLP keys to match model
        new_key = new_key.replace('mlp_s.fc1', 'mlp_s.0').replace('mlp_s.fc2', 'mlp_s.3')
        new_key = new_key.replace('mlp_t.fc1', 'mlp_t.0').replace('mlp_t.fc2', 'mlp_t.3')
        new_state_dict[new_key] = value
    
    try:
        model_pos.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: {e}. Loading with strict=False.")
        model_pos.load_state_dict(new_state_dict, strict=False)
    
    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()
    
    model_pos.eval()
    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:
                batch_input_flip = flip_data(batch_input, opts.num_joints)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip, opts.num_joints)
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0
            else:
                predicted_3d_pos[:, 0, 0, 2] = 0
            if args.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]
            results_all.append(predicted_3d_pos.cpu().numpy())
    
    results_all = np.concatenate(results_all, axis=1)
    # Remove batch dimension
    results_all = np.squeeze(results_all, axis=0)
    
    joint_order = [
        'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
        'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye',
        'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel'
    ] if opts.num_joints == 25 else [
        'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Neck', 'Nose', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist'
    ]
    output_data = []
    for frame_idx in range(results_all.shape[0]):
        for joint_idx, joint_name in enumerate(joint_order):
            output_data.append({
                'Frame': frame_idx,
                'Keypoint': joint_name,
                'X_3D': results_all[frame_idx, joint_idx, 0],
                'Y_3D': results_all[frame_idx, joint_idx, 1],
                'Z_3D': results_all[frame_idx, joint_idx, 2]
            })
    output_df = pd.DataFrame(output_data)
    output_df.to_excel(os.path.join(opts.out_path, "predicted_3d_poses.xlsx"), index=False)
    
    if opts.pixel:
        results_all = results_all * (min(vid_size) / 2.0)
        results_all[:, :, :2] = results_all[:, :, :2] + np.array(vid_size) / 2.0
    
    np.save(os.path.join(opts.out_path, 'X3D.npy'), results_all)
    render_and_save(results_all, os.path.join(opts.out_path, 'X3D.mp4'), keep_imgs=False, fps=fps_in)

if __name__ == "__main__":
    main()