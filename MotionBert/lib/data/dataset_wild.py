# import torch
# import numpy as np
# import ipdb
# import glob
# import os
# import io
# import math
# import random
# import json
# import pickle
# import math
# from torch.utils.data import Dataset, DataLoader
# from lib.utils.utils_data import crop_scale

# def halpe2h36m(x):
#     '''
#         Input: x (T x V x C)  
#        //Halpe 26 body keypoints
#     {0,  "Nose"},
#     {1,  "LEye"},
#     {2,  "REye"},
#     {3,  "LEar"},
#     {4,  "REar"},
#     {5,  "LShoulder"},
#     {6,  "RShoulder"},
#     {7,  "LElbow"},
#     {8,  "RElbow"},
#     {9,  "LWrist"},
#     {10, "RWrist"},
#     {11, "LHip"},
#     {12, "RHip"},
#     {13, "LKnee"},
#     {14, "Rknee"},
#     {15, "LAnkle"},
#     {16, "RAnkle"},
#     {17,  "Head"},
#     {18,  "Neck"},
#     {19,  "Hip"},
#     {20, "LBigToe"},
#     {21, "RBigToe"},
#     {22, "LSmallToe"},
#     {23, "RSmallToe"},
#     {24, "LHeel"},
#     {25, "RHeel"},
#     '''
#     T, V, C = x.shape
#     y = np.zeros([T,17,C])
#     y[:,0,:] = x[:,19,:]
#     y[:,1,:] = x[:,12,:]
#     y[:,2,:] = x[:,14,:]
#     y[:,3,:] = x[:,16,:]
#     y[:,4,:] = x[:,11,:]
#     y[:,5,:] = x[:,13,:]
#     y[:,6,:] = x[:,15,:]
#     y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
#     y[:,8,:] = x[:,18,:]
#     y[:,9,:] = x[:,0,:]
#     y[:,10,:] = x[:,17,:]
#     y[:,11,:] = x[:,5,:]
#     y[:,12,:] = x[:,7,:]
#     y[:,13,:] = x[:,9,:]
#     y[:,14,:] = x[:,6,:]
#     y[:,15,:] = x[:,8,:]
#     y[:,16,:] = x[:,10,:]
#     return y
    
# def read_input(json_path, vid_size, scale_range, focus):
#     with open(json_path, "r") as read_file:
#         results = json.load(read_file)
#     kpts_all = []
#     for item in results:
#         if focus!=None and item['idx']!=focus:
#             continue
#         kpts = np.array(item['keypoints']).reshape([-1,3])
#         kpts_all.append(kpts)
#     kpts_all = np.array(kpts_all)
#     kpts_all = halpe2h36m(kpts_all)
#     if vid_size:
#         w, h = vid_size
#         scale = min(w,h) / 2.0
#         kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([w, h]) / 2.0
#         kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
#         motion = kpts_all
#     if scale_range:
#         motion = crop_scale(kpts_all, scale_range) 
#     return motion.astype(np.float32)

# class WildDetDataset(Dataset):
#     def __init__(self, json_path, clip_len=243, vid_size=None, scale_range=None, focus=None):
#         self.json_path = json_path
#         self.clip_len = clip_len
#         self.vid_all = read_input(json_path, vid_size, scale_range, focus)
        
#     def __len__(self):
#         'Denotes the total number of samples'
#         return math.ceil(len(self.vid_all) / self.clip_len)
    
#     def __getitem__(self, index):
#         'Generates one sample of data'
#         st = index*self.clip_len
#         end = min((index+1)*self.clip_len, len(self.vid_all))
#         return self.vid_all[st:end]
#----------------2------------------------

# import os
# import json
# import numpy as np
# import cv2
# import math
# from torch.utils.data import Dataset
# from lib.utils.utils_data import crop_scale

# def coco2h36m(x):
#     '''
#         Input: x (M x T x V x C)
        
#         COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
#         H36M:
#         0: 'root', 1: 'rhip', 2: 'rkne', 3: 'rank', 4: 'lhip', 5: 'lkne', 6: 'lank',
#         7: 'belly', 8: 'neck', 9: 'nose', 10: 'head', 11: 'lsho', 12: 'lelb', 13: 'lwri',
#         14: 'rsho', 15: 'relb', 16: 'rwri'
#     '''
#     y = np.zeros(x.shape)
#     y[:, :, 0, :] = (x[:, :, 11, :] + x[:, :, 12, :]) * 0.5  # root (LHip + RHip) / 2
#     y[:, :, 1, :] = x[:, :, 12, :]  # rhip (RHip)
#     y[:, :, 2, :] = x[:, :, 14, :]  # rkne (RKnee)
#     y[:, :, 3, :] = x[:, :, 16, :]  # rank (RAnkle)
#     y[:, :, 4, :] = x[:, :, 11, :]  # lhip (LHip)
#     y[:, :, 5, :] = x[:, :, 13, :]  # lkne (LKnee)
#     y[:, :, 6, :] = x[:, :, 15, :]  # lank (LAnkle)
#     y[:, :, 8, :] = (x[:, :, 5, :] + x[:, :, 6, :]) * 0.5  # neck (LShoulder + RShoulder) / 2
#     y[:, :, 7, :] = (y[:, :, 0, :] + y[:, :, 8, :]) * 0.5  # belly (root + neck) / 2
#     y[:, :, 9, :] = x[:, :, 0, :]  # nose (Nose)
#     y[:, :, 10, :] = (x[:, :, 1, :] + x[:, :, 2, :]) * 0.5  # head (Leye + Reye) / 2
#     y[:, :, 11, :] = x[:, :, 5, :]  # lsho (LShoulder)
#     y[:, :, 12, :] = x[:, :, 7, :]  # lelb (LElbow)
#     y[:, :, 13, :] = x[:, :, 9, :]  # lwri (LWrist)
#     y[:, :, 14, :] = x[:, :, 6, :]  # rsho (RShoulder)
#     y[:, :, 15, :] = x[:, :, 8, :]  # relb (RElbow)
#     y[:, :, 16, :] = x[:, :, 10, :]  # rwri (RWrist)
#     return y

# def read_input(json_path, vid_size, scale_range, focus):
#     """Read AlphaPose JSON (COCO format) and convert to H36M format for MotionBERT."""
#     if not os.path.exists(json_path):
#         raise FileNotFoundError(f"JSON file {json_path} not found")
    
#     with open(json_path, "r") as read_file:
#         results = json.load(read_file)
    
#     kpts_all = []
#     for item in results:
#         kpts = np.array(item['keypoints']).reshape(-1, 17, 3)  # COCO: [frames, 17, 3] (x, y, confidence)
#         kpts_all.append(kpts)
    
#     if not kpts_all:
#         raise ValueError(f"No valid keypoints found in {json_path}")
    
#     kpts_all = np.array(kpts_all)  # Shape: [frames, num_people, 17, 3]
    
#     # Select a single person
#     if focus is not None:
#         if focus < kpts_all.shape[1] and 'idx' in results[0] and results[0].get('idx', -1) == focus:
#             kpts_all = kpts_all[:, focus:focus+1, :, :]  # Select specified person
#         else:
#             raise ValueError(f"Focus ID {focus} not found in JSON or invalid")
#     else:
#         avg_scores = np.array([item.get('score', 0.0) for item in results]).mean(axis=0)
#         best_person = np.argmax(avg_scores) if kpts_all.shape[1] > 1 else 0
#         kpts_all = kpts_all[:, best_person:best_person+1, :, :]
    
#     kpts_all = kpts_all.squeeze(1)  # Shape: [frames, 17, 3]
    
#     # Reshape to match coco2h36m input (M x T x V x C) with M=1
#     kpts_all = kpts_all[:, None, :, :]  # Shape: [frames, 1, 17, 3]
    
#     # Convert COCO to H36M
#     mapped_kpts = coco2h36m(kpts_all)  # Shape: [frames, 1, 17, 3]
#     mapped_kpts = mapped_kpts.squeeze(1)  # Back to [frames, 17, 3]
    
#     # Normalize keypoints
#     if vid_size:
#         w, h = vid_size
#         scale = max(w, h)  # Use max dimension
#         mapped_kpts[:, :, [0, 1]] = mapped_kpts[:, :, [0, 1]] / scale  # Normalize to [0, 1]
#         mapped_kpts[:, :, [0, 1]] = mapped_kpts[:, :, [0, 1]] * 2 - 1  # Convert to [-1, 1]
#     if scale_range:
#         mapped_kpts = crop_scale(mapped_kpts, scale_range)
    
#     return mapped_kpts.astype(np.float32)

# class WildDetDataset(Dataset):
#     def __init__(self, json_path, clip_len=243, vid_size=(1080, 1920), scale_range=None, focus=None):
#         self.json_path = json_path
#         self.clip_len = clip_len  # Set to 243 to match MotionBERT's max input length
#         self.vid_all = read_input(json_path, vid_size, scale_range, focus)
        
#     def __len__(self):
#         """Denotes the total number of samples."""
#         return math.ceil(len(self.vid_all) / self.clip_len)
    
#     def __getitem__(self, index):
#         """Generates one sample of data."""
#         st = index * self.clip_len
#         end = min((index + 1) * self.clip_len, len(self.vid_all))
#         data = self.vid_all[st:end]  # Shape: [F, 17, 3], F <= 243
#         if data.ndim == 3:
#             return data
#         else:
#             raise ValueError(f"Unexpected data shape: {data.shape}, expected [F, 17, 3]")

import os
import json
import numpy as np
import cv2
import math
from torch.utils.data import Dataset
from lib.utils.utils_data import crop_scale

def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4-Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root', 1: 'rhip', 2: 'rkne', 3: 'rank', 4: 'lhip', 5: 'lkne', 6: 'lank',
        7: 'belly', 8: 'neck', 9: 'nose', 10: 'head', 11: 'lsho', 12: 'lelb', 13: 'lwri',
        14: 'rsho', 15: 'relb', 16: 'rwri'

        1: 'root', 2: 'rhip', 3: 'rkne', 4: 'rank', 5: 'lhip', 6: 'lkne', 7: 'lank',
        8: 'belly', 9: 'neck', 10: 'nose', 11: 'head', 12: 'lsho', 13: 'lelb', 14: 'lwri',
        15: 'rsho', 16: 'relb', 17: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:, :, 0, :] = (x[:, :, 11, :] + x[:, :, 12, :]) * 0.5  # root (LHip + RHip) / 2
    y[:, :, 1, :] = x[:, :, 12, :]  # rhip (RHip)
    y[:, :, 2, :] = x[:, :, 14, :]  # rkne (RKnee)
    y[:, :, 3, :] = x[:, :, 16, :]  # rank (RAnkle)
    y[:, :, 4, :] = x[:, :, 11, :]  # lhip (LHip)
    y[:, :, 5, :] = x[:, :, 13, :]  # lkne (LKnee)
    y[:, :, 6, :] = x[:, :, 15, :]  # lank (LAnkle)
    y[:, :, 8, :] = (x[:, :, 5, :] + x[:, :, 6, :]) * 0.5  # neck (LShoulder + RShoulder) / 2
    y[:, :, 7, :] = (y[:, :, 0, :] + y[:, :, 8, :]) * 0.5  # belly (root + neck) / 2
    y[:, :, 9, :] = x[:, :, 0, :]  # nose (Nose)
    y[:, :, 10, :] = (x[:, :, 1, :] + x[:, :, 2, :]) * 0.5  # head (Leye + Reye) / 2
    y[:, :, 11, :] = x[:, :, 5, :]  # lsho (LShoulder)
    y[:, :, 12, :] = x[:, :, 7, :]  # lelb (LElbow)
    y[:, :, 13, :] = x[:, :, 9, :]  # lwri (LWrist)
    y[:, :, 14, :] = x[:, :, 6, :]  # rsho (RShoulder)
    y[:, :, 15, :] = x[:, :, 8, :]  # relb (RElbow)
    y[:, :, 16, :] = x[:, :, 10, :]  # rwri (RWrist)
    return y

def read_input(json_path, vid_size, scale_range, focus):
    """Read AlphaPose JSON (COCO format) and convert to H36M format for MotionBERT."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file {json_path} not found")
    
    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    
    kpts_all = []
    for item in results:
        kpts = np.array(item['keypoints']).reshape(-1, 17, 3)  # COCO: [frames, 17, 3] (x, y, confidence)
        kpts_all.append(kpts)
    
    if not kpts_all:
        raise ValueError(f"No valid keypoints found in {json_path}")
    
    kpts_all = np.array(kpts_all)  # Shape: [frames, num_people, 17, 3]
    
    # Select a single person
    if focus is not None:
        if focus < kpts_all.shape[1] and 'idx' in results[0] and results[0].get('idx', -1) == focus:
            kpts_all = kpts_all[:, focus:focus+1, :, :]  # Select specified person
        else:
            raise ValueError(f"Focus ID {focus} not found in JSON or invalid")
    else:
        avg_scores = np.array([item.get('score', 0.0) for item in results]).mean(axis=0)
        best_person = np.argmax(avg_scores) if kpts_all.shape[1] > 1 else 0
        kpts_all = kpts_all[:, best_person:best_person+1, :, :]
    
    kpts_all = kpts_all.squeeze(1)  # Shape: [frames, 17, 3]
    
    # Reshape to match coco2h36m input (M x T x V x C) with M=1
    kpts_all = kpts_all[:, None, :, :]  # Shape: [frames, 1, 17, 3]
    
    # Convert COCO to H36M
    mapped_kpts = coco2h36m(kpts_all)  # Shape: [frames, 1, 17, 3]
    mapped_kpts = mapped_kpts.squeeze(1)  # Back to [frames, 17, 3]
    
    # Apply crop_scale if scale_range is provided
    if scale_range:
        mapped_kpts = crop_scale(mapped_kpts, scale_range)
    
    return mapped_kpts.astype(np.float32)

class WildDetDataset(Dataset):
    def __init__(self, json_path, clip_len=243, vid_size=(1080, 1920), scale_range=None, focus=None):
        self.json_path = json_path
        self.clip_len = clip_len  # Set to 243 to match MotionBERT's max input length
        self.vid_all = read_input(json_path, vid_size, scale_range, focus)
        
    def __len__(self):
        """Denotes the total number of samples."""
        return math.ceil(len(self.vid_all) / self.clip_len)
    
    def __getitem__(self, index):
        """Generates one sample of data."""
        st = index * self.clip_len
        end = min((index + 1) * self.clip_len, len(self.vid_all))
        data = self.vid_all[st:end]  # Shape: [F, 17, 3], F <= 243
        if data.ndim == 3:
            return data
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}, expected [F, 17, 3]")