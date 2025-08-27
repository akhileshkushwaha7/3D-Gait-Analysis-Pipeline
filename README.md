# 3D Gait Analysis Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible pipeline for 3D gait analysis using AlphaPose for 2D pose estimation and MotionBERT for 3D motion reconstruction from videos. Handles video rotation, pose estimation, keypoint plotting with filtering, and 3D conversion. Designed for gait lab videos (e.g., ankle tracking) at 1080x1920 resolution.

## Features
- Video preprocessing (90° counterclockwise rotation).
- 2D pose estimation with AlphaPose.
- JSON output processing to Excel.
- Plotting filtered ankle keypoints (Butterworth low-pass).
- 3D keypoint conversion from MotionBERT .npy to Excel.
- Modular scripts for automation.

## Installation
1. Clone this repo: git clone https://github.com/yourusername/3D-Gait-Analysis-Pipeline.git
cd 3D-Gait-Analysis-Pipeline
2. Initialize submodules for AlphaPose and MotionBERT:
3. Create virtual environment
4. Install dependencies from requirements.txt
5. Follow setup in alphapose/README.md and motionbert/README.md.

## Scripts
1. Rotate video (Make sure it is in 1080X1920 resolution
2. Run AlphaPose ( Env Setup : set PYTHONPATH=%PYTHONPATH%;C:\Users\akhileshsing2024\AlphaPose
Run Alphapose : python scripts\demo_inference.py --cfg configs\coco\resnet\256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models\fast_res50_256x192.pth --video "C:\Users\akhileshsing2024\Downloads\GaitAnalysisProject\openpose\openpose\examples\media\120FPS_short_rotated.mov" --outdir output\ --save_video --pose_track) 
3. Process AlphaPose JSON to Excel
4. Plot keypoints for Alphapose(just to verfiy)
5. Run Motionbert ( python infer_wild.py --vid_path "C:\Users\akhileshsing2024\Downloads\GaitAnalysisProject\openpose\openpose\examples\media\120fps_rotated.mov" --json_path "C:\Users\akhileshsing2024\AlphaPose\output\alphapose-results.json" --out_path "C:\Users\akhileshsing2024\AlphaPose\results" )
6. Process MotionBERT 3D .npy to Excel

   
## Pipeline Overview
1. Rotate portrait video.
2. Estimate 2D poses with AlphaPose.
3. Convert JSON to Excel and plot filtered ankles.
4. (Optional) Run MotionBERT inference (see motionbert/README.md).
5. Convert 3D .npy to Excel.

## Related Papers
- [AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time](docs/alphapose_summary.md)
- [MotionBERT: A Unified Perspective on Learning Human Motion Representations](docs/motionbert_summary.md)


## Contributing
Fork and PR. See issues for bugs.

## Acknowledgments
- AlphaPose: MVIG-SJTU
- MotionBERT: Walter0807
- Date: August 27, 2025

  MIT License

Copyright (c) 2025 Akhilesh Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
