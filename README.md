# 3D Gait Analysis Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible pipeline for **3D gait analysis** using [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) for 2D pose estimation and [MotionBERT](https://github.com/Walter0807/MotionBERT) for 3D motion reconstruction from videos.  

The pipeline supports video rotation, pose estimation, JSON-to-Excel conversion, keypoint filtering and plotting, and 3D motion reconstruction. It is designed for gait lab videos (e.g., ankle tracking) at **1080x1920** resolution.

---

## Features
- ✅ Video preprocessing (90° counterclockwise rotation)  
- ✅ 2D pose estimation with AlphaPose  
- ✅ JSON output processing → Excel conversion  
- ✅ Filtered ankle keypoint plotting (Butterworth low-pass)  
- ✅ 3D keypoint reconstruction with MotionBERT (`.npy → Excel`)  
- ✅ Modular scripts for automation  

---

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/yourusername/3D-Gait-Analysis-Pipeline.git
   cd 3D-Gait-Analysis-Pipeline
Initialize submodules (for AlphaPose and MotionBERT)

bash
Copy code
git submodule update --init --recursive
Create a virtual environment

bash
Copy code
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies

bash
Copy code
pip install -r requirements.txt
Follow additional setup in:

alphapose/README.md

motionbert/README.md

Usage
1. Rotate video
Ensure your input video is 1080x1920 portrait resolution.

2. Run AlphaPose
Set environment variable:

bash
Copy code
set PYTHONPATH=%PYTHONPATH%;C:\Users\akhileshsing2024\AlphaPose
Run inference:

bash
Copy code
python scripts\demo_inference.py ^
  --cfg configs\coco\resnet\256x192_res50_lr1e-3_1x.yaml ^
  --checkpoint pretrained_models\fast_res50_256x192.pth ^
  --video "C:\path\to\your\rotated_video.mov" ^
  --outdir output\ ^
  --save_video ^
  --pose_track
3. Convert AlphaPose JSON → Excel
bash
Copy code
python scripts\process_alphapose_json.py
4. Plot keypoints (verification)
bash
Copy code
python scripts\plot_keypoints.py
5. Run MotionBERT
bash
Copy code
python infer_wild.py ^
  --vid_path "C:\path\to\your\rotated_video.mov" ^
  --json_path "C:\Users\akhileshsing2024\AlphaPose\output\alphapose-results.json" ^
  --out_path "C:\Users\akhileshsing2024\AlphaPose\results"
6. Convert MotionBERT 3D .npy → Excel
bash
Copy code
python scripts\process_motionbert_npy.py
Pipeline Overview
Rotate portrait video

Estimate 2D poses with AlphaPose

Convert JSON → Excel & filter ankle trajectories

(Optional) Run MotionBERT for 3D motion reconstruction

Convert 3D .npy → Excel

Related Papers
AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time

MotionBERT: A Unified Perspective on Learning Human Motion Representations

Contributing
Contributions are welcome!

Fork the repo

Create a new branch

Submit a Pull Request

Please check the issues tab for open tasks or bugs.

Acknowledgments
AlphaPose: MVIG-SJTU

MotionBERT: Walter0807

License
MIT License © 2025 Akhilesh Singh

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

pgsql
Copy code
