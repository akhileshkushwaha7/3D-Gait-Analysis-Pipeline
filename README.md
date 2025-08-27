3D Gait Analysis Pipeline

A robust and reproducible pipeline for 3D gait analysis leveraging AlphaPose for 2D pose estimation and MotionBERT for 3D motion reconstruction from videos. This pipeline is optimized for gait lab videos (e.g., ankle tracking) at 1080x1920 resolution, supporting video preprocessing, pose estimation, data conversion, keypoint filtering, and 3D reconstruction.

Table of Contents

Features
Installation
Usage
Pipeline Overview
Related Papers
Contributing
Acknowledgments
License


Features

Video Preprocessing: Automatic 90° counterclockwise rotation for portrait videos.
2D Pose Estimation: Utilizes AlphaPose for accurate multi-person pose tracking.
Data Conversion: Converts AlphaPose JSON outputs to Excel for easy analysis.
Keypoint Filtering: Applies Butterworth low-pass filter for smooth ankle trajectory plotting.
3D Motion Reconstruction: Employs MotionBERT to reconstruct 3D keypoints from 2D poses.
Modular Automation: Scripts designed for seamless integration and scalability.


Installation
Follow these steps to set up the pipeline on your local machine.
Prerequisites

Python 3.8 or higher
Git
Virtualenv (recommended)

Steps

Clone the Repository
git clone https://github.com/yourusername/3D-Gait-Analysis-Pipeline.git
cd 3D-Gait-Analysis-Pipeline


Initialize Submodules
git submodule update --init --recursive


Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install Dependencies
pip install -r requirements.txt


Additional SetupRefer to the setup instructions in:

AlphaPose README
MotionBERT README




Usage
The pipeline processes videos through a series of modular scripts. Ensure all paths and configurations are correctly set before running.
1. Rotate Video
Ensure your input video is in 1080x1920 portrait resolution. Use the provided script to rotate the video 90° counterclockwise if needed.
2. Run AlphaPose for 2D Pose Estimation
Set the environment variable for AlphaPose:
set PYTHONPATH=%PYTHONPATH%;C:\Users\akhileshsing2024\AlphaPose

Run inference:
python scripts\demo_inference.py ^
  --cfg configs\coco\resnet\256x192_res50_lr1e-3_1x.yaml ^
  --checkpoint pretrained_models\fast_res50_256x192.pth ^
  --video "C:\path\to\your\rotated_video.mov" ^
  --outdir output\ ^
  --save_video ^
  --pose_track

3. Convert AlphaPose JSON to Excel
python scripts\process_alphapose_json.py

4. Plot Keypoints (Verification)
Visualize filtered ankle trajectories:
python scripts\plot_keypoints.py

5. Run MotionBERT for 3D Reconstruction
python infer_wild.py ^
  --vid_path "C:\path\to\your\rotated_video.mov" ^
  --json_path "C:\Users\akhileshsing2024\AlphaPose\output\alphapose-results.json" ^
  --out_path "C:\Users\akhileshsing2024\AlphaPose\results"

6. Convert MotionBERT 3D .npy to Excel
python scripts\process_motionbert_npy.py


Pipeline Overview

Video Rotation: Adjusts portrait videos for consistent orientation.
2D Pose Estimation: Extracts 2D keypoints using AlphaPose.
Data Processing: Converts JSON outputs to Excel and applies Butterworth filtering for ankle trajectories.
3D Reconstruction (Optional): Reconstructs 3D keypoints using MotionBERT.
Output Conversion: Saves 3D keypoint data as Excel for further analysis.


Related Papers

AlphaPose: "Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time"Paper
MotionBERT: "A Unified Perspective on Learning Human Motion Representations"Paper


Contributing
Contributions are highly encouraged! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Submit a Pull Request.

Check the Issues tab for open tasks or bugs.

Acknowledgments

AlphaPose: Developed by MVIG-SJTU
MotionBERT: Developed by Walter0807
Special thanks to the open-source community for their invaluable contributions.


License
MIT License © 2025 Akhilesh Singh
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
