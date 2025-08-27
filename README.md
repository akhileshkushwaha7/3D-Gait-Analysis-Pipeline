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
1. Clone the repo:
