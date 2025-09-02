# 3D Gait Analysis Pipeline
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Active-green)

A modular pipeline for **2D and 3D human gait analysis**. This project integrates [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) for **2D pose estimation** and [MotionBERT](https://github.com/walter0807/MotionBERT) for **3D motion reconstruction**, with scripts for preprocessing, data conversion, and visualization.

---

## Techniques

- **Git Submodules**: Manages dependencies like [AlphaPose](AlphaPose/) and [MotionBERT](MotionBERT/) as submodules ([Git documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules)).  
- **Virtual Environments**: Uses [venv](https://docs.python.org/3/library/venv.html) for isolated Python environments.  
- **Video Processing**: Handles rotation of portrait videos programmatically before analysis.  
- **2D Pose Extraction**: Uses AlphaPose to detect keypoints in each frame.  
- **Data Transformation**: Converts JSON keypoints to Excel using [pandas](https://pandas.pydata.org/).  
- **Signal Filtering**: Applies a [Butterworth filter](https://en.wikipedia.org/wiki/Butterworth_filter) for smoothing ankle trajectories.  
- **3D Pose Reconstruction**: Uses MotionBERT to lift 2D keypoints into 3D motion representations.  
- **Visualization**: Generates plots of keypoint trajectories using [Matplotlib](https://matplotlib.org/) with the [DejaVu Sans font](https://dejavu-fonts.github.io/).  

---

## Non-Obvious Technologies

- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) – real-time multi-person pose estimation.  
- [MotionBERT](https://github.com/walter0807/MotionBERT) – transformer-based motion reconstruction.  
- [NumPy](https://numpy.org/) – efficient array operations and `.npy` support.  
- [pandas](https://pandas.pydata.org/) – conversion of JSON and `.npy` to Excel.  
- [Matplotlib](https://matplotlib.org/) – trajectory visualization.  

---

## Project Structure

```bash
├── scripts/
│   └── Custom scripts for rotation, processing, and visualization
├── output/
│   ├── alphapose-results/     # JSON and video outputs from AlphaPose
│   └── results/               # MotionBERT 3D outputs
├── pretrained_models/         # AlphaPose checkpoints
├── configs/                   # AlphaPose model configurations
├── images/                    # Example figures and visualizations
├── AlphaPose/                 # AlphaPose submodule
├── MotionBERT/                # MotionBERT submodule
├── requirements.txt
└── README.md
```
## Project Structure

- **scripts/**: Pipeline automation scripts.  
- **output/**: Stores all generated files.  
- **images/**: Static example plots.  
- **AlphaPose/** & **MotionBERT/**: Submodules containing core libraries.  

---

## Pipeline Overview

1. **Video Rotation** – Ensures portrait videos have consistent orientation.  
2. **2D Pose Estimation** – Extracts 2D keypoints using AlphaPose.  
3. **Data Conversion** – JSON to Excel and filtering of ankle trajectories.  
4. **3D Reconstruction** – Converts 2D keypoints to 3D using MotionBERT.  
5. **Excel Output** – Saves 3D keypoints for further analysis.  

---

## Related Papers

- **AlphaPose**: *Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time*  
- **MotionBERT**: *A Unified Perspective on Learning Human Motion Representations*  

---

## Contributing

1. Fork the repository.  
2. Create a branch:  
   ```bash
   git checkout -b feature-branch
3. git commit -m "Add feature"
4. git push origin feature-branch

---

# License  

This work is licensed under the MIT License © 2025 Akhilesh Singh.  
See [LICENSE](LICENSE) for details. Third-party datasets are subject to their respective licenses.  
If you use our code/models in your research, please cite this repository.  


