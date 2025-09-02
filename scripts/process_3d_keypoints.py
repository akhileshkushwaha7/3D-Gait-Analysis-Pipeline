#python infer_wild.py --vid_path "C:\Users\akhileshsing2024\Downloads\GaitAnalysisProject\openpose\openpose\examples\media\120fps_rotated.mov" --json_path "C:\Users\akhileshsing2024\AlphaPose\output\alphapose-results.json" --out_path "C:\Users\akhileshsing2024\AlphaPose\results" 
#Motionbert .npy to excel
import numpy as np
import pandas as pd
import argparse

def main(npy_path, excel_path):
    data = np.load(npy_path)  # Shape: [frames, 17, 3]
    frames, joints, dims = data.shape
    columns = [f"Joint{j}_{dim}" for j in range(joints) for dim in ['X', 'Y', 'Z']]

    flattened_data = data.reshape(frames, joints * dims)
    df = pd.DataFrame(flattened_data, columns=columns)
    df.to_excel(excel_path, index=False)
    print(f"Excel file saved to: {excel_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MotionBERT .npy to Excel.")
    parser.add_argument("--npy", required=True, help="Input .npy path")
    parser.add_argument("--excel", required=True, help="Output Excel path")
    args = parser.parse_args()
    main(args.npy, args.excel)