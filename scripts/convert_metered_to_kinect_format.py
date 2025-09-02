import pandas as pd
import numpy as np
import os
import argparse

# -----------------------------------
# Kinect Joint Mapping
# -----------------------------------
JOINT_MAPPING = {
    "Hip": 0,        # Pelvis
    "RHip": 22,      # Hip Right
    "RKnee": 23,     # Knee Right
    "RFoot": 25,     # Foot Right
    "LHip": 18,      # Hip Left
    "LKnee": 19,     # Knee Left
    "LFoot": 21,     # Foot Left
    "Spine": 1,      # Spine Naval
    "Thorax": 2,     # Spine Chest
    "Neck": 3,       # Neck
    "Head": 26,      # Head
    "LShoulder": 5,  # Shoulder Left
    "LElbow": 6,     # Elbow Left
    "LWrist": 7,     # Wrist Left
    "RShoulder": 12, # Shoulder Right
    "RElbow": 13,    # Elbow Right
    "RWrist": 14     # Wrist Right
}

# -----------------------------------
# Conversion Function
# -----------------------------------
def convert_excel_to_kinect_csv(input_file, output_folder, frame_rate=120):
    """
    Convert 3D motion data (in meters) to Kinect-style CSV (in millimeters).
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate output file path
    output_file = os.path.join(
        output_folder,
        os.path.splitext(os.path.basename(input_file))[0] + "_kinect.csv"
    )

    # Load Excel file
    df = pd.read_excel(input_file)

    # Compute timestamps
    num_frames = len(df)
    timestamps = np.linspace(0, num_frames / frame_rate, num_frames)

    # Prepare output list
    output_data = []

    for frame_idx, row in df.iterrows():
        timestamp = timestamps[frame_idx]
        person_id = row.get("PersonID", 0)  # default to 0 if not present

        for joint, azure_idx in JOINT_MAPPING.items():
            # Convert from meters to millimeters
            pos_x = row[f"{joint}_X_m"] * 1000
            pos_y = row[f"{joint}_Y_m"] * 1000
            pos_z = row[f"{joint}_Z_m"] * 1000

            output_data.append({
                "Timestamp": timestamp,
                "BodyID": person_id,
                "Joint_": azure_idx,
                "Position_x_": pos_x,
                "Position_y_": pos_z,  # Kinect coordinate swap
                "Position_z_": pos_y
            })

    # Save as CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)

    print(f"Converted data saved to {output_file}")

# -----------------------------------
# CLI Entry Point
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Excel 3D joints to Kinect-style CSV")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--output_folder", required=True, help="Folder to save the CSV")
    parser.add_argument("--fps", type=int, default=120, help="Frame rate (default: 120)")

    args = parser.parse_args()
    convert_excel_to_kinect_csv(args.input, args.output_folder, args.fps)
