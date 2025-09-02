import numpy as np
import pandas as pd
import argparse

# -------------------------------
# Conversion Function
# -------------------------------
def convert_to_meters(input_file, output_file, height=1.7, 
                      y_min=-0.00259, y_max=0.92198, 
                      depth_min=7.936, depth_max=2.519):
    """
    Convert normalized MotionBERT joint coordinates into real-world meters.
    """

    # Joint names (consistent with MotionBERT output order)
    joint_names = [
        "Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot",
        "Spine", "Thorax", "Neck", "Head", "LShoulder", "LElbow",
        "LWrist", "RShoulder", "RElbow", "RWrist"
    ]

    # Example Z difference (Head to Hip); adjust if needed
    z_diff = abs(0.193932 - (-0.062291108071804))
    scaling_factor = height / z_diff

    y_range = y_max - y_min
    depth_range = depth_max - depth_min

    # Load input Excel
    df = pd.read_excel(input_file, sheet_name="Sheet1")
    print(f"Number of frames in input: {len(df)}")

    # Prepare output DataFrame
    output_columns = []
    for joint in joint_names:
        output_columns.extend([f"{joint}_X_m", f"{joint}_Y_m", f"{joint}_Z_m"])
    output_df = pd.DataFrame(index=df.index, columns=output_columns)

    # Convert each frame
    for idx in df.index:
        for joint_idx, joint in enumerate(joint_names):
            x_n = df.at[idx, f"Joint{joint_idx}_X"]
            y_n = df.at[idx, f"Joint{joint_idx}_Y"]
            z_n = df.at[idx, f"Joint{joint_idx}_Z"]

            if any(pd.isna(val) for val in [x_n, y_n, z_n]):
                x_real = y_real = z_real = float('nan')
            else:
                x_real = x_n * scaling_factor
                y_real = depth_min + ((y_n - y_min) / y_range) * depth_range
                z_real = z_n * scaling_factor

            output_df.at[idx, f"{joint}_X_m"] = round(x_real, 3)
            output_df.at[idx, f"{joint}_Y_m"] = round(y_real, 3)
            output_df.at[idx, f"{joint}_Z_m"] = round(z_real, 3)

    # Save result
    output_df.to_excel(output_file, index=False)
    print(f"Converted coordinates saved to {output_file}")

    # Print first frame for verification
    print("\nFirst frame converted coordinates (meters):")
    for joint in joint_names:
        x = output_df.at[0, f"{joint}_X_m"]
        y = output_df.at[0, f"{joint}_Y_m"]
        z = output_df.at[0, f"{joint}_Z_m"]
        print(f"{joint}: ({x:.3f}, {y:.3f}, {z:.3f})")

# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MotionBERT 3D keypoints to meters")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--output", required=True, help="Path to save converted Excel file")
    parser.add_argument("--height", type=float, default=1.7, help="Subject height in meters (default: 1.7)")
    parser.add_argument("--y_min", type=float, default=-0.00259, help="Min normalized Y (default: -0.00259)")
    parser.add_argument("--y_max", type=float, default=0.92198, help="Max normalized Y (default: 0.92198)")
    parser.add_argument("--depth_min", type=float, default=7.936, help="Minimum depth in meters (default: 7.936)")
    parser.add_argument("--depth_max", type=float, default=2.519, help="Maximum depth in meters (default: 2.519)")

    args = parser.parse_args()
    convert_to_meters(args.input, args.output, args.height, args.y_min, args.y_max, args.depth_min, args.depth_max)
