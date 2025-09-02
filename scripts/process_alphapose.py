import json
import pandas as pd
import argparse

def convert_json_to_excel(json_file, output_excel):
    keypoint_names = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
    ]

    data = []

    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # json_data is a list of people dictionaries
    people = json_data

    for person_idx, person in enumerate(people):
        keypoints = person.get("keypoints", [])
        if not keypoints:
            continue

        for i in range(0, len(keypoints), 3):
            kp_id = i // 3
            x = keypoints[i]
            y = keypoints[i+1]
            confidence = keypoints[i+2]
            kp_name = keypoint_names[kp_id] if kp_id < len(keypoint_names) else f"Keypoint_{kp_id}"
            data.append({
                "Frame": person_idx + 1,
                "Keypoint": kp_name,
                "X": x,
                "Y": y,
                "Confidence": confidence
            })

    df = pd.DataFrame(data)
    df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f"Excel file saved to: {output_excel}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert AlphaPose JSON to Excel")
    parser.add_argument("--json", required=True, help="Path to AlphaPose JSON file")
    parser.add_argument("--out", required=True, help="Output Excel file path")

    args = parser.parse_args()
    convert_json_to_excel(args.json, args.out)
