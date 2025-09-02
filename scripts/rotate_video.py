import cv2
import argparse

def main(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Swapped for rotation
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out.write(rotated_frame)

    cap.release()
    out.release()
    print(f"Saved rotated video as: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotate video 90° counterclockwise.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    args = parser.parse_args()
    main(args.input, args.output)