
# set PYTHONPATH=%PYTHONPATH%;C:\Users\akhileshsing2024\AlphaPose
# python scripts\demo_inference.py --cfg configs\coco\resnet\256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models\fast_res50_256x192.pth --video "C:\Users\akhileshsing2024\Downloads\GaitAnalysisProject\openpose\openpose\examples\media\120FPS_short_rotated.mov" --outdir output\ --save_video --pose_track

import subprocess
import argparse

def main(video_path, outdir):
    # From your commented command; adjust as needed
    cmd = [
        "python", "scripts/demo_inference.py",
        "--cfg", "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml",
        "--checkpoint", "pretrained_models/fast_res50_256x192.pth",
        "--video", video_path,
        "--outdir", outdir,
        "--save_video",
        "--pose_track"
    ]
    subprocess.run(cmd, check=True)
    print(f"AlphaPose run complete. Outputs in {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AlphaPose on video.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--outdir", default="output/", help="Output directory")
    args = parser.parse_args()
    main(args.video, args.outdir)