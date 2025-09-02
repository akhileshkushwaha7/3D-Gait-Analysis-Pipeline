import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import argparse

# -------------------------------
# Butterworth Filter Functions
# -------------------------------
def butter_lowpass(cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# -------------------------------
# Main Plotting Function
# -------------------------------
def plot_keypoints(file_path, cutoff=2, order=5, fs=60):
    # Load Excel file
    df = pd.read_excel(file_path)

    # Extract Y-axis data for left and right foot
    lfoot_y = df['Joint3_Y']
    rfoot_y = df['Joint6_Y']

    # Apply Butterworth filter
    lfoot_y_filtered = apply_lowpass_filter(lfoot_y, cutoff, fs, order)
    rfoot_y_filtered = apply_lowpass_filter(rfoot_y, cutoff, fs, order)

    # Create subplots
    plt.figure(figsize=(12, 5))

    # Subplot 1: Original data
    plt.subplot(1, 2, 1)
    plt.plot(lfoot_y, label='Left Foot Y (Original)', color='blue', alpha=0.5)
    plt.plot(rfoot_y, label='Right Foot Y (Original)', color='red', alpha=0.5)
    plt.title('Original Data')
    plt.xlabel('Frame')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Filtered data
    plt.subplot(1, 2, 2)
    plt.plot(lfoot_y_filtered, label='Left Foot Y (Filtered)', color='blue')
    plt.plot(rfoot_y_filtered, label='Right Foot Y (Filtered)', color='red')
    plt.title('Filtered Data')
    plt.xlabel('Frame')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and filter gait keypoints from Excel data")
    parser.add_argument("--file", required=True, help="Path to Excel file containing MotionBERT keypoints")
    parser.add_argument("--cutoff", type=float, default=2, help="Cutoff frequency in Hz (default: 2)")
    parser.add_argument("--order", type=int, default=5, help="Filter order (default: 5)")
    parser.add_argument("--fs", type=float, default=60, help="Sampling rate in Hz (default: 60)")

    args = parser.parse_args()
    plot_keypoints(args.file, args.cutoff, args.order, args.fs)
