import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import argparse

# -------------------------------
# Butterworth Filter Functions
# -------------------------------
def butter_lowpass(cutoff, fs, order):
    """Design a Butterworth low-pass filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order):
    """Apply Butterworth filter to data."""
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# -------------------------------
# Main Function
# -------------------------------
def process_feet(input_file, cutoff=2.2, order=5, fs=120):
    """
    Apply low-pass Butterworth filter to left and right foot Y-coordinates
    and plot original vs filtered signals.
    """

    # Load Excel
    df = pd.read_excel(input_file)

    # Extract Y-axis data
    lfoot_y = df['LFoot_Y_m']
    rfoot_y = df['RFoot_Y_m']

    # Filter signals
    lfoot_y_filtered = apply_lowpass_filter(lfoot_y, cutoff, fs, order)
    rfoot_y_filtered = apply_lowpass_filter(rfoot_y, cutoff, fs, order)

    # Plot
    plt.figure(figsize=(12, 5))

    # Original
    plt.subplot(1, 2, 1)
    plt.plot(lfoot_y, label='Left Foot (Original)', color='blue', alpha=0.5, linestyle='--')
    plt.plot(rfoot_y, label='Right Foot (Original)', color='red', alpha=0.5, linestyle='--')
    plt.title('Original Data')
    plt.xlabel('Frame')
    plt.ylabel('Y Coordinate (m)')
    plt.legend()
    plt.grid(True)

    # Filtered
    plt.subplot(1, 2, 2)
    plt.plot(lfoot_y_filtered, label='Left Foot (Filtered)', color='blue')
    plt.plot(rfoot_y_filtered, label='Right Foot (Filtered)', color='red')
    plt.title('Filtered Data')
    plt.xlabel('Frame')
    plt.ylabel('Y Coordinate (m)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Butterworth filter for left/right foot Y-axis data")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--cutoff", type=float, default=2.2, help="Cutoff frequency in Hz (default: 2.2)")
    parser.add_argument("--order", type=int, default=5, help="Filter order (default: 5)")
    parser.add_argument("--fs", type=float, default=120, help="Sampling rate in Hz (default: 120)")

    args = parser.parse_args()
    process_feet(args.input, args.cutoff, args.order, args.fs)
