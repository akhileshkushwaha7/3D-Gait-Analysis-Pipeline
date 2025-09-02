import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import argparse

def butter_lowpass_filter(data, cutoff=1, fs=120.0, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def main(excel_path):
    df = pd.read_excel(excel_path)

    # Filter for ankles (assuming Keypoint 15: Left Ankle, 16: Right Ankle)
    ankle_df = df[df['Keypoint'].isin([15, 16])]
    ankle_df['Keypoint'] = ankle_df['Keypoint'].map({15: 'Left Ankle', 16: 'Right Ankle'})

    pivot_x = ankle_df.pivot(index='Frame', columns='Keypoint', values='X')
    pivot_y = ankle_df.pivot(index='Frame', columns='Keypoint', values='Y')

    pivot_x_filtered = pivot_x.apply(butter_lowpass_filter)
    pivot_y_filtered = pivot_y.apply(butter_lowpass_filter)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(pivot_x_filtered.index, pivot_x_filtered['Left Ankle'], label='Left Ankle X (Filtered)')
    plt.plot(pivot_x_filtered.index, pivot_x_filtered['Right Ankle'], label='Right Ankle X (Filtered)')
    plt.xlabel('Frame')
    plt.ylabel('X Coordinate')
    plt.title('Filtered Ankle Joint X Coordinates')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(pivot_y_filtered.index, pivot_y_filtered['Left Ankle'], label='Left Ankle Y (Filtered)')
    plt.plot(pivot_y_filtered.index, pivot_y_filtered['Right Ankle'], label='Right Ankle Y (Filtered)')
    plt.xlabel('Frame')
    plt.ylabel('Y Coordinate')
    plt.title('Filtered Ankle Joint Y Coordinates')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot filtered keypoints from Excel.")
    parser.add_argument("--excel", required=True, help="Input Excel path")
    args = parser.parse_args()
    main(args.excel)