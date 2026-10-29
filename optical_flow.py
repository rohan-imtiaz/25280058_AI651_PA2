import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from utils import load_videos, split_videos

VIDEO_COUNT = 50
OUTPUT_DIR = "figures"


def direction_fractions(angles):
    counts, _ = np.histogram(angles, bins = 18, range = (0, 360))
    return counts / counts.sum()


def main():
    videos = load_videos()
    train_videos, _, _ = split_videos(videos)
    frames_all = train_videos[:VIDEO_COUNT, :, 0].numpy() # (50, 20, 64, 64) uint8, equal to the float frames times 255

    angles_list = []
    speeds_list = []
    for video_index in range(VIDEO_COUNT):
        for step in range(19):
            previous = np.ascontiguousarray(frames_all[video_index, step])
            following = np.ascontiguousarray(frames_all[video_index, step + 1])
            flow = cv2.calcOpticalFlowFarneback(previous, following, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_x = flow[..., 0] # Along columns
            flow_y = flow[..., 1] # Along rows, increasing downward
            magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)
            kept = (previous > 51) & (following > 51) & (magnitude > 0.5) # Intensity above 0.2 in both frames
            angles_list.append(np.degrees(np.arctan2(-flow_y[kept], flow_x[kept])) % 360)
            speeds_list.append(magnitude[kept])
    angles = np.concatenate(angles_list)
    speeds = np.concatenate(speeds_list)
    median_speed = np.median(speeds)

    os.makedirs(OUTPUT_DIR, exist_ok = True)
    figure, axes = plt.subplots(1, 2, figsize = (11, 4))
    fractions = direction_fractions(angles)
    axes[0].bar(np.arange(18) * 20 + 10, fractions, width = 20, edgecolor = "black")
    axes[0].axhline(1 / 18, color = "red", linestyle = "--", label = "Uniform (1/18)")
    axes[0].set_xlabel("Flow direction (degrees, 0 = right, 90 = up)")
    axes[0].set_ylabel("Fraction of kept pixels")
    axes[0].set_xticks(np.arange(0, 361, 60))
    axes[0].legend()
    axes[0].set_title("(a) Direction")
    speed_counts, speed_edges = np.histogram(speeds, bins = 40)
    axes[1].bar(speed_edges[:-1], speed_counts / speed_counts.sum(), width = np.diff(speed_edges), align = "edge", edgecolor = "black")
    axes[1].axvline(median_speed, color = "red", linestyle = "--", label = "Median " + "{:.3f}".format(median_speed))
    axes[1].set_xlabel("Flow speed (pixels per frame)")
    axes[1].set_ylabel("Fraction of kept pixels")
    axes[1].legend()
    axes[1].set_title("(b) Speed")
    figure.tight_layout()
    figure.savefig(os.path.join(OUTPUT_DIR, "figure_1.png"), dpi = 200)

    print("Caption numbers: median speed", "{:.3f}".format(median_speed), "pixels per frame, kept pixels", len(speeds))
    print("Speed percentiles (5, 25, 50, 75, 95):", np.percentile(speeds, [5, 25, 50, 75, 95]).round(3).tolist())
    print("Direction fractions per 20 degree bin:", fractions.round(4).tolist())
    # For Q2.1: how far each histogram moves when every video is mirrored or rotated, 0 means identical (total variation distance)
    mirrored_angles = (180 - angles) % 360
    rotated_angles = (angles + 90) % 360
    print("Total variation, mirrored vs original:", round(0.5 * np.abs(direction_fractions(mirrored_angles) - fractions).sum(), 4))
    print("Total variation, rotated vs original:", round(0.5 * np.abs(direction_fractions(rotated_angles) - fractions).sum(), 4))
    print("Total variation, original vs uniform:", round(0.5 * np.abs(fractions - 1 / 18).sum(), 4))


if __name__ == "__main__":
    main()