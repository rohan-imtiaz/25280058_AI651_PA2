import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from evaluate import load_model, per_video_errors
from utils import load_videos, split_videos, windows_at_speed, mirror, rotate, make_symmetrised

TRANSFORMS = {"M": mirror, "R": rotate}


def render_panel(rows, row_labels, shown_future, title = ""):
    # rows: list of arrays (columns, 64, 64), the first 5 columns are inputs and the rest are future frames
    columns = rows[0].shape[0]
    figure, axes = plt.subplots(len(rows), columns, figsize = (columns * 1.1 + 2, len(rows) * 1.1 + 0.6), squeeze = False)
    for row_index, row in enumerate(rows):
        for column in range(columns):
            axis = axes[row_index, column]
            axis.axis("off")
            if column < 5 or column - 5 < shown_future:
                axis.imshow(row[column], cmap = "gray", vmin = 0, vmax = 1)
            if row_index == 0:
                axis.set_title(str(column + 1), fontsize = 8)
        axes[row_index, 0].text(-0.15, 0.5, row_labels[row_index], transform = axes[row_index, 0].transAxes, ha = "right", va = "center", fontsize = 8)
    figure.suptitle(title, fontsize = 9)
    figure.subplots_adjust(left = 0.2, right = 0.99, top = 0.88, bottom = 0.02, wspace = 0.05, hspace = 0.05)
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return image


def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    Image.fromarray(image).save(path)


def save_gif(frames, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all = True, append_images = images[1:], duration = 800, loop = 0)


def to_row(first, second):
    return torch.cat([first, second], dim = 1)[0, :, 0].cpu().numpy() # (frames, 64, 64)


@torch.no_grad()
def consistency_rows(forecaster, inputs, targets, transform):
    # The four rows of Figure 3, inputs and targets have shape (1, 5, 1, 64, 64)
    prediction = forecaster(inputs)
    transformed_inputs = transform(inputs)
    transformed_prediction = forecaster(transformed_inputs)
    return [
        to_row(inputs, prediction),
        to_row(transformed_inputs, transform(targets)),
        to_row(transformed_inputs, transformed_prediction),
        to_row(transformed_inputs, transform(prediction)),
    ]


def make_consistency_figure(forecaster, selections, inputs_all, targets_all, tag, figure_name, output_dir):
    # Figure 3 (or 6a): one panel per selected video, plus one GIF per panel
    panels = []
    gif_paths = []
    for video_index, transform_name in selections:
        inputs = inputs_all[video_index:video_index + 1]
        targets = targets_all[video_index:video_index + 1]
        rows = consistency_rows(forecaster, inputs, targets, TRANSFORMS[transform_name])
        labels = ["X | f(X)", transform_name + "(X) | " + transform_name + "(Y)", transform_name + "(X) | f(" + transform_name + "(X))", transform_name + "(X) | " + transform_name + "(f(X))"]
        title = "Video " + str(video_index) + ", T = " + transform_name
        panels.append(render_panel(rows, labels, 5, title))
        frames = [render_panel(rows, labels, shown, title) for shown in range(1, 6)]
        gif_path = os.path.join(output_dir, "gifs", tag + "_video" + str(video_index) + "_" + transform_name + ".gif")
        save_gif(frames, gif_path)
        gif_paths.append(gif_path)
    save_image(np.concatenate(panels, axis = 0), os.path.join(output_dir, figure_name + ".png"))
    print(figure_name, "GIF paths:", gif_paths)


@torch.no_grad()
def make_speed_figure(forecaster, test_videos, video_index, figure_name, output_dir):
    # Figure 5 (or 6b): inputs plus the first two true and predicted frames at 1x, 2x and 3x
    panels = []
    for speed in (1, 2, 3):
        inputs, targets, _ = windows_at_speed(test_videos[video_index:video_index + 1], speed)
        prediction = forecaster(inputs)
        rows = [to_row(inputs, targets[:, :2]), to_row(inputs, prediction[:, :2])]
        panels.append(render_panel(rows, ["Truth", "Prediction"], 2, "Video " + str(video_index) + ", speed " + str(speed) + "x"))
    save_image(np.concatenate(panels, axis = 0), os.path.join(output_dir, figure_name + ".png"))


@torch.no_grad()
def make_prediction_figure(forecaster, test_videos, video_indices, figure_name, output_dir):
    # Figure 2: input, ground truth and prediction for a few test videos
    panels = []
    for video_index in video_indices:
        inputs, targets, _ = windows_at_speed(test_videos[video_index:video_index + 1], 1)
        prediction = forecaster(inputs)
        rows = [to_row(inputs, targets), to_row(inputs, prediction)]
        panels.append(render_panel(rows, ["Input | truth", "Input | prediction"], 5, "Video " + str(video_index)))
    save_image(np.concatenate(panels, axis = 0), os.path.join(output_dir, figure_name + ".png"))


def make_histogram_figure(per_video_cm, per_video_cr, selections, output_dir):
    # Figure 4: per-video consistency gaps of model A with the Figure 3 videos marked
    figure, axes = plt.subplots(1, 2, figsize = (11, 4))
    for axis, values, name in ((axes[0], per_video_cm, "M"), (axes[1], per_video_cr, "R")):
        axis.hist(values, bins = 50, edgecolor = "black")
        axis.set_yscale("log")
        axis.set_xlabel("Per-video C" + name)
        axis.set_ylabel("Number of test videos")
        for video_index, transform_name in selections:
            if transform_name == name:
                axis.axvline(values[video_index], color = "red", linestyle = "--")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "figure_4.png"), dpi = 200)
    plt.close(figure)


def overlap_scores(test_videos, frame_numbers):
    # Digits cover fewer bright pixels while they overlap, so a drop in the count signals overlap (a heuristic, check the images)
    counts = (test_videos[:, :, 0] > 51).sum(dim = (2, 3)).float()
    selected = counts[:, [frame - 1 for frame in frame_numbers]]
    return (selected.max(dim = 1).values - selected.min(dim = 1).values) / selected.max(dim = 1).values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_model", choices = ["D", "DB"], default = "D")
    parser.add_argument("--checkpoint_dir", default = "checkpoints")
    parser.add_argument("--output_dir", default = "figures")
    parser.add_argument("--results_dir", default = "results")
    parser.add_argument("--figure_2_only", action = "store_true", help = "Run this first, the full script reveals Figures 3 to 5")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_videos = split_videos(load_videos())
    test_videos = test_videos.to(device)
    inputs_all, targets_all, _ = windows_at_speed(test_videos, 1)
    os.makedirs(arguments.results_dir, exist_ok = True)

    model_a, _ = load_model(os.path.join(arguments.checkpoint_dir, "A_seed0.pt"), device)

    # Figure 2: one video with overlapping digits and two without
    scores_first_ten = overlap_scores(test_videos, list(range(1, 11)))
    overlapping_video = int(scores_first_ten.argmax().item())
    clean_videos = [int(index) for index in scores_first_ten.argsort()[:2].tolist()]
    figure_2_videos = [overlapping_video] + clean_videos
    print("Figure 2 videos (first one has overlap):", figure_2_videos, "scores:", [round(scores_first_ten[index].item(), 3) for index in figure_2_videos])
    make_prediction_figure(model_a, test_videos, figure_2_videos, "figure_2", arguments.output_dir)
    if arguments.figure_2_only:
        return

    # Figure 3 and 4: the two largest per-video CM and the two largest per-video CR of model A
    results = per_video_errors(model_a, test_videos)
    selections = []
    for name in ("M", "R"):
        top_videos = results["C" + name].argsort(descending = True)[:2].tolist()
        selections = selections + [(int(index), name) for index in top_videos]
    with open(os.path.join(arguments.results_dir, "figure_3_selections.json"), "w") as file:
        json.dump(selections, file)
    print("Figure 3 selections (video index, transform):", selections)
    make_consistency_figure(model_a, selections, inputs_all, targets_all, "modelA", "figure_3", arguments.output_dir)
    make_histogram_figure(results["CM"].cpu().numpy(), results["CR"].cpu().numpy(), selections, arguments.output_dir)

    # Figure 5: a video whose digits do not overlap in any displayed true frame
    displayed_frames = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16, 19]
    figure_5_video = int(overlap_scores(test_videos, displayed_frames).argmin().item())
    print("Figure 5 video:", figure_5_video)
    make_speed_figure(model_a, test_videos, figure_5_video, "figure_5", arguments.output_dir)

    # Figure 6: the same videos with the symmetrised final model
    final_path = os.path.join(arguments.checkpoint_dir, arguments.final_model + "_seed0.pt")
    if os.path.exists(final_path):
        final_model, _ = load_model(final_path, device)
        final_forecaster = make_symmetrised(final_model)
        make_consistency_figure(final_forecaster, selections, inputs_all, targets_all, "final", "figure_6a", arguments.output_dir)
        make_speed_figure(final_forecaster, test_videos, figure_5_video, "figure_6b", arguments.output_dir)
    else:
        print("Final model checkpoint not found, skipped Figure 6:", final_path)


if __name__ == "__main__":
    main()