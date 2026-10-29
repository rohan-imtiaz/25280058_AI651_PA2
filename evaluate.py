import argparse
import json
import math
import os
import time
import numpy as np
import torch
from model import build_model, count_parameters
from utils import (
    load_videos, split_videos, windows_at_speed, horizon_mse, predict_in_batches,
    mirror, rotate, blank_forecaster, copy_last_forecaster, make_symmetrised,
)


def load_model(path, device):
    checkpoint = torch.load(path, map_location = device)
    model = build_model(checkpoint["design"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def per_video_errors(forecaster, test_videos):
    # Everything for Tables 2 to 5 at 1x with t = 1, all values are per video (N,) unless stated otherwise
    inputs, targets, _ = windows_at_speed(test_videos, 1)
    prediction = predict_in_batches(forecaster, inputs)
    results = {"horizon_errors": horizon_mse(prediction, targets)} # (N, 5)
    for name, transform in (("M", mirror), ("R", rotate)):
        transformed_prediction = predict_in_batches(forecaster, transform(inputs))
        results["C" + name] = horizon_mse(transformed_prediction, transform(prediction)).mean(dim = 1) # Consistency gap
        results["E" + name] = horizon_mse(transformed_prediction, transform(targets)).mean(dim = 1) # Transformed error
    return results


def speed_errors(forecaster, test_videos):
    errors = {}
    for speed in (1, 2, 3):
        inputs, targets, _ = windows_at_speed(test_videos, speed)
        prediction = predict_in_batches(forecaster, inputs)
        horizon_errors = horizon_mse(prediction, targets)
        errors["E2_" + str(speed) + "x"] = horizon_errors[:, :2].mean().item()
        if speed <= 2:
            errors["E5_" + str(speed) + "x"] = horizon_errors.mean().item() # All five targets exist at 1x and 2x
    return errors


@torch.no_grad()
def time_inference(forecaster, test_videos, device, batch_size = 32):
    inputs, _, _ = windows_at_speed(test_videos, 1)
    forecaster(inputs[:batch_size]) # Untimed warm-up so that cuDNN setup is not counted
    total_seconds = 0
    for start in range(0, inputs.shape[0], batch_size):
        batch = inputs[start:start + batch_size]
        if device.type == "cuda":
            torch.cuda.synchronize()
        begin = time.time()
        forecaster(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_seconds = total_seconds + (time.time() - begin)
    return total_seconds


def evaluate_forecaster(forecaster, test_videos, device, parameters, training_time):
    results = per_video_errors(forecaster, test_videos)
    horizon_means = results["horizon_errors"].mean(dim = 0).tolist()
    metrics = {"MSE" + str(index + 1): value for index, value in enumerate(horizon_means)}
    metrics["E5"] = results["horizon_errors"].mean().item()
    for name in ("CM", "EM", "CR", "ER"):
        metrics[name] = results[name].mean().item()
    metrics.update(speed_errors(forecaster, test_videos))
    metrics["params"] = parameters
    metrics["train_time"] = training_time
    metrics["inference_time"] = time_inference(forecaster, test_videos, device)
    return metrics, results


def format_number(value):
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return "{:.2e}".format(value) # Three significant figures in scientific notation
    decimals = max(0, 2 - math.floor(math.log10(abs(value))))
    return "{:.{}f}".format(value, decimals)


def format_cell(values, key):
    values = [entry[key] for entry in values]
    if values[0] is None:
        return "-"
    if key == "params":
        return "{:,}".format(values[0])
    if len(values) == 1:
        return format_number(values[0])
    mean = sum(values) / len(values)
    return format_number(mean) + " (" + ", ".join(format_number(value) for value in values) + ")"


def make_table(title, columns, rows):
    # columns: list of (header, key), rows: list of (label, list of metric dicts, one per seed)
    lines = ["### " + title, "", "| Model | " + " | ".join(header for header, _ in columns) + " |"]
    lines.append("|---|" + "---|" * len(columns))
    for label, entries in rows:
        lines.append("| " + label + " | " + " | ".join(format_cell(entries, key) for _, key in columns) + " |")
    return "\n".join(lines) + "\n"


def table_2_only(checkpoint_dir, output_dir, test_videos, device):
    # Runs before Q2.5 is written: only horizon errors of model A, Blank and Copy-last, nothing from Testing the Baseline
    inputs, targets, _ = windows_at_speed(test_videos, 1)
    rows = []
    for label, forecaster, parameters in (("Blank", blank_forecaster, 0), ("Copy-last", copy_last_forecaster, 0)):
        rows.append((label, [horizon_entry(forecaster, inputs, targets, parameters)]))
    entries = []
    for seed in (0, 1):
        path = os.path.join(checkpoint_dir, "A_seed" + str(seed) + ".pt")
        model, _ = load_model(path, device)
        entries.append(horizon_entry(model, inputs, targets, count_parameters(model)))
    rows = [("A", entries)] + rows
    horizon_columns = [("MSE" + str(index), "MSE" + str(index)) for index in range(1, 6)]
    text = make_table("Table 2", horizon_columns + [("E(5)", "E5"), ("Parameters", "params")], rows)
    with open(os.path.join(output_dir, "table_2.md"), "w") as file:
        file.write(text)
    print(text)


def horizon_entry(forecaster, inputs, targets, parameters):
    horizon_errors = horizon_mse(predict_in_batches(forecaster, inputs), targets)
    entry = {"MSE" + str(index + 1): value for index, value in enumerate(horizon_errors.mean(dim = 0).tolist())}
    entry["E5"] = horizon_errors.mean().item()
    entry["params"] = parameters
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default = "checkpoints")
    parser.add_argument("--output_dir", default = "results")
    parser.add_argument("--table_2_only", action = "store_true", help = "Run this first, the full evaluation reveals Table 3 and 4")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Hardware:", torch.cuda.get_device_name(0))
    _, _, test_videos = split_videos(load_videos())
    test_videos = test_videos.to(device)
    os.makedirs(arguments.output_dir, exist_ok = True)
    if arguments.table_2_only:
        table_2_only(arguments.checkpoint_dir, arguments.output_dir, test_videos, device)
        return

    all_metrics = {} # Label -> list of metric dicts (one per seed)
    for label, forecaster in (("Blank", blank_forecaster), ("Copy-last", copy_last_forecaster)):
        metrics, _ = evaluate_forecaster(forecaster, test_videos, device, 0, None)
        all_metrics[label] = [metrics]

    for model_name in ("A", "B", "C", "D", "DB"):
        entries = []
        symmetrised_entries = []
        for seed in (0, 1):
            path = os.path.join(arguments.checkpoint_dir, model_name + "_seed" + str(seed) + ".pt")
            if not os.path.exists(path):
                continue
            model, checkpoint = load_model(path, device)
            parameters = count_parameters(model)
            training_time = sum(checkpoint["epoch_times"]) / len(checkpoint["epoch_times"])
            metrics, results = evaluate_forecaster(model, test_videos, device, parameters, training_time)
            entries.append(metrics)
            if model_name == "A" and seed == 0:
                np.savez(os.path.join(arguments.output_dir, "per_video_A_seed0.npz"), CM = results["CM"].cpu().numpy(), CR = results["CR"].cpu().numpy())
            if model_name in ("B", "D", "DB"):
                symmetrised_metrics, _ = evaluate_forecaster(make_symmetrised(model), test_videos, device, parameters, training_time)
                symmetrised_entries.append(symmetrised_metrics)
            print("Evaluated", model_name, "seed", seed)
        if entries:
            all_metrics[model_name] = entries
        if symmetrised_entries:
            all_metrics["Symmetrised " + model_name] = symmetrised_entries

    with open(os.path.join(arguments.output_dir, "metrics.json"), "w") as file:
        json.dump(all_metrics, file, indent = 2)

    horizon_columns = [("MSE" + str(index), "MSE" + str(index)) for index in range(1, 6)]
    table_2 = make_table("Table 2", horizon_columns + [("E(5)", "E5"), ("Parameters", "params")], [(label, all_metrics[label]) for label in ("A", "Blank", "Copy-last") if label in all_metrics])
    table_3 = make_table("Table 3", [("E(5)", "E5"), ("CM", "CM"), ("EM", "EM"), ("CR", "CR"), ("ER", "ER")], [(label, all_metrics[label]) for label in ("A", "Blank", "Copy-last") if label in all_metrics])
    table_4 = make_table("Table 4", [("E(2) 1x", "E2_1x"), ("E(2) 2x", "E2_2x"), ("E(2) 3x", "E2_3x"), ("E(5) 1x", "E5_1x"), ("E(5) 2x", "E5_2x")], [(label, all_metrics[label]) for label in ("A", "Copy-last") if label in all_metrics])
    full_columns = [
        ("E(5) 1x", "E5"), ("CM", "CM"), ("EM", "EM"), ("CR", "CR"), ("ER", "ER"),
        ("E(2) 1x", "E2_1x"), ("E(2) 2x", "E2_2x"), ("E(2) 3x", "E2_3x"),
        ("Parameters", "params"), ("Train s/epoch", "train_time"), ("Inference s", "inference_time"),
    ]
    table_5 = make_table("Table 5", full_columns, [(label, all_metrics[label]) for label in ("A", "B", "C", "D", "DB") if label in all_metrics])
    table_6 = make_table("Table 6", full_columns, [(label, all_metrics[label]) for label in ("Symmetrised B", "Symmetrised D", "Symmetrised DB") if label in all_metrics])

    text = "\n".join([table_2, table_3, table_4, table_5, table_6])
    with open(os.path.join(arguments.output_dir, "tables.md"), "w") as file:
        file.write(text)
    print(text)


if __name__ == "__main__":
    main()