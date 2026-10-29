import argparse
import json
import os
import time
import torch
from model import TRAINING_SETUPS, build_model, count_parameters, design_for_model
from utils import (
    load_videos, split_videos, gather_windows, windows_at_speed, horizon_mse, masked_loss,
    apply_symmetry, predict_in_batches,
)

BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def sample_batch(videos, batch_indices, speed_augmentation, symmetry_augmentation, device):
    # One random window per video, drawn under the rules in the Setup section
    batch_size = len(batch_indices)
    if speed_augmentation:
        speeds = torch.randint(1, 4, (batch_size,))
        maximum_starts = 20 - 6 * speeds # At least two target frames must exist
    else:
        speeds = torch.ones(batch_size, dtype = torch.long)
        maximum_starts = torch.full((batch_size,), 11) # All five target frames must exist
    starts = 1 + torch.floor(torch.rand(batch_size) * maximum_starts).long() # Uniform over 1..maximum_starts
    inputs, targets, mask = gather_windows(videos, batch_indices.to(device), starts.to(device), speeds.to(device))
    if symmetry_augmentation:
        group_indices = torch.randint(0, 8, (batch_size,)).to(device)
        for group_index in range(8):
            selected = group_indices == group_index
            if selected.any():
                inputs[selected] = apply_symmetry(inputs[selected], group_index) # The same g is used for input and target
                targets[selected] = apply_symmetry(targets[selected], group_index)
    return inputs, targets, mask


def validation_errors(model, validation_videos):
    # E(2) at each speed on the validation split, used only for monitoring and for choosing the design of model D
    errors = {}
    for speed in (1, 2, 3):
        inputs, targets, mask = windows_at_speed(validation_videos, speed)
        prediction = predict_in_batches(model, inputs)
        errors["E2_" + str(speed) + "x"] = horizon_mse(prediction[:, :2], targets[:, :2]).mean().item()
    return errors


def train(model_name, seed, design_name, epochs, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speed_augmentation, symmetry_augmentation = TRAINING_SETUPS[model_name]
    design_name = design_for_model(model_name, design_name)
    if device.type == "cuda":
        print("Hardware:", torch.cuda.get_device_name(0))

    torch.manual_seed(seed) # One seed controls initialisation, shuffling, window sampling and augmentation
    videos = load_videos()
    train_videos, validation_videos, _ = split_videos(videos)
    train_videos = train_videos.to(device)
    validation_videos = validation_videos.to(device)

    model = build_model(design_name).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    print("Model", model_name, "design", design_name, "seed", seed, "parameters", count_parameters(model))

    epoch_times = []
    validation_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        begin = time.time()
        permutation = torch.randperm(train_videos.shape[0])
        epoch_loss = 0
        batch_count = 0
        for start in range(0, train_videos.shape[0], BATCH_SIZE):
            batch_indices = permutation[start:start + BATCH_SIZE]
            inputs, targets, mask = sample_batch(train_videos, batch_indices, speed_augmentation, symmetry_augmentation, device)
            prediction = model(inputs) # No teacher forcing, gradients flow through the fed-back predictions
            loss = masked_loss(prediction, targets, mask)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss = epoch_loss + loss.item()
            batch_count = batch_count + 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.time() - begin) # Validation below is not part of the training time

        model.eval()
        errors = validation_errors(model, validation_videos)
        errors["epoch"] = epoch
        errors["train_loss"] = epoch_loss / batch_count
        validation_log.append(errors)
        print(json.dumps(errors), "epoch time", round(epoch_times[-1], 2))

    os.makedirs(output_dir, exist_ok = True)
    path = os.path.join(output_dir, model_name + "_seed" + str(seed) + ".pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "design": design_name,
            "model_name": model_name,
            "seed": seed,
            "epoch_times": epoch_times,
            "validation_log": validation_log,
            "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
        path,
    ) # Final-epoch weights are kept
    print("Saved", path, "mean epoch time", sum(epoch_times) / len(epoch_times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices = list(TRAINING_SETUPS.keys()), required = True)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--design", default = None, help = "Only used for D and DB, defaults to MODEL_D_DESIGN in model.py")
    parser.add_argument("--epochs", type = int, default = 30)
    parser.add_argument("--output_dir", default = "checkpoints")
    arguments = parser.parse_args()
    train(arguments.model, arguments.seed, arguments.design, arguments.epochs, arguments.output_dir)