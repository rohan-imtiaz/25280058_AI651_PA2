import numpy as np
import torch

DATA_ROOT = "data"
NPY_PATH = None # Set this to the path of mnist_test_seq.npy if Kaggle internet is off (shape 20 x 10000 x 64 x 64)
TRAIN_END = 8000
VALIDATION_END = 9000


def load_videos():
    # Returns all 10000 videos as a uint8 tensor of shape (10000, 20, 1, 64, 64)
    if NPY_PATH is not None:
        array = np.load(NPY_PATH)
        videos = torch.from_numpy(array).permute(1, 0, 2, 3).unsqueeze(2).contiguous()
    else:
        from torchvision.datasets import MovingMNIST
        dataset = MovingMNIST(root = DATA_ROOT, split = None, download = True)
        videos = torch.stack([dataset[index] for index in range(len(dataset))])
        if videos.dim() == 4:
            videos = videos.unsqueeze(2)
    assert videos.dtype == torch.uint8, "Expected uint8 videos"
    assert tuple(videos.shape) == (10000, 20, 1, 64, 64), "Unexpected video shape " + str(tuple(videos.shape))
    return videos


def split_videos(videos):
    # Manual split by video index, not by frame
    return videos[:TRAIN_END], videos[TRAIN_END:VALIDATION_END], videos[VALIDATION_END:]


def mirror(x):
    return torch.flip(x, dims = (-1,)) # (Mx)[i, j] = x[i, 63 - j]


def rotate(x, times = 1):
    return torch.rot90(x, k = times, dims = (-2, -1)) # 90 degrees anticlockwise per time


def apply_symmetry(x, group_index):
    # group_index = 4 * mirror_flag + a, element is M^flag R^a (rotate first, then mirror)
    rotations = group_index % 4
    mirror_flag = group_index // 4
    output = rotate(x, rotations)
    if mirror_flag == 1:
        output = mirror(output)
    return output


def inverse_symmetry(x, group_index):
    rotations = group_index % 4
    mirror_flag = group_index // 4
    output = x
    if mirror_flag == 1:
        output = mirror(output) # Mirror is its own inverse, and it must be undone before the rotation
    return rotate(output, -rotations)


def window_frames(start, speed):
    # Frame numbers are 1-indexed, only target frames that exist (<= 20) are kept
    input_frames = [start + speed * step for step in range(5)]
    target_frames = [start + speed * step for step in range(5, 10)]
    target_frames = [frame for frame in target_frames if frame <= 20]
    return input_frames, target_frames


def make_window(video, start, speed):
    # video has shape (20, 1, 64, 64)
    input_frames, target_frames = window_frames(start, speed)
    inputs = video[[frame - 1 for frame in input_frames]]
    targets = video[[frame - 1 for frame in target_frames]]
    return inputs, targets


def maximum_start(speed, minimum_targets):
    return 20 - speed * (4 + minimum_targets) # Last input frame and the needed target frames must be <= 20


def gather_windows(videos, video_indices, starts, speeds):
    # videos: uint8 (N, 20, 1, 64, 64) on the device, the other three are long tensors of shape (B,)
    steps = torch.arange(10, device = videos.device)
    frame_numbers = starts[:, None] + speeds[:, None] * steps[None, :]
    available = frame_numbers <= 20
    frame_indices = (frame_numbers - 1).clamp(max = 19) # Missing frames are clamped here and masked out later
    clips = videos[video_indices[:, None], frame_indices].float() / 255
    return clips[:, :5], clips[:, 5:], available[:, 5:]


def windows_at_speed(videos, speed, start = 1):
    # Evaluation windows for every video in videos at one speed
    count = videos.shape[0]
    video_indices = torch.arange(count, device = videos.device)
    starts = torch.full((count,), start, dtype = torch.long, device = videos.device)
    speeds = torch.full((count,), speed, dtype = torch.long, device = videos.device)
    return gather_windows(videos, video_indices, starts, speeds)


def horizon_mse(prediction, targets):
    # Both have shape (B, H, 1, 64, 64), returns the per-video error of every horizon, shape (B, H)
    return ((prediction - targets) ** 2).mean(dim = (2, 3, 4))


def masked_loss(prediction, targets, mask):
    frame_errors = horizon_mse(prediction, targets)
    mask = mask.float()
    video_errors = (frame_errors * mask).sum(dim = 1) / mask.sum(dim = 1) # Every video counts equally
    return video_errors.mean()


@torch.no_grad()
def predict_in_batches(forecaster, inputs, batch_size = 32):
    outputs = []
    for start in range(0, inputs.shape[0], batch_size):
        outputs.append(forecaster(inputs[start:start + batch_size]))
    return torch.cat(outputs)


def blank_forecaster(inputs):
    return torch.zeros_like(inputs)


def copy_last_forecaster(inputs):
    return inputs[:, -1:].repeat(1, 5, 1, 1, 1)


def make_symmetrised(forecaster):
    # Equation 7: transform, predict, undo the transform and average over the 8 symmetries
    def symmetrised(inputs):
        total = torch.zeros_like(inputs)
        for group_index in range(8):
            prediction = forecaster(apply_symmetry(inputs, group_index))
            total = total + inverse_symmetry(prediction, group_index)
        return total / 8
    return symmetrised