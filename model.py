import torch
import torch.nn as nn
import torch.nn.functional as F

# Each design is one choice of patch size, hidden channels, kernel size and dilation
DESIGNS = {
    "baseline": {"patch_size": 4, "hidden_channels": 64, "kernel_size": 3, "dilation": 1},
    "D1": {"patch_size": 8, "hidden_channels": 64, "kernel_size": 3, "dilation": 1}, # Bigger patch: one grid cell is 8 pixels
    "D2": {"patch_size": 4, "hidden_channels": 64, "kernel_size": 3, "dilation": 2}, # Dilated convolutions on the 4 pixel grid
}
MODEL_D_DESIGN = "D1" # Must be changed in case D2 wins or is preferred. I opted for D1 due to its better speed (4s vs 10-18s).

# (speed augmentation, symmetry augmentation) for every model
TRAINING_SETUPS = {
    "A": (False, False),
    "B": (False, True),
    "C": (True, False),
    "D": (True, False),
    "DB": (True, True),
}


def design_for_model(model_name, design_name = None):
    if model_name in ("D", "DB"):
        if design_name is None:
            return MODEL_D_DESIGN
        return design_name
    return "baseline"


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, dilation):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding = padding,
            dilation = dilation,
            bias = True,
        ) # One convolution produces all four gates

    def forward(self, inputs, hidden, cell):
        gates = self.conv(torch.cat([inputs, hidden], dim = 1))
        input_gate, forget_gate, output_gate, candidate = torch.chunk(gates, 4, dim = 1) # Order is [i, f, o, g]
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class ConvLSTMForecaster(nn.Module):
    def __init__(self, patch_size = 4, hidden_channels = 64, kernel_size = 3, dilation = 1):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_channels = hidden_channels
        folded_channels = patch_size * patch_size
        self.layer_1 = ConvLSTMCell(folded_channels, hidden_channels, kernel_size, dilation)
        self.layer_2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size, dilation)
        self.output_conv = nn.Conv2d(hidden_channels, folded_channels, 1, bias = True)

    def forward(self, inputs):
        # inputs: (B, 5, 1, 64, 64), returns (B, 5, 1, 64, 64)
        batch_size = inputs.shape[0]
        grid = 64 // self.patch_size
        zeros = torch.zeros(batch_size, self.hidden_channels, grid, grid, device = inputs.device, dtype = inputs.dtype)
        hidden_1, cell_1, hidden_2, cell_2 = zeros, zeros, zeros, zeros
        predictions = []
        frame = None
        for step in range(1, 10):
            if step <= 5:
                frame = inputs[:, step - 1]
            folded = F.pixel_unshuffle(frame, self.patch_size)
            hidden_1, cell_1 = self.layer_1(folded, hidden_1, cell_1)
            hidden_2, cell_2 = self.layer_2(hidden_1, hidden_2, cell_2)
            frame = torch.sigmoid(F.pixel_shuffle(self.output_conv(hidden_2), self.patch_size)) # Fed back as the next input after step 5
            if step >= 5:
                predictions.append(frame) # Outputs of steps 1 to 4 are discarded
        return torch.stack(predictions, dim = 1)


def build_model(design_name):
    return ConvLSTMForecaster(**DESIGNS[design_name])


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def reach_in_pixels(design_name):
    # Longest one-step path is input -> layer 1 convolution -> layer 2 convolution (or h1 of the last step -> layer 1 -> layer 2)
    # Each convolution moves information by dilation * (kernel_size - 1) / 2 grid cells, one cell is patch_size original pixels
    design = DESIGNS[design_name]
    cells_per_convolution = design["dilation"] * (design["kernel_size"] - 1) // 2
    number_of_convolutions = 2
    return design["patch_size"] * number_of_convolutions * cells_per_convolution


if __name__ == "__main__":
    for design_name in DESIGNS:
        model = build_model(design_name)
        print(design_name, "parameters:", count_parameters(model), "reach (pixels):", reach_in_pixels(design_name))
    assert count_parameters(build_model("baseline")) == 480784, "Model A must have 480784 parameters"
    for design_name in ("D1", "D2"):
        assert count_parameters(build_model(design_name)) <= 721176, "Model D is over the parameter budget"
        assert reach_in_pixels(design_name) > reach_in_pixels("baseline"), "Model D reach must be strictly larger"
    output = build_model("baseline")(torch.rand(2, 5, 1, 64, 64))
    assert tuple(output.shape) == (2, 5, 1, 64, 64)
    print("All model checks passed")