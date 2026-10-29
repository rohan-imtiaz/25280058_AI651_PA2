import torch
from utils import mirror, rotate, apply_symmetry, inverse_symmetry, make_window


def bright_pixel(video):
    location = (video > 0.5).nonzero()[0].tolist()
    return tuple(location[0:1] + location[2:]) # (frame, row, column)


def frames_used(function_output):
    return [int(round(value)) for value in function_output[:, 0, 0, 0].tolist()]


torch.manual_seed(0)
x = torch.rand(20, 1, 64, 64)
marker = torch.zeros(20, 1, 64, 64)
marker[7, 0, 10, 20] = 1

difference_a = (mirror(mirror(x)) - x).abs().max().item()
equal_a = torch.equal(mirror(mirror(x)), x)

four_rotations = x
for step in range(4):
    four_rotations = rotate(four_rotations)
difference_b = (four_rotations - x).abs().max().item()
equal_b = torch.equal(four_rotations, x)

difference_c = (mirror(rotate(mirror(x))) - rotate(x, -1)).abs().max().item()
equal_c = torch.equal(mirror(rotate(mirror(x))), rotate(x, -1))

ramp = torch.arange(1, 21).float().view(20, 1, 1, 1).repeat(1, 1, 64, 64) # Frame f is filled with the value f
inputs_1, targets_1 = make_window(ramp, 1, 1)
inputs_2, targets_2 = make_window(ramp, 1, 2)
inputs_3, targets_3 = make_window(ramp, 1, 3)

rows = [
    ("(a) max|M(M(x)) - x|", "0", str(difference_a) + " equal = " + str(equal_a), difference_a == 0 and equal_a),
    ("(b) max|R4(x) - x|", "0", str(difference_b) + " equal = " + str(equal_b), difference_b == 0 and equal_b),
    ("(c) max|MRM(x) - R-1(x)|", "0", str(difference_c) + " equal = " + str(equal_c), difference_c == 0 and equal_c),
    ("(d) bright pixel of M(p)", "(7, 10, 43)", str(bright_pixel(mirror(marker))), bright_pixel(mirror(marker)) == (7, 10, 43)),
    ("(e) bright pixel of R(p)", "(7, 43, 10)", str(bright_pixel(rotate(marker))), bright_pixel(rotate(marker)) == (7, 43, 10)),
    ("(f) 1x input / target frames", "1-5 / 6-10", str(frames_used(inputs_1)) + " / " + str(frames_used(targets_1)), frames_used(inputs_1) == [1, 2, 3, 4, 5] and frames_used(targets_1) == [6, 7, 8, 9, 10]),
    ("(g) 2x input / target frames", "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19", str(frames_used(inputs_2)) + " / " + str(frames_used(targets_2)), frames_used(inputs_2) == [1, 3, 5, 7, 9] and frames_used(targets_2) == [11, 13, 15, 17, 19]),
    ("(h) 3x input / target frames", "1, 4, 7, 10, 13 / 16, 19", str(frames_used(inputs_3)) + " / " + str(frames_used(targets_3)), frames_used(inputs_3) == [1, 4, 7, 10, 13] and frames_used(targets_3) == [16, 19]),
]

print("| Check | Expected | Yours | Pass |")
print("|---|---|---|---|")
for name, expected, yours, passed in rows:
    print("| " + name + " | " + expected + " | " + yours + " | " + str(passed) + " |")

# Extra check: every group element must be undone exactly by its inverse, and the 8 outputs must all differ
for group_index in range(8):
    assert torch.equal(inverse_symmetry(apply_symmetry(x, group_index), group_index), x), "Inverse failed for " + str(group_index)
distinct = {tuple(apply_symmetry(x[0, 0], group_index).flatten()[:5].tolist()) for group_index in range(8)}
assert len(distinct) == 8, "The 8 symmetries should give 8 different images"
assert all(row[3] for row in rows), "At least one Table 1 row does not match, fix utils.py before continuing"
print("All Table 1 checks passed")