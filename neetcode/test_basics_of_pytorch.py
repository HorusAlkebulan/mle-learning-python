# test_basics_of_pytorch.py

import torch
import torch.nn
from torchtyping import TensorType

# Helpful functions:
# https://pytorch.org/docs/stable/generated/torch.reshape.html
# https://pytorch.org/docs/stable/generated/torch.mean.html
# https://pytorch.org/docs/stable/generated/torch.cat.html
# https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html

###

# Round your answers to 4 decimal places using torch.round(input_tensor, decimals = 4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # torch.reshape() will be useful - check out the documentation
        m = to_reshape.shape[0]
        n = to_reshape.shape[1]
        new_m = (m * (n // 2))
        new_n = 2
        out = to_reshape.reshape(new_m, new_n)
        return out

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # torch.mean() will be useful - check out the documentation
        avg = torch.mean(to_avg, dim=0)
        return avg

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # torch.cat() will be useful - check out the documentation
        concat = torch.cat([cat_one, cat_two], dim=1)
        return concat

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # torch.nn.functional.mse_loss() will be useful - check out the documentation
        loss = torch.nn.functional.mse_loss(prediction, target)
        return loss

def test_reshape():
    x = torch.randn(3, 4)
    solver = Solution()
    actual = solver.reshape(x)
    actual_shape = actual.shape
    expected_shape = torch.Size([6, 2])
    assert actual_shape == expected_shape

def test_average_cols():

    x_arr = [
        [4, 5, 8, 9],
        [5, 6, 9, 10],
        [6, 7, 10, 11],
    ]
    x = torch.tensor(x_arr, dtype=torch.float32)
    solver = Solution()
    actual: torch.Tensor = solver.average(x)
    expected_arr = [5, 6, 9, 10]
    expected: torch.Tensor = torch.tensor(expected_arr, dtype=torch.float32)
    assert torch.equal(actual, expected)

def test_concat():
    # combine a MxN (3x5) and (3x3) into 3x(5+3)
    x_arr = [
        [4, 5, 8, 9],
        [5, 6, 9, 10],
        [6, 7, 10, 11],
    ]
    x = torch.tensor(x_arr, dtype=torch.float32)
    y_arr = [
        [4, 5, 8],
        [5, 6, 9],
        [6, 7, 10],
    ]
    y = torch.tensor(y_arr, dtype=torch.float32)

    solver = Solution()
    actual: torch.Tensor = solver.concatenate(x, y)
    expected_arr = [
        [4, 5, 8, 9, 4, 5, 8],
        [5, 6, 9, 10, 5, 6, 9],
        [6, 7, 10, 11, 6, 7, 10],
    ]
    expected: torch.Tensor = torch.tensor(expected_arr, dtype=torch.float32)
    assert torch.equal(actual, expected)

def test_loss():
    solver = Solution()
    prediction = torch.ones(3, 4) * 1.1
    target = torch.ones(3, 4)

    actual: torch.Tensor = solver.get_loss(prediction, target)
    expected: torch.Tensor = torch.tensor(0.01)
    assert torch.allclose(actual, expected)
