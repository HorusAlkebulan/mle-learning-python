# test_linear_regression_forward.py

import numpy as np
from numpy.typing import NDArray
import random

# Helpful functions:
# https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
# https://numpy.org/doc/stable/reference/generated/numpy.mean.html
# https://numpy.org/doc/stable/reference/generated/numpy.square.html

seed = 12345
random.seed(seed)
np.random.seed(seed)

class Solution:
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is an Nx3 NumPy array
        # weights is a 3x1 NumPy array
        # HINT: np.matmul() will be useful
        # return np.round(your_answer, 5)
        
        pred = np.matmul(X, weights)
        return np.round(pred, 5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # model_prediction is an Nx1 NumPy array
        # ground_truth is an Nx1 NumPy array
        # HINT: np.mean(), np.square() will be useful
        # return round(your_answer, 5)
        
        error = model_prediction - ground_truth
        sq_error = np.square(error)
        
        # n = model_prediction.shape[0]
        mse = np.mean(sq_error)
        return np.round(mse, 5)


def test_regression_forward_ex1():

    solver = Solution()

    # size = rows, columns
    X = np.random.normal(size=[100, 3])
    print(f"X={X}")

    weights = np.random.normal(size=[3, 1])
    print(f"weights={weights}")

    ground_truth = np.random.normal(size=[100, 1])
    print(f"ground_truth={ground_truth}")

    res = solver.get_model_prediction(X, weights)
    error = solver.get_error(res, ground_truth)

    print(f"res({res.shape})={res}")
    print(f"error({error.shape})={error}")

if __name__ == "__main__":
    test_regression_forward_ex1()