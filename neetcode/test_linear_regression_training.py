# test_linear_regression_training.py
import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_derivative(self, 
                       model_prediction: NDArray[np.float64], 
                       ground_truth: NDArray[np.float64], 
                       N: int, 
                       X: NDArray[np.float64], 
                       desired_weight: int
                       ) -> float:
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N
    
    def get_model_prediction(self,
                             X: NDArray[np.float64],
                             weights: NDArray[np.float64]
                             ) -> NDArray[np.float64]:
        res = np.matmul(X, weights)
        res = np.squeeze(res)
        return res
    
    learning_rate = 0.01

    def train_model(self,
                    X: NDArray[np.float64],
                    Y: NDArray[np.float64],
                    num_iterations: int,
                    initial_weights: NDArray[np.float64],
                    ) -> NDArray[np.float64]:
        
        weights = initial_weights
        N = X.shape[0]

        for _ in range(num_iterations):
            # do prediction given current weights
            predictions = self.get_model_prediction(X, weights)

            # calculate partial derivatives
            derivative_0 = self.get_derivative(predictions, Y, N, X, 0)
            derivative_1 = self.get_derivative(predictions, Y, N, X, 1)
            derivative_2 = self.get_derivative(predictions, Y, N, X, 2)

            # step adjust weights
            derivative = np.array([derivative_0, derivative_1, derivative_2])
            gradient_step = derivative * self.learning_rate
            weights = weights - gradient_step

            # repeat

        final_weights = np.round(weights, 5)
        return final_weights

"""
Input:
X = [[1, 2, 3], [1, 1, 1]]
Y = [6, 3]
num_iterations = 10
initial_weights = [0.2, 0.1, 0.6]

Output:
[0.50678, 0.59057, 1.27435]

""" 
def test_example_1():

    X = [[1, 2, 3], [1, 1, 1]]
    Y = [6, 3]
    num_iterations = 10
    initial_weights = [0.2, 0.1, 0.6]
    expected = [0.50678, 0.59057, 1.27435]

    X = np.array(X)
    Y = np.array(Y)
    expected = np.array(expected)

    solver = Solution()
    actual = solver.train_model(X, Y, num_iterations, initial_weights)
    is_equal = np.equal(expected, actual)
    assert np.all(is_equal)

test_example_1()
