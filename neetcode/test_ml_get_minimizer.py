# ml_get_minimizer.py


# Clean solution
class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:

        x = init

        # calculate derivative of function x**2
        # dx/dt(f) = 2x
        for _ in range(iterations):
            gradient_step = 2 * x * learning_rate
            x = x - gradient_step
        x = round(x, 5)
        return x

class WorkingSolution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:

        model = Model(init)
        x = init

        # calculate derivative of function x**2
        # dx/dt(f) = 2x
        print(f"Beginning training loop")
        for i in range(iterations):

            print(f"BEGIN ITERATION locals: {locals()}")

            # get expected label
            expected = 0.0

            # guess using x
            actual = model.forward(x)

            # compute step change: x = x - (derivative * learning*rate)
            # determine gradient step, remember derivative is actually slop
            gradient_step = 2 * x * learning_rate
            x = x - gradient_step

            print(f"\tEND ITERATION locals(): {locals()}")
        x = round(x, 5)
        return x


class Model:
    def __init__(self, initial_guess: int):
        print(f"Initializing model")
        self.init = initial_guess

    def forward(self, x: float):
        print(f"Computing forward pass with x={x}")
        return x**2


"""
Input: iterations = 0, learning_rate = 0.01, init = 5

Output:
5
"""


def test_example_1():

    iterations = 0
    learning_rate = 0.01
    init = 5
    expected = 5

    print(f"Starting test with locals: {locals()}")
    solver = Solution()
    actual = solver.get_minimizer(iterations, learning_rate, init)
    assert actual == expected


"""
Input: 
iterations = 10, learning_rate = 0.01, init = 5

Output:
4.08536
"""


def test_example_2():

    iterations = 10
    learning_rate = 0.01
    init = 5
    expected = 4.08536

    print(f"Starting test with locals: {locals()}")
    solver = Solution()
    actual = solver.get_minimizer(iterations, learning_rate, init)
    # mse = (actual - expected) ** 2.0
    # assert mse < learning_rate
    assert actual == expected
