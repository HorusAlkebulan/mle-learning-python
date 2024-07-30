# test_common_experiment_functions.py

from common_experiment_functions import get_data, type_to_numeric

def test_type_to_numeric():
    x = "versicolor"
    expected = 1
    actual = type_to_numeric(x)
    assert actual == expected

def test_get_data():
    X_np, Y_np = get_data()
    print("Data recieved successfully")
    expected_X_shape = (150, 4)
    expected_Y_shape = (150, 3)
    assert X_np.shape == expected_X_shape
    assert Y_np.shape == expected_Y_shape

if __name__ == "__main__":
    test_type_to_numeric()
    test_get_data()