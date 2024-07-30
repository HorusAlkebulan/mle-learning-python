# test_common_experiment_functions.py

from common_experiment_functions import get_data, type_to_numeric

def test_type_to_numeric():
    x = "versicolor"
    expected = 1
    actual = type_to_numeric(x)
    assert actual == expected
    
def test_get_data():
    get_data()
    print("Data recieved successfully")