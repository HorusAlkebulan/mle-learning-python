# manipulate_tensors.py
import torch
import numpy as np


def print_tensor_details(tensor_name: str, tensor: torch.Tensor) -> None:
    print(
        f"{tensor_name}: type={type(tensor)}, shape={tensor.shape} dtype={tensor.dtype} rank={tensor.ndim}\n    data={tensor}"
    )


one_dim_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
print_tensor_details("one_dim_tensor", one_dim_tensor)

element = one_dim_tensor[2]
print(f"index 2 element: {element} {type(element)}")

scalar = element.item()
print(f"index 2 element as scalar: {scalar} {type(scalar)}")

# remember syntax is [start:end:step] and is end-exluding
subset = one_dim_tensor[1:4]
print_tensor_details("subset", subset)

two_dim_tensor = torch.tensor(
    [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24],
    ]
)
print_tensor_details("two_dim_tensor", two_dim_tensor)

subset_2d = two_dim_tensor[1][3]
print_tensor_details("subset_2d", subset_2d)

first_three_2nd_row = two_dim_tensor[1][0:3]
print_tensor_details("first_three_2nd_row", first_three_2nd_row)

select_given_criteria = two_dim_tensor[two_dim_tensor < 11]
print_tensor_details("select_given_criteria", select_given_criteria)

stacked = torch.stack([two_dim_tensor, two_dim_tensor])
print_tensor_details("stacked", stacked)

concatted = torch.cat([two_dim_tensor, two_dim_tensor])
print_tensor_details("concatted", concatted)