import torch
import numpy as np


def print_tensor_details(tensor_name: str, tensor: torch.Tensor) -> None:
    print(f"{tensor_name}: {tensor} {tensor.shape} {tensor.dtype} rank={tensor.ndim}")

test_list  = [1, 2, 3, 4, 5]
tensor_from_list = torch.tensor(test_list)
test_tuple = (1,2,3,4,5)
tensor_from_tuple = torch.tensor(test_tuple)
test_np = np.array(test_list)
tensor_from_np = torch.from_numpy(test_np)

print(f"tensor_from_list: {tensor_from_list} {tensor_from_list.shape} {tensor_from_list.dtype}")
print(f"tensor_from_tuple: {tensor_from_tuple} {tensor_from_tuple.shape} {tensor_from_tuple.dtype}")
print(f"tensor_from_np: {tensor_from_np} {tensor_from_tuple.shape} {tensor_from_tuple.dtype}")

test_emp = torch.empty(1, 4)
test_ones = torch.ones(1, 4)
test_zeros = torch.zeros(1, 4)

print(f"test_emp: {test_emp} {test_emp.shape} {test_emp.dtype}")
print(f"test_ones: {test_ones} {test_ones.shape} {test_ones.dtype}")
print(f"test_zeros: {test_zeros} {test_zeros.shape} {test_zeros.dtype}")

test_rand = torch.rand(1, 4)
test_randn = torch.randn(1, 4)
test_randint = torch.randint(5, 10, (1, 4))

print(f"test_rand: {test_rand} {test_rand.shape} {test_rand.dtype}")
print(f"test_randn: {test_randn} {test_randn.shape} {test_randn.dtype}")
print(f"test_randint: {test_randint} {test_randint.shape} {test_randint.dtype} {test_randint.ndim}")

# data types

test_int = torch.randint(1, 10, (1, 5), dtype=torch.int8)
print(f"test_int: {test_int} {test_int.shape} {test_int.dtype} {test_int.ndim}")

test_to_float = test_int.float()
test_to_int = test_randn.to(torch.int8)
print(f"test_to_float: {test_to_float} {test_to_float.shape} {test_to_float.dtype} {test_to_float.ndim}")
print(f"test_to_int: {test_to_int} {test_to_int.shape} {test_to_int.dtype} {test_to_int.ndim}")

torch.manual_seed(111)
test_rand_uniform = torch.rand(3, 3) # 0 - 1
print_tensor_details("test_rand_uniform", test_rand_uniform)
test_rand_norm = torch.randn(3, 3) # 0 - 1
print_tensor_details("test_rand_norm", test_rand_norm)
