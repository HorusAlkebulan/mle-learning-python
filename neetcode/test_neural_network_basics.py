# test_neural_network_basics.py

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # define model
        self.layer_0 = nn.Linear(4, 6)
        self.layer_1 = nn.Linear(6, 6)
        self.layer_2 = nn.Linear(6, 2)

    def forward(self, x):
        out = self.layer_0(x)
        out = self.layer_1(out)
        out = self.layer_2(out)
        return out
    
def test_my_model():

    x = torch.randn(3, 4)
    model = MyModel()
    result = model(x)
    print(f"x={x}, model={model}, result={result}")