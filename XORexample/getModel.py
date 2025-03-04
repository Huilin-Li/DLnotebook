import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Classifier(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x
    
    def show_weight_bias(self):
        print(f"Parameter linear1.weight = {self.linear1.weight}")
        print(f"Parameter linear1.bias = {self.linear1.bias}")
        print(f"Parameter linear2.weight = {self.linear2.weight}")
        print(f"Parameter linear2.bias = {self.linear2.bias}")
    

