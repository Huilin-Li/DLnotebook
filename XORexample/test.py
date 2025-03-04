from genXORdataset import XORDataset
from getModel import Classifier
import torch


# dataset = XORDataset(size=200)

# model = Classifier(num_inputs=2, num_hidden=4, num_outputs=1)

torch.manual_seed(42) 
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z0 = torch.matmul(x, w)
z = torch.matmul(x, w)+b

print("x:")
print(x)
print("===================================================")
print("w:")
print(w)
print("===================================================")
print("b:")
print(b)
print("===================================================")
print("z0:")
print(z0)
print("===================================================")
print("z:")
print(z)
print("===================================================")
print("y:")
print(y)
print("===================================================")