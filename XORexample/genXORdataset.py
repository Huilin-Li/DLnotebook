import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np




class XORDataset(data.Dataset):

    def __init__(self, size, std=0.1):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        data_point = self.data[index]
        data_label = self.label[index]
        return data_point, data_label
    
    def visual(self):
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.cpu().numpy()
        if isinstance(self.label, torch.Tensor):
            self.label = self.label.cpu().numpy()
        data_0 = self.data[self.label == 0]
        data_1 = self.data[self.label == 1]

        plt.figure(figsize=(4,4))
        plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
        plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
        plt.title("Dataset samples")
        plt.ylabel(r"$x_2$")
        plt.xlabel(r"$x_1$")
        plt.legend()
        plt.savefig("/storage/shenhuaizhongLab/lihuilin/PPIall/PytorchWORK/XORexample/XORexample_dataset.png")
        plt.close()

        
    
