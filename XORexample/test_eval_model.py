from getModel import Classifier
import torch
import torch.nn as nn

# load weights from well-trained 
state_dict = torch.load("/storage/shenhuaizhongLab/lihuilin/DLnotebook/XORexample/XOR_Classifier_model.tar")

# Create a new model and load the state
new_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
new_model.load_state_dict(state_dict)