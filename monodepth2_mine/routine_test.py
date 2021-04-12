import torch.optim as optim
import torch
import os

device = torch.device('cpu')
load_weights_folder = os.path.join(os.path.expanduser("~"), "tmp/mono_model/models/weights_13")
optimizer_load_path = os.path.join(load_weights_folder, "adam.pth")
optimizer_dict = torch.load(optimizer_load_path,map_location=device)
print('learning rate is ',optimizer_dict['param_groups'][0]['lr'])
a=torch.Tensor([1,3,5,7])
print(a>2)