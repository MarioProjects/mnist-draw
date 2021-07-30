'''
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
'''

import torch
import torch.nn as nn
import os
from collections import OrderedDict
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def name(self):
        return "MLP"


def load_model(model_name, states_path="", model_path="", data_parallel=False):

    if model_path!="" and os.path.exists(model_path):
        return torch.load(model_path)
    elif model_path != "": assert False, "Wrong Model Path!"

    if not os.path.exists(states_path): assert False, "Wrong Models_States Path!"

    if 'MLP' in model_name:
        my_model = MLPNet().cpu()
    else: assert False, "Model '" + str(model_name) + "' not configured!"
    if data_parallel: my_model = torch.nn.DataParallel(my_model, device_ids=range(torch.cuda.device_count()))

    model_state_dict = torch.load(states_path, map_location=lambda storage, location: storage)

    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if 'module' in k:
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            my_model.load_state_dict(model_state_dict)
            return my_model

    # load params
    my_model.load_state_dict(new_state_dict)
    return my_model