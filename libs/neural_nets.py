import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

class PandasDataset(Dataset):

    def __init__(self, dataframe):
        self.data = dataframe.drop(columns=['hteamwin']).values  # Features
        self.labels = dataframe['hteamwin'].values.astype(np.int16)  # Targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    


# Define architecture for the dense neural network
class DenseNN(nn.Module):
    def __init__(self,inputNum, hidden_nodes, output_nodes):

        super(DenseNN, self).__init__()
        
        self.inputNum=inputNum
        
        self.fc1 = nn.Linear(in_features=inputNum, out_features=hidden_nodes)  # Input layer
        self.fc2 = nn.Linear(in_features=hidden_nodes, out_features=hidden_nodes)     # Hidden layer
        self.fc3 = nn.Linear(in_features=hidden_nodes, out_features=hidden_nodes)     # Hidden layer
        self.fc4 = nn.Linear(in_features=hidden_nodes, out_features=hidden_nodes)     # Hidden layer
        # self.fc5 = nn.Linear(in_features=hidden_nodes, out_features=hidden_nodes)     # Hidden layer
        self.fc5 = nn.Linear(in_features=hidden_nodes, out_features=output_nodes)     # Output layer

    def forward(self, x):
        
        x = x.view(-1, self.inputNum)   # Flatten the input
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        # x = torch.relu(self.fc5(x))
        x = self.fc5(x)

        return F.softmax(x, dim=1)
        # return x