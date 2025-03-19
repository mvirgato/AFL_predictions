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
    def __init__(self, inputNum, hidden_nodes, output_nodes, num_hidden_layers):
        super(DenseNN, self).__init__()
        
        self.inputNum = inputNum
        
        # Input layer
        self.input_layer = nn.Linear(in_features=inputNum, out_features=hidden_nodes)
        
        # Dynamically create hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_features=hidden_nodes, out_features=hidden_nodes) 
            for _ in range(num_hidden_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(in_features=hidden_nodes, out_features=output_nodes)

    def forward(self, x):
        x = x.view(-1, self.inputNum)  # Flatten the input
        
        # Pass through input layer
        x = torch.relu(self.input_layer(x))
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        
        # Pass through output layer
        x = self.output_layer(x)
        
        # return F.softmax(x, dim=1)
        return x
    
    def predict(self, x):
        """
        Perform inference and return probabilities using softmax.
        """
        self.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():  # Disable gradient computation for inference
            logits = self.forward(x)  # Get raw logits
            probabilities = F.softmax(logits, dim=1)  # Apply softmax to get probabilities

        return probabilities