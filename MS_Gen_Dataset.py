from easy_to_hard_data import MazeDataset
import torch
import random
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class MazeDataset(Dataset):
    def __init__(self, root_name):
        self.inputs = torch.tensor(np.load(root_name + '/inputs.npy'), 
        dtype=torch.float32, requires_grad=True, device=device)
        self.solutions = torch.tensor(np.load(root_name + '/solutions.npy'),
        dtype=torch.long, device=device)

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.solutions[idx]

if __name__ == '__main__':
    # MazeDataset(root='data', train= True, size= 9, download= True)
    inputs = np.load('data/maze_data_train_9/inputs.npy')
    solutions = np.load('data/maze_data_train_9/solutions.npy')
    print(inputs.shape)
    print(solutions.shape)

    train_dataloader = DataLoader(MazeDataset('data/maze_data_train_9'))
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    # print(train_features)
    # print(train_labels)

