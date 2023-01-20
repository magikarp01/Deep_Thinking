from easy_to_hard_data import MazeDataset
import torch
import random
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class MazeSolvingDataset(Dataset):
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
    MazeDataset(root='data', train= False, size= 13, download= True)
