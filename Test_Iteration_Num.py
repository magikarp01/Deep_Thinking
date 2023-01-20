# Plotting the performance of neural nets with different numbers of iterations

import torch
from torch import nn
from PS_Architecture import PrefixSumNN_DT, PrefixSumNN_FF
from PS_Gen_Dataset import PrefixSumDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from Test_Model import test_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

testing_PS_data = PrefixSumDataset("data/PS_test_X.pt", "data/PS_test_y.pt")
testing_PS_dataloader = DataLoader(testing_PS_data, batch_size=10, shuffle=True)
training_PS_data = PrefixSumDataset("data/PS_train_X.pt", "data/PS_train_y.pt")

init_dt_nn = PrefixSumNN_DT(num_iter=2).to(device)
init_dt_nn.load_state_dict(torch.load('models/PS_DT.pth'))

# Two metrics: accuracy (rounding to 0 or 1) or typical loss
def get_accuracy(pred, y):
    logits = torch.nn.functional.softmax(pred, dim=1)[:, 1]
    rounded = torch.round(logits)
    return torch.sum(rounded == y)/y.numel()

loss_fn = nn.CrossEntropyLoss()
def get_loss(pred, y):
    return loss_fn(pred, y)

def iteration_loss(num_iter, init_dt_nn=init_dt_nn):
    dt_nn = PrefixSumNN_DT(num_iter=num_iter).to(device)
    dt_nn.expand_iterations(init_dt_nn)
    
    # loss, accuracy
    loss, accuracy = test_model(dt_nn, testing_PS_dataloader)
    return accuracy


xaxis = np.arange(0, 5)
yaxis = []
for i in tqdm(xaxis):
    yaxis.append(iteration_loss(i).item())
plt.plot(xaxis, yaxis)
plt.show()