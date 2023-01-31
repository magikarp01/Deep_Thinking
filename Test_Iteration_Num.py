# Plotting the performance of neural nets with different numbers of iterations

import torch
from torch import nn
from PS_Architecture import PrefixSumNN_DT, PrefixSumNN_FF
from MS_Architecture import MazeSolvingNN_FF, MazeSolvingNN_DT
from MS_Architecture_Recall import MazeSolvingNN_Recall
from PS_Gen_Dataset import PrefixSumDataset
from MS_Gen_Dataset import MazeSolvingDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from MS_Architecture_Overthinking_Recall import MazeSolvingNN_Recall_ProgLoss

from torch.utils.data import Dataset, DataLoader
from Test_Model import test_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def plot_iteration_accuracy(dt_nn, nn_constructor, dataloader, title, xrange=(0, 5)):
    # Two metrics: accuracy (rounding to 0 or 1) or typical loss
    def get_accuracy(pred, y):
        logits = torch.nn.functional.softmax(pred, dim=1)[:, 1]
        rounded = torch.round(logits)
        return torch.sum(rounded == y)/y.numel()

    loss_fn = nn.CrossEntropyLoss()
    def get_loss(pred, y):
        return loss_fn(pred, y)

    def iteration_loss(num_iter, init_dt_nn=dt_nn):
        dt_nn = nn_constructor(num_iter=num_iter).to(device)
        dt_nn.expand_iterations(init_dt_nn)
        
        # loss, accuracy
        loss, accuracy = test_model(dt_nn, dataloader)
        return accuracy


    xaxis = np.arange(xrange[0], xrange[1], 2)
    yaxis = []
    for i in tqdm(xaxis):
        yaxis.append(iteration_loss(i).item())
    plt.plot(xaxis, yaxis, linestyle='--', marker='o',)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    testing_PS_data = PrefixSumDataset("data/PS_test_X.pt", "data/PS_test_y.pt")
    testing_PS_dataloader = DataLoader(testing_PS_data, batch_size=10, shuffle=True)

    testing_MS_data = MazeSolvingDataset("data/maze_data_test_13")
    testing_MS_dataloader = DataLoader(testing_MS_data, batch_size=10, shuffle=False)

    init_PS_dt_nn = PrefixSumNN_DT(num_iter=20).to(device)
    init_PS_dt_nn.load_state_dict(torch.load('models/PS_DT.pth'))

    # init_MS_dt_nn = MazeSolvingNN_DT(num_iter=6).to(device)
    # init_MS_dt_nn.load_state_dict(torch.load('models/MS_DT_6.pth'))

    init_MS_dt_nn = MazeSolvingNN_DT(num_iter=20).to(device)
    init_MS_dt_nn.load_state_dict(torch.load('models/MS_DT_20.pth'))

    init_MS_recall_nn = MazeSolvingNN_Recall(num_iter=20).to(device)
    init_MS_recall_nn.load_state_dict(torch.load('models/MS_Recall_20.pth'))

    init_MS_recall_progloss_nn = MazeSolvingNN_Recall_ProgLoss(num_iter=20).to(device)
    init_MS_recall_progloss_nn.load_state_dict(torch.load('models/MS_Recall_ProgLoss.pth'))


    # plot_iteration_accuracy(init_PS_dt_nn, PrefixSumNN_DT, testing_PS_dataloader, 
    # title="Prefix Sum Model Trained on 20 Iterations 2", xrange=(5,40))

    # plot_iteration_accuracy(init_MS_dt_nn, MazeSolvingNN_DT, testing_MS_dataloader, 
    # title="Maze Solving Model Trained on 20 Iterations 2", xrange=(5, 40))

    # plot_iteration_accuracy(init_MS_dt_nn, MazeSolvingNN_DT, testing_MS_dataloader, 
    # title="Maze Solving Model Trained on 20 Iterations 2", xrange=(5,40))


    # plot_iteration_accuracy(init_MS_dt_nn, MazeSolvingNN_DT, testing_MS_dataloader, 
    # title="Maze Solving DT Loss Model Trained on 20 Iterations", xrange=(5,40))

    # plot_iteration_accuracy(init_MS_recall_nn, MazeSolvingNN_Recall, testing_MS_dataloader, 
    # title="Maze Solving Recall Loss Model Trained on 20 Iterations", xrange=(5,40))

    plot_iteration_accuracy(init_MS_recall_progloss_nn, MazeSolvingNN_Recall_ProgLoss, testing_MS_dataloader, 
    title="Maze Solving Recall Progressive Loss Model Trained on Max 20 Iterations", xrange=(5,80))



