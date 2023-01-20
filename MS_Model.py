import torch
from torch import nn
from MS_Architecture import MazeSolvingNN_FF, MazeSolvingNN_DT
from MS_Gen_Dataset import MazeSolvingDataset
from Train_Model import train_loop
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

loss_fn = nn.CrossEntropyLoss()

training_MS_data = MazeSolvingDataset("data/maze_data_train_9")
training_dataloader = DataLoader(training_MS_data, batch_size=50, shuffle=True)

num_epochs = 10
num_iter = 6
max_batches = 200
if __name__=="__main__":
    print("Training ff_nn")
    ff_nn = MazeSolvingNN_FF(num_iter=num_iter).to(device)
    ff_optimizer = torch.optim.SGD(ff_nn.parameters(), lr=.001,
    weight_decay=2e-4, momentum=.9)

    for i in range(num_epochs):
        print(f"At epoch {i}, ", end="")
        train_loop(ff_nn, training_dataloader, ff_optimizer, max_batches=max_batches)
    
    torch.save(ff_nn.state_dict(), 'models/MS_FF.pth')


    print("Training dt_nn")
    dt_nn = MazeSolvingNN_DT(num_iter=num_iter).to(device)
    ff_optimizer = torch.optim.SGD(dt_nn.parameters(), lr=.001,
    weight_decay=2e-4, momentum=.9)

    for i in range(num_epochs):
        print(f"At epoch {i}, ", end="")
        train_loop(dt_nn, training_dataloader, ff_optimizer, max_batches=max_batches)
    
    torch.save(dt_nn.state_dict(), 'models/MS_DT.pth')