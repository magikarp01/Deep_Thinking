from overthinking_train_loop import overthink_train_loop
import torch
from torch import nn
from MS_Architecture_Recall import MazeSolvingNN_Recall
from MS_Gen_Dataset import MazeSolvingDataset
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

loss_fn = nn.CrossEntropyLoss()

training_MS_data = MazeSolvingDataset("data/maze_data_train_9")
training_dataloader = DataLoader(training_MS_data, batch_size=50, shuffle=True)

num_epochs = 10
num_iter = 20
max_batches = 100
iter_range=[1, 20]
if __name__=="__main__":

    print("Training dt_nn")
    dt_nn = MazeSolvingNN_Recall(num_iter=num_iter).to(device)
    dt_optimizer = torch.optim.Adam(dt_nn.parameters(), weight_decay=2e-4, lr=.001)

    for i in range(num_epochs):
        print(f"At epoch {i}, ", end="")
        overthink_train_loop(dt_nn, training_dataloader, dt_optimizer, max_batches=max_batches, iter_range=iter_range)
    
    torch.save(dt_nn.state_dict(), 'models/MS_Overthinking_Recall_' + str(num_iter) + '.pth')