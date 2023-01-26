import torch
# from torch import nn
import random
from MS_Architecture_Overthinking_Recall import MazeSolvingNN_Recall_ProgLoss
from MS_Gen_Dataset import MazeSolvingDataset
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = torch.nn.CrossEntropyLoss()
# max_batches = 10

def overthink_train_loop(nn, dataloader, optimizer, max_batches=None, gradient_clipping=False, max_iter=20):
    running_loss = 0
    num_batches = 0
    new_iter = -1

    for batch, (X, y) in enumerate(dataloader):
        print(batch)
        # clear existing gradient
        optimizer.zero_grad()
        
        # train with random number of iterations
        partial_iter = random.randint(0, max_iter-1)
        training_iter = random.randint(1, max_iter - partial_iter)
        nn.set_iters(partial_iter, training_iter)

        pred = nn(X)
        loss = loss_fn(pred, y)
        loss.backward()
        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=1)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        
        if max_batches is not None:
            if num_batches == max_batches:
                break

    print(f"Running loss is {running_loss}")


if __name__ == '__main__':
    num_epochs = 5
    max_iter=20
    max_batches = 20

    training_MS_data = MazeSolvingDataset("data/maze_data_train_9")
    training_dataloader = DataLoader(training_MS_data, batch_size=10, shuffle=True)
    dt_nn = MazeSolvingNN_Recall_ProgLoss(num_iter=1).to(device)
    ff_optimizer = torch.optim.SGD(dt_nn.parameters(), lr=.001,
    weight_decay=2e-4, momentum=.9)

    for i in range(num_epochs):
        print(f"At epoch {i}, ", end="")
        overthink_train_loop(dt_nn, training_dataloader, ff_optimizer, gradient_clipping=True, 
        max_iter=max_iter, max_batches=max_batches)
    
    torch.save(dt_nn.state_dict(), 'models/MS_DT_ProgLoss.pth')