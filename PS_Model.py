import torch
from torch import nn
from PS_Architecture import PrefixSumNN_DT, PrefixSumNN_FF
from PS_Gen_Dataset import PrefixSumDataset

from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

loss_fn = nn.CrossEntropyLoss()

training_PS_data = PrefixSumDataset("data/PS_train_X.pt", "data/PS_train_y.pt")
training_dataloader = DataLoader(training_PS_data, batch_size=150, shuffle=True)

def get_accuracy(pred, y):
    logits = torch.nn.functional.softmax(pred, dim=1)[:, 1]
    rounded = torch.round(logits)
    return torch.sum(rounded == y)/y.numel()

def train_loop(nn, optimizer):
    running_loss = 0
    running_accuracy = 0
    num_batches = 0
    for batch, (X, y) in enumerate(training_dataloader):
        # clear existing gradient
        optimizer.zero_grad()
        
        pred = nn(X)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=1)
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += get_accuracy(pred, y)
        num_batches += 1

    print(f"Running loss is {running_loss}")
    print(f"Running accuracy is {running_accuracy/num_batches}")


num_epochs = 5
if __name__=="__main__":
    print("Training ff_nn")
    ff_nn = PrefixSumNN_FF(num_iter=2).to(device)
    ff_optimizer = torch.optim.Adam(ff_nn.parameters(), weight_decay=2e-4, 
    lr=.001)
    ff_scheduler = torch.optim.lr_scheduler.ExponentialLR(ff_optimizer, gamma=0.9)
    for i in range(num_epochs):
        print(f"At epoch {i}, ", end="")
        train_loop(ff_nn, ff_optimizer)
    torch.save(ff_nn.state_dict(), 'models/PS_FF.pth')

    print("Training dt_nn")
    dt_nn = PrefixSumNN_DT(num_iter=2).to(device)
    dt_optimizer = torch.optim.Adam(dt_nn.parameters(), weight_decay=2e-4, lr=.001)
    for i in range(num_epochs):
        print(f"At epoch {i}, ", end="")
        train_loop(dt_nn, dt_optimizer)
    torch.save(dt_nn.state_dict(), 'models/PS_DT.pth')


