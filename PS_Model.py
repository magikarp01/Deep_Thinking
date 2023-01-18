import torch
from torch import nn
from PS_Architecture import PrefixSumNN_DT, PrefixSumNN_FF
from PS_Gen_Dataset import PrefixSumDataset

from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

training_PS_data = PrefixSumDataset("train_X.pt", "train_y.pt")
training_dataloader = DataLoader(training_PS_data, batch_size=150, shuffle=True)

testing_PS_data = PrefixSumDataset("test_X.pt", "test_y.pt")
testing_dataloader = DataLoader(testing_PS_data, batch_size=100, shuffle=True)

ff_nn = PrefixSumNN_FF(num_iter=2).to(device)
dt_nn = PrefixSumNN_DT(num_iter=2).to(device)

# print(ff_nn)
loss_fn = nn.CrossEntropyLoss()

ff_optimizer = torch.optim.Adam(ff_nn.parameters(), weight_decay=2e-4, lr=.001)
dt_optimizer = torch.optim.Adam(dt_nn.parameters(), weight_decay=2e-4, lr=.001)

def train_loop(nn, optimizer):

    running_loss = 0
    for batch, (X, y) in enumerate(training_dataloader):
        # clear existing gradient
        optimizer.zero_grad()
        
        pred = nn(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 6 == 5:
            print(f"At batch {batch}, running loss is {running_loss}")
            running_loss = 0
        # torch.nn.utils.clip_grad_norm(nn.parameters(), args.clip)

# train_loop(ff_nn, ff_optimizer)
# train_loop(dt_nn, dt_optimizer)
# torch.save(ff_nn.state_dict(), 'PS_FF.pth')
# torch.save(dt_nn.state_dict(), 'PS_DT.pth')


