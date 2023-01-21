import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss()
# max_batches = 10

def train_loop(nn, dataloader, optimizer, max_batches=None, gradient_clipping=False):
    running_loss = 0
    num_batches = 0
    for batch, (X, y) in enumerate(dataloader):
        # clear existing gradient
        optimizer.zero_grad()
        
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
