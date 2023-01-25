import torch
# from torch import nn
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = torch.nn.CrossEntropyLoss()
# max_batches = 10

def overthink_train_loop(nn, dataloader, optimizer, max_batches=None, gradient_clipping=False, iter_range=[1, 20]):
    running_loss = 0
    num_batches = 0
    new_iter = -1

    for batch, (X, y) in enumerate(dataloader):
        # clear existing gradient
        optimizer.zero_grad()
        
        # train with random number of iterations
        new_iter = random.randint(iter_range[0], iter_range[1])
        nn.change_iter(new_iter)

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
