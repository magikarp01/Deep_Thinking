import torch
from torch import nn
from PS_Architecture import PrefixSumNN_DT, PrefixSumNN_FF
from PS_Gen_Dataset import PrefixSumDataset

from torch.utils.data import Dataset, DataLoader


def get_accuracy(pred, y):
    logits = torch.nn.functional.softmax(pred, dim=1)[:, 1]
    rounded = torch.round(logits)
    return torch.sum(rounded == y)/y.numel()

def test_model(nn, dataloader, loss_fn = nn.CrossEntropyLoss(), verbose=False):
    torch.no_grad()
    running_loss = 0
    running_accuracy = 0
    num_batches = 0
    
    for batch, (X, y) in enumerate(dataloader):
        pred = nn(X)
        loss = loss_fn(pred, y)
        
        running_accuracy += get_accuracy(pred, y)
        num_batches += 1
        running_loss += loss.item()

    if verbose:
        print(f"Testing loss is {running_loss}")
        print(f"Testing accuracy is {running_accuracy/num_batches}")
    return running_loss, running_accuracy/num_batches


if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    testing_PS_data = PrefixSumDataset("test_X.pt", "test_y.pt")
    testing_dataloader = DataLoader(testing_PS_data, batch_size=10, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()

    ff_nn = PrefixSumNN_FF(num_iter=2).to(device)
    ff_nn.load_state_dict(torch.load('PS_FF.pth'))

    init_dt_nn = PrefixSumNN_DT(num_iter=2).to(device)
    init_dt_nn.load_state_dict(torch.load('PS_DT.pth'))
    dt_nn = PrefixSumNN_DT(num_iter=1).to(device)
    dt_nn.expand_iterations(init_dt_nn)

    batch, (test_input, test_sol) = next(enumerate(testing_dataloader))
    # print(test_input)
    ff_output = ff_nn(test_input)
    dt_output = dt_nn(test_input)

    ff_probs = torch.nn.functional.softmax(ff_output, dim=1)[0][1]
    dt_probs = torch.nn.functional.softmax(dt_output, dim=1)[0][1]
    # print(test_input)
    # print(test_sol)
    # print(ff_probs)
    # print(torch.round(ff_probs))
    print(get_accuracy(ff_output, test_sol))
    print(get_accuracy(dt_output, test_sol))

    test_model(ff_nn, testing_dataloader)
    test_model(dt_nn, testing_dataloader)
    