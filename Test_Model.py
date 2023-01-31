import torch
from torch import nn
from PS_Architecture import PrefixSumNN_DT, PrefixSumNN_FF
from MS_Architecture import MazeSolvingNN_FF, MazeSolvingNN_DT
from PS_Gen_Dataset import PrefixSumDataset
from MS_Gen_Dataset import MazeSolvingDataset
from MS_Architecture_Recall import MazeSolvingNN_Recall
from MS_Architecture_Overthinking_Recall import MazeSolvingNN_Recall_ProgLoss

from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils



def get_accuracy(pred, y):
    logits = torch.nn.functional.softmax(pred, dim=1)[:, 1]
    rounded = torch.round(logits)
    return torch.sum(rounded == y)/y.numel()

def test_model(nn, dataloader, loss_fn = nn.CrossEntropyLoss(), verbose=False, 
max_batches=None):
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
        if max_batches is not None and batch >= max_batches:
            break

    if verbose:
        print(f"Testing loss is {running_loss}")
        print(f"Testing accuracy is {running_accuracy/num_batches}")
    return running_loss, running_accuracy/num_batches


if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # testing_data = PrefixSumDataset("data/PS_test_X.pt", "data/PS_test_y.pt")
    # testing_data = MazeSolvingDataset("data/maze_data_train_9")

    testing_data = MazeSolvingDataset("data/maze_data_test_13")
    # indices = torch.arange(1000)
    # te_10k = data_utils.Subset(testing_data, indices)

    # testing_dataloader = DataLoader(te_10k, batch_size=10, shuffle=True)
    testing_dataloader = DataLoader(testing_data, batch_size=10, shuffle=False)
    torch.save(testing_dataloader, 'maze_13_dataloader.pth')
    # testing_dataloader = torch.load('data/maze_13_dataloader.pth')

    loss_fn = nn.CrossEntropyLoss()

    # ff_nn = PrefixSumNN_FF(num_iter=2).to(device)
    # ff_nn.load_state_dict(torch.load('models/PS_FF.pth'))
    # ff_nn = MazeSolvingNN_FF(num_iter=20).to(device)
    # ff_nn.load_state_dict(torch.load('models/MS_FF_20.pth'))

    init_dt_nn = PrefixSumNN_DT(num_iter=2).to(device)
    init_dt_nn.load_state_dict(torch.load('models/PS_DT.pth'))
    # init_dt_nn = MazeSolvingNN_DT(num_iter=10).to(device)
    # init_dt_nn.load_state_dict(torch.load('models/MS_DT_20.pth'), strict=False)
    # init_dt_nn = MazeSolvingNN_Recall_ProgLoss(num_iter=20).to(device)
    # init_dt_nn.load_state_dict(torch.load('models/MS_Recall_ProgLoss.pth'), strict=False)

    # init_dt_nn = MazeSolvingNN_Recall(num_iter=20).to(device)
    # init_dt_nn.load_state_dict(torch.load('models/MS_Recall_20.pth'))

    # dt_nn = MazeSolvingNN_DT(num_iter=30).to(device)
    # dt_nn.expand_iterations(init_dt_nn)

    # dt_nn = MazeSolvingNN_Recall_ProgLoss(num_iter=30).to(device)
    # dt_nn.expand_iterations(init_dt_nn)

    # dt_nn = MazeSolvingNN_Recall(num_iter=30).to(device)
    # dt_nn.expand_iterations(init_dt_nn)

    # print("For FF_NN: ")
    # test_model(ff_nn, testing_dataloader, verbose=True)
    # print()

    print("For initial DT_NN: ")
    test_model(init_dt_nn, testing_dataloader, verbose=True)
    
    print("For altered DT_NN: ")
    test_model(dt_nn, testing_dataloader, verbose=True)
    