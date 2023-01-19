import torch
from torch import nn
from PS_Architecture import PrefixSumNN_DT, PrefixSumNN_FF
from PS_Gen_Dataset import PrefixSumDataset

from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def extend_ps_nn(file, nn_class, start_iter, end_iter):
    # start_nn = nn_class(num_iter=start_iter).to(device)
    nn = nn_class(num_iter=end_iter).to(device)
    # iterations 

if __name__ == '__main__':
    start_nn = PrefixSumNN_DT(num_iter=2)
    start_nn.load_state_dict(torch.load('PS_DT.pth'))
    print(start_nn.state_dict().keys())
    state_dictionary = start_nn.state_dict()
    print(state_dictionary['iterations.0.conv1.weight'] == state_dictionary['iterations.1.conv1.weight'])
    print(state_dictionary['iter_block.conv2.weight'] == state_dictionary['iterations.1.conv2.weight'])