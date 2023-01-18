import torch
from torch import nn
from PS_Architecture import PrefixSumNN_DT, PrefixSumNN_FF
from PS_Gen_Dataset import PrefixSumDataset

from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

testing_PS_data = PrefixSumDataset("test_X.pt", "test_y.pt")
testing_dataloader = DataLoader(testing_PS_data, batch_size=100, shuffle=True)

loss_fn = nn.CrossEntropyLoss()

ff_nn = PrefixSumNN_FF(num_iter=2).to(device)
ff_nn.load_state_dict(torch.load('PS_FF.pth'))

dt_nn = PrefixSumNN_DT(num_iter=2).to(device)
dt_nn.load_state_dict(torch.load('PS_DT.pth'))

batch, (test_input, test_sol) = next(enumerate(testing_dataloader))
# print(test_input)
ff_output = ff_nn(test_input)
dt_output = dt_nn(test_input)
# print(ff_output)
# print(dt_output)

print(f"ff loss: {loss_fn(ff_output, test_sol)}")
print(f"dt loss: {loss_fn(dt_output, test_sol)}")