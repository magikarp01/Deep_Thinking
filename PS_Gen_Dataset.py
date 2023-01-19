import torch
import random
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def gen_string(num_bits):
    return np.array([random.randint(0, 1) for i in range(num_bits)])

def prefix_sum(bit_list):
    num_bits = len(bit_list)
    sum_list = np.zeros_like(bit_list)
    sum_list[0] = bit_list[0]
    for i in range(1, num_bits):
        sum_list[i] = (sum_list[i-1] + bit_list[i]) % 2
    
    return sum_list


def gen_data(X_name="train_X.pt", y_name="train_y.pt", num_samples=10000, size=32):
    X_data = torch.zeros([num_samples, 1, size], dtype=torch.float32)
    y_data = torch.zeros([num_samples, size], dtype=torch.long)
    # y_data = torch.zeros([num_samples, 1, size], dtype=torch.long)
    for i in tqdm(range(num_samples)):
        new_str = np.expand_dims(gen_string(size),axis=0)
        X_data[i] = torch.tensor(new_str, dtype = torch.float32, requires_grad = True)
        output_str = prefix_sum(new_str)
        # output_str = np.expand_dims(prefix_sum(new_str),axis=0)
        y_data[i] = torch.tensor(output_str, dtype = torch.long)

    print("done")

    torch.save(X_data, X_name)
    torch.save(y_data, y_name)

def gen_alt_data(X_name="train_alt_X.pt", y_name="train_alt_y.pt", num_samples=10000, size=32):
    X_data = torch.zeros([num_samples, 1, size], dtype=torch.float32)
    # y_data = torch.zeros([num_samples, size], dtype=torch.long)
    y_data = torch.zeros([num_samples, 1, size], dtype=torch.float32)
    for i in tqdm(range(num_samples)):
        new_str = np.expand_dims(gen_string(size),axis=0)
        X_data[i] = torch.tensor(new_str, dtype = torch.float32, requires_grad = True)
        # output_str = prefix_sum(new_str)
        output_str = np.expand_dims(prefix_sum(new_str),axis=0)
        y_data[i] = torch.tensor(output_str, dtype = torch.float32)

    print("done")

    torch.save(X_data, X_name)
    torch.save(y_data, y_name)

if __name__ == "__main__":
    gen_data("train_X.pt", "train_y.pt", num_samples=10000, size=16)
    gen_data("test_X.pt", "test_y.pt", num_samples=10000, size=132)
    
    # gen_alt_data("train_alt_X.pt", "train_alt_y.pt", num_samples=10000)
    # gen_alt_data("test_alt_X.pt", "test_alt_y.pt", num_samples=10000, size=132)

device = "cuda" if torch.cuda.is_available() else "cpu"

class PrefixSumDataset(Dataset):
    def __init__(self, X_file, y_file):
        self.X_data = torch.load(X_file).to(device)
        self.y_data = torch.load(y_file).to(device)
    
    def __len__(self):
        return self.X_data.shape[0]
    
    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]


# def makePSDDataLoader():
#     training_PS_data = PrefixSumDataset("train_X.npy", "train_y.npy")
#     return DataLoader(training_PS_data, batch_size=64, shuffle=True)

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# print(train_features)
# print(train_labels)