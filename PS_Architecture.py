import torch
from torch import nn
from torch.nn.functional import relu


# num channels in the basic block convolutional layers
in_channels = 120
out_channels = 120

# iteration block of 2 residual blocks
class IterationBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # kernel_size = 3 according to paper, padding = 1 to maintain size
        # next, can try bias=True
        self.conv1 = nn.Conv1d(in_channels=120, out_channels=120,
        kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=120, out_channels=120,
        kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv1d(in_channels=120, out_channels=120,
        kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv1d(in_channels=120, out_channels=120,
        kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        # regular relu of convolution output
        l1_out = relu(self.conv1(x))
        # add x for shortcut
        l2_out = relu(self.conv2(l1_out) + x)
        
        l3_out = relu(self.conv3(l2_out))
        l4_out = relu(self.conv4(l3_out) + l2_out)

        return l4_out

def make_iter_block():
    return IterationBlock()

class PrefixSumNN_FF(nn.Module):
    def __init__(self, num_iter):
        super().__init__()
        self.l1 = nn.Conv1d(in_channels = 1, out_channels = 120, 
        kernel_size=3, padding=1, bias=False)

        iter_blocks = [make_iter_block() for i in range(num_iter)]
        self.iterations = nn.Sequential(*iter_blocks)

        # in_channels should be 120, out_channels = 60
        self.l2 = nn.Conv1d(in_channels=120, out_channels = 60, 
        kernel_size=3, padding=1, bias=False)

        # in_channels 60, out_channels 30
        self.l3 = nn.Conv1d(in_channels=60, out_channels = 30, 
        kernel_size=3, padding=1, bias=False)

        self.l4 = nn.Conv1d(in_channels=30, out_channels = 2, 
        kernel_size=3, padding=1, bias=False)
        
        self.layers = nn.Sequential(self.l1, self.iterations, self.l2, 
        self.l3, self.l4)

    def forward(self, x):
        return self.layers(x)
    

class PrefixSumNN_DT(nn.Module):
    def __init__(self, num_iter):
        super().__init__()
        self.l1 = nn.Conv1d(in_channels = 1, out_channels = 120, 
        kernel_size=3, padding=1, bias=False)

        iter_block = make_iter_block()
        # want to repeat the iter_block
        self.iterations = nn.Sequential(*[iter_block for i in range(num_iter)])

        # in_channels should be 120, out_channels = 60
        self.l2 = nn.Conv1d(in_channels = 120, out_channels = 60, 
        kernel_size=3, padding=1, bias=False)

        # in_channels 60, out_channels 30
        self.l3 = nn.Conv1d(in_channels=60, out_channels = 30, 
        kernel_size=3, padding=1, bias=False)

        self.l4 = nn.Conv1d(in_channels=30, out_channels = 2, 
        kernel_size=3, padding=1, bias=False)
        
        self.layers = nn.Sequential(self.l1, self.iterations, self.l2, 
        self.l3, self.l4)


    def forward(self, x):
        return self.layers(x)


# def expand_dt_net()