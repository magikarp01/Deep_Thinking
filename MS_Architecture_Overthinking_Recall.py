# Architecture for Maze Solving models

import torch
from torch import nn
from torch.nn.functional import relu
from MS_Architecture_Recall import MazeSolvingNN_Recall
import MS_Architecture

in_channels = 128
out_channels = 128

# only thing that needs to change is the forward method
class MazeSolvingNN_Recall_ProgLoss(MazeSolvingNN_Recall):
    
    def set_iters(self, partial_iters, training_iters):
        self.partial_iters = partial_iters
        self.training_iters = training_iters

    def forward(self, x):
        prev_output = self.l1(x)
        # don't record gradient here for partial solution
        with torch.no_grad():
            for i in range(self.partial_iters):
                # concatenate previous output with input
                recall_input = torch.cat((prev_output, x), dim=1)
                next_input = self.recall_conv(recall_input)
                prev_output = self.iter_block(next_input)

        # record gradient here
        for i in range(self.training_iters):
            # concatenate previous output with input
            recall_input = torch.cat((prev_output, x), dim=1)
            next_input = self.recall_conv(recall_input)
            prev_output = self.iter_block(next_input)
        
        return self.end_layers(prev_output)
    
    
