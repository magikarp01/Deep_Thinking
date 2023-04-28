# Deep_Thinking

A summary of my replication efforts and continuing research for the two papers, “Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks” and “End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking”. 

A formal writeup is here: https://docs.google.com/document/d/1Ssn8vlZJX_Hndt1ide_V_ojr1Bl9z0M3vqX2cD8b_lg/edit?usp=sharing.

Deep Thinking models are pseudo-recurrent neural networks that replace multiple layers with one "recurrent" layer: one layer that continues
feeding output back to itself. This allows the network to scale up to any arbitrary "effective depth": for harder problems, networks can 
theoretically increase their effective depth and gain better accuracy. I train a deep thinking CNN on solving 9x9 mazes, and then scale up
the effective depth to solve 13x13 mazes. 

I also continue research by exploring the effectiveness of multiple layers, different versions for deep thinking 
(with more complex model architecture and different training regimes), and attempt to visualize the loss function of deep thinking models
according to research done in "Visualizing the Loss Landscape of Neural Nets" (https://arxiv.org/abs/1712.09913).
