import torch
import torch.nn as nn

class Joint_model(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.pred = nn.Linear(3, 154)
    def forward(self,x):
        # x :b, t, v, c
        x = self.pred(x)
        return x