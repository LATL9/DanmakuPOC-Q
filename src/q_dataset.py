from common import *

import torch

class QDataset(torch.utils.data.Dataset):
    def __init__(self, inps, outs):
        self.inps = inps
        self.outs = outs

    def __len__(self):
        return len(self.inps) # inps and outs are same length

    def __getitem__(self, idx):
        return self.inps[idx], self.outs[idx]
