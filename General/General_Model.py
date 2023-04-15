import torch
from torch import nn
from torch.nn import functional as F

from learner import Learner
from copy import deepcopy


class Memory_Meta(nn.Module):
    def __init__(self, config):
        super(Memory_Meta, self).__init__()
        self.net = Learner(config)

    def forward(self, x_spt, y_spt):
        net = deepcopy(self.net)
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        return loss

    def test(self, x_spt):
        net = deepcopy(self.net)
        with torch.no_grad():
            # print(x_spt.shape)
            logits = net(x_spt)
            logits = F.softmax(logits)
        return logits[:, 1].cpu().numpy()
