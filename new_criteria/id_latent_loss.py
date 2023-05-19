import torch
from torch import nn


class IDLatentLoss(nn.Module):

    def __init__(self):
        super(IDLatentLoss, self).__init__()

    def forward(self, latent_a, latent_b):
        diff = latent_a - latent_b
        return torch.mean(torch.norm(diff, p=2, dim=1))