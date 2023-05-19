import torch
from torch import nn


class RegLoss(nn.Module):
    def __init__(self, lambda_s=1):
        super(RegLoss, self).__init__()
        self.lambda_s = lambda_s

    def forward(self, ws_id, sf_d1, stage=18):

        l_ws = torch.zeros(ws_id.shape[0]).cuda()
        for i in range(1, min(ws_id.shape[1], stage+1)):
            delta = ws_id[:, i, :] - ws_id[:, 0, :]  # ∆i = WSID[i] - WSID[1]
            l_ws += torch.norm(delta, p=2, dim=1)  # Σ(||∆i||^2)

        l_sfd1 = torch.zeros(sf_d1.shape[0]).cuda()
        for i in range(min(sf_d1.shape[1], stage+1)):
            l_sfd1 += torch.norm(sf_d1[:, i, :], p=2, dim=1)  # Σ(||SFD1[j]||^2)

        l_reg = l_ws + (self.lambda_s * l_sfd1)
        return torch.mean(l_reg)
