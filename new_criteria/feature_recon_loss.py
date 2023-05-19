import torch
from torch import nn

class FeatureReconLoss(nn.Module):

    def __init__(self, feature_extractor, mask):
        super(FeatureReconLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.mask = mask

    def forward(self, x, y):
        x_features = self.feature_extractor(x * self.mask)
        y_features = self.feature_extractor(y * self.mask)
        loss = 0
        for xf, yf in zip(x_features, y_features):
            loss += nn.MSELoss()(xf, yf)
        return loss / len(x_features)
