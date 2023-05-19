import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x_hat):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y) # input image feature
        # x_feats = self.extract_feats(x) # input image feature
        y_hat_feats = self.extract_feats(y_hat) # generated image feature
        y_feats = y_feats.detach()
        x_hat_feats = self.extract_feats(x_hat) # generated image feature
        # x_feats = x_feats.detach()
        loss = 0
        for i in range(n_samples):
            diff_input_y = y_hat_feats[i].dot(y_feats[i]) # similarity of generated image feature and input image feature
            diff_input_x = x_hat_feats[i].dot(y_feats[i]) # similarity of generated image feature and input image feature
            loss += (1 - diff_input_y)+(1-diff_input_x)

        return (loss / n_samples) / 2