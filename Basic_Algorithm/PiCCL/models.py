import torch.nn as nn
import torch


class twins(nn.Module):

    def __init__(self, base_encoder, size=32, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(weights=None)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        if size==32:
            shape = (3, 64, 3, 1, 1)
            print(f"picture size is 32 by 32, will change conv1 to {shape[2]}*{shape[2]} stride {shape[3]}, removing maxpooling layer")
            self.enc.conv1 = nn.Conv2d(*shape, bias=False)
            self.enc.maxpool = nn.Identity()
        elif size==96:
            shape = (3, 64, 5, 2, 1)
            print(f"picture size is 96 by 96, will change conv1 to {shape[2]}*{shape[2]} stride {shape[3]}")
            self.enc.conv1 = nn.Conv2d(*shape, bias=False)
            #self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection

class SimSiam_network(nn.Module):

    def __init__(self, base_encoder, size=32, dim=2048, pred_dim=512):
        super().__init__()
        self.enc = base_encoder(pretrained=False)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        if size==32:
            shape = (3, 64, 3, 1, 1)
            print(f"picture size is 32 by 32, will change conv1 to {shape[2]}*{shape[2]} stride {shape[3]}, removing maxpooling layer")
            self.enc.conv1 = nn.Conv2d(*shape, bias=False)
            self.enc.maxpool = nn.Identity()
        elif size==96:
            shape = (3, 64, 5, 2, 1)
            print(f"picture size is 96 by 96, will change conv1 to {shape[2]}*{shape[2]} stride {shape[3]}")
            self.enc.conv1 = nn.Conv2d(*shape, bias=False)
            self.enc.maxpool = nn.Identity()

        prev_dim = self.enc.fc.weight.shape[1]
        self.enc.fc = nn.Identity()        
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        prediction = self.predictor(projection)
        return feature, (projection, prediction.detach())


