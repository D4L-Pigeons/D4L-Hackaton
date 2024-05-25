import torch


class GaussianKLD(nn.Module):
    def __init__(self):
        super(GaussianKLD, self).__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
