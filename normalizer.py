import torch.nn as nn
import numpy as np


"""Georg Götz, Ricardo Falcón Pérez, Sebastian J. Schlecht, and Ville Pulkki, "Neural network for multi-exponential sound energy decay analysis", The Journal of the Acoustical Society of America, 152(2), pp. 942-953, 2022, https://doi.org/10.1121/10.0013416."""
class Normalizer(nn.Module):
    """ Normalizes the data to have zero mean and unit variance for each feature."""

    def __init__(self, means, stds):
        super(Normalizer, self).__init__()
        self.means = means
        self.stds = stds
        self.eps = np.finfo(np.float32).eps

    def forward(self, x):
        out = x - self.means
        out = out / (self.stds + self.eps)

        return out