import math
from torch.distributions import Uniform, SigmoidTransform, AffineTransform, TransformedDistribution
import torch


def uniform_distribution(device, dimension=2.0):
    return Uniform(torch.tensor(-dimension/2, device=device), torch.tensor(dimension/2, device=device))


def clip_params(x, min_value, max_value):
    with torch.no_grad():
        x.clamp_(min=min_value, max=max_value)


def logistic_distribution(device, mean=0.0, std_dev=1.0):
    scale = std_dev * math.sqrt(3.) / math.pi

    base_distribution = Uniform(torch.tensor(0., device=device), torch.tensor(1., device=device))
    transforms = [SigmoidTransform().inv, AffineTransform(loc=torch.tensor(mean, device=device), scale=torch.tensor(scale, device=device))]
    logistic = TransformedDistribution(base_distribution, transforms)

    return logistic
