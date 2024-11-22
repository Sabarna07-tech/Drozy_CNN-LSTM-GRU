import torch
import numpy as np

def normalizelayer(data):
    eps = 1e-05
    a_mean = data - torch.mean(data, [0, 2, 3], True).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))
    b = torch.div(a_mean, torch.sqrt(torch.mean((a_mean)**2, [0, 2, 3], True) + eps).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3))))
    return b