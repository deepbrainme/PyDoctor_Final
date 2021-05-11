import torch

import numpy as np

def gray_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().repeat(3,1,1).unsqueeze(0)

def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)


def torch_to_numpy(a: torch.Tensor):
    return a.squeeze(0).permute(1,2,0).numpy()

