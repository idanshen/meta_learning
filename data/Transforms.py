import math
import numpy as np

import torch
from torch.nn import functional as F


class FlattenData(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample
        image = image.reshape(-1)
        return image

class ProjectData:
    def __init__(self, seq_len, num_tasks, input_dim, label_dim, device):
        self.seq_len = seq_len
        self.num_tasks = num_tasks
        self.input_dim = input_dim
        self.label_dim = label_dim

        # Create projection matrices
        self.sigma = math.sqrt(1.0 / input_dim)

        self.train_random_projection = self.sigma * torch.randn(num_tasks, input_dim, input_dim, device='cpu', dtype=torch.bfloat16)
        self.val_random_projection = self.sigma * torch.randn(num_tasks, input_dim, input_dim, device='cpu', dtype=torch.bfloat16)

    def __call__(self, input, label, val=False):
        assert input.shape[0] % self.seq_len == 0, "Input should include batch X num_seq samples"
        b, n = input.shape
        input = input.reshape(b // self.seq_len, self.seq_len, n)
        label = label.reshape(b // self.seq_len, self.seq_len, 1)

        # Project inputs
        if val:
            indices = np.random.choice(self.num_tasks, b // self.seq_len, True)
            matrices = self.val_random_projection[indices, :, :].to(device=input.device, non_blocking=True)
        else:
            indices = np.random.choice(self.num_tasks, b // self.seq_len, True)
            matrices = self.train_random_projection[indices, :, :].to(device=input.device, non_blocking=True)

        projected_input = torch.einsum("b s i, b i n -> b s n", input.type(torch.bfloat16), matrices)
        # projected_input = input.type(torch.bfloat16)
        # Append zero to the beginning and cut out the last label
        one_hot_labels = F.one_hot(label, num_classes=self.label_dim).squeeze()
        to_concat = torch.concat([torch.zeros((b//self.seq_len, 1, self.label_dim), device=label.device), one_hot_labels[:,:-1,:]], dim=1)
        input = torch.concat([projected_input, to_concat], dim=-1)
        return input, label
