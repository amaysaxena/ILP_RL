from __future__ import absolute_import, division, print_function, unicode_literals
import time
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class BBModel(nn.Module):
    def __init__(self, num_vars, num_constraints):
        super(BBModel, self).__init__()
        self.n = num_vars
        self.m = num_constraints
        self.constraint_encoder = nn.Sequential(
            nn.Linear(self.n + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh())

    def forward(self, A, b, c, x_val, branch_candidates):
        """
        Arguments:
            (A, b, c): ILP Problem in standard form.
            x_val: solution to LP relaxation of (A, b, c).
            branch_candidates: indices for variables to choose from to
                branch on.
        """
        encoded_constraints = self.constraint_encoder(torch.cat([A, b[:, None]], dim=1))

        constraint_lt = torch.cat([torch.eye(self.n)[branch_candidates],
                                   torch.floor(x_val[branch_candidates])[:, None]], dim=1)

        constraint_gt = torch.cat([-torch.eye(self.n)[branch_candidates],
                                   -torch.ceil(x_val[branch_candidates])[:, None]], dim=1)
        encoded_lt = self.constraint_encoder(constraint_lt)
        encoded_gt = self.constraint_encoder(constraint_gt)
        encoded_candidates = torch.max(encoded_lt, encoded_gt)

        dot_prods = torch.mm(encoded_candidates, encoded_constraints.t())
        candidate_features = (1 / self.m) * torch.sum(dot_prods, dim=1)
        return F.softmax(candidate_features)


if __name__ == '__main__':
    m, n, candidates = 30, 25, 5
    A = torch.ones((n, m))
    b = torch.Tensor(np.arange(n, dtype=np.float32))
    c = torch.Tensor(np.arange(m, dtype=np.float32))
    x = torch.ones(m)
    branch_candidates = torch.LongTensor(np.arange(candidates, dtype=np.int32))

    model = BBModel(m, n)
    t0 = time.time()
    out = model(A, b, c, x, branch_candidates)
    print(time.time() - t0)
    print(out)
    for p in model.parameters():
        if len(p.shape) == 2:
            p[:, :] = 0
    for p in model.parameters():
        print(p)
