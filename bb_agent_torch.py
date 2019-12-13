from __future__ import absolute_import, division, print_function, unicode_literals
import time
import numpy as np
from bb_solver import is_integer, BBSolver, DFSFringe, random_heuristic
from generate_instances import random_maxcut_instance, random_packing_instance, random_knapsack_instance


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

# Example of how an agent may be used as a heuristic.
def get_heuristic_from(model):
    def rl_heuristic(A, b, c, x):
        candidates = torch.LongTensor([i for i, v in enumerate(x) if not is_integer(v)])
        A, b, c, x = torch.Tensor(A), torch.Tensor(b), torch.Tensor(c), torch.Tensor(x)
        action_dist = model(A, b, c, x, candidates)
        # During training, we should sample from the output distribution, but during
        # testing, we should just take the argmax.
        return candidates[torch.multinomial(action_dist, 1).item()].item() 
        # The above line just returns one sample from a categorical distribution
        # with probs given in the vector action_dist. I had to use item() to turn a 1 element tensor
        # into a number.
        
    return rl_heuristic

def rl_heuristic_example():
    A, b, c = random_maxcut_instance(5, 10, list(9*np.random.uniform(size=100)))
    m, n = A.shape
    rl_heuristic = get_heuristic_from(BBModel(n, m))
    solver = BBSolver(A, b, c, DFSFringe, rl_heuristic)
    sol, obj = solver.solve()
    print("Solution:", sol)
    print("Objective:", obj)
    print("Problems Expanded:", solver.num_problems_expanded)

if __name__ == '__main__':
    rl_heuristic_example()
