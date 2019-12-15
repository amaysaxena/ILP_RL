from __future__ import absolute_import, division, print_function, unicode_literals
import time
import numpy as np
from bb_solver import is_integer, BBSolver, DFSFringe, random_heuristic
from generate_instances import random_maxcut_instance, random_packing_instance, random_knapsack_instance
from bb_agent_torch import BBModel, get_heuristic_from

from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
import datetime

@contextmanager
def model_with_new_params(model, deltas, scale):
    original_params = []
    for p, delta in zip(model.parameters(), deltas):
        original_params.append(p.data)
        p.data = p.data + scale * delta
    yield model
    for p, old_p in zip(model.parameters(), original_params):
        p.data = old_p

class ESTrainer(object):
    def __init__(self, sigma, lr, gamma, horizon, model,
        num_epsiodes_per_update=10, num_noise=15000, max_updates=1000):

        self.sigma = sigma
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.horizon = horizon
        self.num_noise = num_noise
        self.noise_by_param = self.generate_random_noise(self.num_noise)
        self.n = num_epsiodes_per_update
        self.param_shapes = [tuple(x.data.shape) for x in self.model.parameters()]
        self.max_updates = max_updates
        self.heuristic = get_heuristic_from(self.model)

    def run_epsiode(self, instance):
        """
        Rolls out one episode of B&B. Returns the reward.
        """
        A, b, c = instance
        m, n = A.shape
        solver = BBSolver(np.array(A), np.array(b), np.array(c), DFSFringe, self.heuristic)
        reward = 0
        done = False
        t = 0
        while not done:
            if t > self.horizon:
                done = True
            else:
                problem, x, value, is_int, branched = solver.step()
                if is_int:
                    reward += (self.gamma ** t) * 1000
                    done = True
                else:
                    reward += (self.gamma ** t) * (-10)
            t += 1
        return reward, solver.num_problems_expanded

    def get_param_shapes(self):
        result = []
        for p in self.model.parameters():
            result.append(tuple(p.shape))
        return result

    def generate_random_noise(self, num_samples):
        shapes = self.get_param_shapes()
        noise = []
        for shape in shapes:
            noise.append(torch.normal(0.0, 1.0, size=(num_samples,) + shape))
        return noise

    def update_weights(self, delta, scale):
        for p, d in zip(self.model.parameters(), delta):
            p.data += scale * d

    def update_once(self, As, bs, cs):
        noise_ind = torch.randint(high=self.num_noise, size=(self.n,))
        to_update = [torch.zeros(size=shape) for shape in self.param_shapes]
        total_reward = 0

        for ep, (ind, A, b, c) in enumerate(zip(noise_ind, As, bs, cs)):
            eps = [noise[ind] for noise in self.noise_by_param]

            with model_with_new_params(self.model, eps, self.sigma):
                rew, exp = self.run_epsiode((A, b, c))
                total_reward += rew
            print("Episode", ep, "Reward", rew, "Nodes Expanded", exp)

            to_update = [param + e * rew for param, e in zip(to_update, eps)]
        self.update_weights(to_update, self.lr / (self.n * self.sigma))
        print()
        print("Average Reward:", total_reward / self.n)
        print()

    def train(self, training_data):
        print("Beginning Training...")
        t0 = time.time()
        As, bs, cs = training_data
        for it in range(self.max_updates):
            print("========= Iteration " + str(it) + " =========")
            ind = torch.randint(high=len(As), size=(self.n,))
            A, b, c = As[ind], bs[ind], cs[ind]
            self.update_once(A, b, c)
        t1 = time.time()
        print("Training Time:", dattime.timedelta(seconds=t1 - t0))

def generate_and_save_training_dataset():
    As, bs, cs = [], [], []
    for _ in range(30):
        A, b, c = random_packing_instance(20, 20, list(range(6)), list(range(1, 10)))
        As.append(A)
        bs.append(b)
        cs.append(c)
    np.savez('data/train', A=np.array(As), b=np.array(bs), c=np.array(cs))

def main():
    data = np.load('data/train.npz')
    As, bs, cs = data['A'], data['b'], data['c']
    m, n = As[0].shape
    As, bs, cs = torch.FloatTensor(As), torch.FloatTensor(bs), torch.FloatTensor(cs)
    model = BBModel(n, m)
    trainer = ESTrainer(0.2, 0.01, 0.999, 500, model)
    trainer.train((As, bs, cs))

if __name__ == '__main__':
    main()
