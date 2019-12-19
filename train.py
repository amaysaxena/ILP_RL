from __future__ import absolute_import, division, print_function, unicode_literals
import time
import numpy as np
from bb_solver import is_integer, BBSolver, DFSFringe, BFSFringe, random_heuristic, nonint_heuristic
from generate_instances import random_maxcut_instance, random_packing_instance, random_knapsack_instance
from bb_agent_torch import BBModel, BBModel2, BBModel2Big, get_heuristic_from

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
    def __init__(self, sigma, lr, gamma, horizon, model, fringe_maker,
        num_epsiodes_per_update=10, num_noise=20000, max_updates=1100,
        save_every=50):

        self.fringe_maker = fringe_maker
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
        self.save_every = save_every

    def run_epsiode(self, instance):
        """
        Rolls out one episode of B&B. Returns the reward.
        """
        A, b, c = instance
        m, n = A.shape
        solver = BBSolver(np.array(A), np.array(b), np.array(c), self.fringe_maker, self.heuristic)
        reward = 0
        done = False
        t = 0
        while not done:
            if t > self.horizon:
                done = True
            else:
                problem, x, value, is_int, branched = solver.step()
                if solver.fringe.isempty():
                    reward += (self.gamma ** t) * 2000
                    done = True
                elif is_int:
                    reward += (self.gamma ** t) * 50
                else:
                    reward += (self.gamma ** t) * (-15)
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
            eps = torch.normal(0.0, 1.0, size=(num_samples // 2,) + shape)
            eps = torch.cat([eps, -eps])
            noise.append(eps)
        return noise

    def update_weights(self, delta, scale):
        for p, d in zip(self.model.parameters(), delta):
            p.data += scale * d

    def update_once(self, As, bs, cs):
        noise_ind = torch.randint(high=self.num_noise // 2, size=(self.n // 2,))
        noise_ind = torch.cat([noise_ind, self.num_noise // 2 + noise_ind])
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
        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y--%H:%M:%S")
        print("Beginning Training at " + dt_string + "...")
        t0 = time.time()
        As, bs, cs = training_data
        for it in range(self.max_updates):
            print("========= Iteration " + str(it) + " =========")
            print("Time since start:", datetime.timedelta(seconds=time.time() - t0))
            ind = torch.randint(high=len(As), size=(1,)).item()
            A, b, c = As[ind], bs[ind], cs[ind]
            self.update_once([A] * self.n, [b] * self.n, [c] * self.n)
            
            if it % self.save_every == 0:
                self.save_model('model-'+dt_string+'-iter-'+str(it))

        t1 = time.time()
        print("Training Time:", datetime.timedelta(seconds=t1 - t0))

    def save_model(self, name):
        torch.save(self.model.state_dict(), 'models/' + name)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

def generate_and_save_training_dataset(name):
    As, bs, cs = [], [], []
    for _ in range(30):
        #A, b, c = random_knapsack_instance(20, 10*np.random.uniform(size=100), 10*np.random.uniform(size=100), 20 + 40*np.random.uniform(size=100))
        #A, b, c = random_maxcut_instance(20, 40, list(10*np.random.uniform(size=100)))
        A, b, c = random_packing_instance(10, 10, list(range(6)), list(range(1, 10)))
        As.append(A)
        bs.append(b)
        cs.append(c)
    np.savez('data/' + name, A=np.array(As), b=np.array(bs), c=np.array(cs))

def main(dataset, model_maker, fringe_maker):
    data = np.load('data/' + dataset)
    As, bs, cs = data['A'], data['b'], data['c']
    m, n = As[0].shape
    As, bs, cs = torch.FloatTensor(As), torch.FloatTensor(bs), torch.FloatTensor(cs)
    model = model_maker(n, m)
    trainer = ESTrainer(0.2, 0.025, 1.0, 1000, model, fringe_maker, max_updates=1200)
    trainer.train((As, bs, cs))


def eval_nonint_heuristic(dataset, fringe_maker):
    data = np.load('data/' + dataset)
    As, bs, cs = data['A'], data['b'], data['c']
    m, n = As[0].shape
    print("m, n =", m, n)
    num_expanded = []
    for A,b,c in zip(As, bs, cs):
        solver = BBSolver(A, b, c, fringe_maker, nonint_heuristic)
        sol, obj = solver.solve()
        print("Problems Expanded:", solver.num_problems_expanded)
        num_expanded.append(solver.num_problems_expanded)
    print(num_expanded)
    print(np.mean(num_expanded))


def eval_random_heuristic(dataset, fringe_maker, num_trials):
    data = np.load('data/' + dataset)
    As, bs, cs = data['A'], data['b'], data['c']
    m, n = As[0].shape
    print("m, n =", m, n)
    
    average_expanded = np.zeros(len(As))
    for _ in range(num_trials):
        print("Doing Trial", _)
        for i, (A,b,c) in enumerate(zip(As, bs, cs)):
            solver = BBSolver(A, b, c, fringe_maker, random_heuristic)
            sol, obj = solver.solve()
            average_expanded[i] += solver.num_problems_expanded
    print(list(average_expanded / num_trials))
    print(np.mean(average_expanded / num_trials))

def eval(dataset, model_path, model_maker, fringe_maker):
    data = np.load('data/' + dataset)
    As, bs, cs = data['A'], data['b'], data['c']
    m, n = As[0].shape
    print("m, n =", m, n)
    model = model_maker(n, m)
    model.load_state_dict(torch.load("models/" + model_path))
    heuristic = get_heuristic_from(model, train=False)
    num_expanded = []
    for A,b,c in zip(As, bs, cs):
        solver = BBSolver(A, b, c, fringe_maker, heuristic)
        sol, obj = solver.solve()
        print("Problems Expanded:", solver.num_problems_expanded)
        num_expanded.append(solver.num_problems_expanded)
    print(num_expanded)
    print(np.mean(num_expanded))

if __name__ == '__main__':
     main('train-maxcut.npz', BBModel, DFSFringe)
