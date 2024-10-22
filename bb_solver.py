from generate_instances import random_maxcut_instance, random_packing_instance, random_knapsack_instance

import numpy as np
from collections import deque, namedtuple
import cvxpy as cp
import random

class DFSFringe(object):
    def __init__(self):
        self.fringe = []

    def push(self, item):
        self.fringe.append(item)

    def pop(self):
        return self.fringe.pop()

    def isempty(self):
        return len(self.fringe) == 0

class BFSFringe(object):
    def __init__(self):
        self.fringe = deque([])

    def push(self, item):
        self.fringe.append(item)

    def pop(self):
        return self.fringe.popleft()

    def isempty(self):
        return len(self.fringe) == 0

Solution = namedtuple('Solution', ['solution', 'objective_value', 'is_integer'])

def is_integer(a, tol=1e-3):
    return np.all(np.abs(a - np.rint(a)) < tol)

class LPProblem(object):
    def __init__(self, A, b, c):
        self.A, self.b, self.c = A, b, c

    def build_problem(self):
        m, n = self.A.shape
        self.x = cp.Variable(shape=(n,), nonneg=True)
        self.objective = cp.Minimize(self.c * self.x)
        self.constraints = [self.A * self.x <= self.b]
        self.problem = cp.Problem(self.objective, self.constraints)

    def solve(self, verbose=False):
        res = self.problem.solve(verbose=verbose, solver=cp.SCS)
        if self.problem.status in ['infeasible', 'unbounded']:
            return None
        return Solution(self.x.value, res, is_integer(self.x.value))

    def branch_on(self, index, value):
        new_prob1 = LPProblem(np.vstack((self.A, np.eye(self.A.shape[1])[index])),
                              np.hstack((self.b, np.floor(value))), self.c)
        new_prob1.x = self.x
        new_prob1.objective = self.objective
        new_prob1.constraints = self.constraints + [self.x[index] <= np.floor(value)]
        new_prob1.problem = cp.Problem(new_prob1.objective, new_prob1.constraints)

        new_prob2 = LPProblem(np.vstack((self.A, -np.eye(self.A.shape[1])[index])),
                              np.hstack((self.b, -np.ceil(value))), self.c)
        new_prob2.x = self.x
        new_prob2.objective = self.objective
        new_prob2.constraints = self.constraints + [self.x[index] >= np.ceil(value)]
        new_prob2.problem = cp.Problem(new_prob1.objective, new_prob2.constraints)

        return new_prob1, new_prob2

class BBSolver(object):
    """
    A pure branch-and-bound solver that allows the specification of
    a custom branching heuristic and fringe processing behavior.

    heuristic should be a function that takes in the problem in standard form
    along with the current solution, and returns the index of the variable
    to branch on.

    signature: heuristic(A, b, c, x)
    returns: index in range(len(x))
    """

    def __init__(self, A, b, c, fringe_maker, heuristic):
        self.fringe = fringe_maker()
        self.problem = LPProblem(A, b, c)
        self.problem.build_problem()
        self.fringe.push(self.problem)
        self.heuristic = heuristic

        self.best_objective = float('inf')
        self.best_solution = None

        self.num_problems_expanded = 0

    def step(self):
        problem = self.fringe.pop()
        sol = problem.solve()
        self.num_problems_expanded += 1
        # print("Problems Expanded:", self.num_problems_expanded)
        branched = False
        if sol:
            x, value, is_int = sol
            if value > self.best_objective - 1e-4:
                pass # Even relaxed solution is terrible. Abandon node.
            elif is_int:
                # New best integral solution found.
                self.best_solution = x
                self.best_objective = value
            else:
                # Gotta branch
                index_to_branch = self.heuristic(problem.A, problem.b, problem.c, x)
                for prob in problem.branch_on(index_to_branch, x[index_to_branch]):
                    self.fringe.push(prob)
                branched = True
            return problem, x, value, is_int, branched
        return None, None, None, None, branched

    def solve(self):
        while not self.fringe.isempty():
            self.step()
        return self.best_solution, self.best_objective

def random_heuristic(A, b, c, x):
    nonint = [i for i, v in enumerate(x) if not is_integer(v)]
    return random.choice(nonint)

def nonint_heuristic(A, b, c, x):
    nonint = [i for i, v in enumerate(x) if not is_integer(v)]
    dev = [abs(x[i] - np.rint(x[i])) for i in nonint]
    return nonint[np.argmax(dev)]

def main():
    data = np.load('data/train-maxcut.npz')
    As, bs, cs = data['A'], data['b'], data['c']
    # A, b, c = random_maxcut_instance(30, 50, list(9*np.random.uniform(size=100)))
    for A,b,c in zip(As, bs, cs):
        print("m, n =", A.shape)
        solver = BBSolver(A, b, c, DFSFringe, random_heuristic)
        sol, obj = solver.solve()
        print("Solution:", np.rint(sol))
        print("Objective:", obj)
        print("Problems Expanded:", solver.num_problems_expanded)

if __name__ == '__main__':
    # main()
    A, b, c = random_knapsack_instance(20, 10*np.random.uniform(size=100), 10*np.random.uniform(size=100), 20 + 40*np.random.uniform(size=100))
    print("m, n =", A.shape)
    solver = BBSolver(A, b, c, DFSFringe, nonint_heuristic)
    sol, obj = solver.solve()
    print("Solution:", np.rint(sol))
    print("Objective:", obj)
    print("Problems Expanded:", solver.num_problems_expanded)
