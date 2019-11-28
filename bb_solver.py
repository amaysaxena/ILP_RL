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

Solution = namedtuple('Solution', ['solution', 'objective_value', 'is_integer'])

def is_integer(a, tol=1e-4):
	return min(abs(a - np.floor(a)), abs(a - np.ceil(a))) < tol

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
		check_integer = all(is_integer(x) for x in self.x.value)
		return Solution(self.x.value, res, check_integer)

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

	def solve(self):
		while not self.fringe.isempty():
			problem = self.fringe.pop()
			sol = problem.solve()
			if sol:
				x, value, is_int = sol
				if value > self.best_objective - 1e-3:
					continue # Even relaxed solution is terrible. Abandon node.
				elif is_int:
					# New best integral solution found.
					self.best_solution = x
					self.best_objective = value
				else:
					# Gotta branch
					index_to_branch = self.heuristic(problem.A, problem.b, problem.c, x)
					print("Index to branch:", index_to_branch)
					for prob in problem.branch_on(index_to_branch, x[index_to_branch]):
						self.fringe.push(prob)
		return self.best_solution, self.best_objective

def random_heuristic(A, b, c, x):
	nonint = [i for i, v in enumerate(x) if not is_integer(v)]
	return random.choice(nonint)

def main():
	c = np.array([-100, -150])
	b = np.array([40000, 200])
	A = np.array([[8000, 4000],[15, 30]])

	solver = BBSolver(A, b, c, DFSFringe, random_heuristic)
	sol, obj = solver.solve()
	print("Solution:", sol)
	print("Objective:", obj)

if __name__ == '__main__':
	main()


