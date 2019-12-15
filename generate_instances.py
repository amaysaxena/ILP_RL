import mip
import numpy as np
from graph import random_connected_graph

def mip_from_std_form(A, b, c):
    # initialize
    n = len(c)
    m = len(b)
    ilp = mip.Model()

    # variables
    x = [ilp.add_var(name='x({})'.format(i), var_type=mip.INTEGER) for i in range(n)]

    # constraints
    for j in range(m):
        a, bi = A[j, :], b[j]
        cons = mip.xsum(a[i] * x[i] for i in range(n) if a[i] != 0) <= 0
        cons.add_const(-bi)
        ilp.add_constr(cons)

    # objective
    ilp.objective = mip.minimize(mip.xsum(c[i] * x[i] for i in range(n)))
    return ilp

def random_maxcut_instance(n_vertices, n_edges, weight_population=None):
    """Returns a random instance of maxcut formulated as an ILP
    in standard form:

    min c^T x

    subject to:
        Ax <= b
        0 <= x
        x \in Z
    """
    A = np.zeros((3 * n_edges + n_vertices, n_vertices + n_edges), dtype=np.float32)
    c = np.zeros(n_vertices + n_edges, dtype=np.float32)
    b = np.zeros(3 * n_edges + n_vertices, dtype=np.float32)
    graph = random_connected_graph(n_vertices, n_edges, weight_population or [1])

    for i, (u, v) in enumerate(graph.edges):
        c[n_vertices + i] = -graph.weight(u, v)
        
        A[i, n_vertices + i] = 1
        A[i, u] = -1
        A[i, v] = -1
        b[i] = 0

        A[n_edges + i, n_vertices + i] = 1
        A[n_edges + i, u] = 1
        A[n_edges + i, v] = 1
        b[n_edges + i] = 2

        A[2 * n_edges + i, n_vertices + i] = 1
        b[2 * n_edges + i] = 1

    for u in range(n_vertices):
        A[3 * n_edges + u, u] = 1
        b[3 * n_edges + u] = 1

    return A, b, c

def random_packing_instance(n_items, n_res_contr, A_coeff_population, c_population):
    """Returns a random instance of the packing problem formulated as an ILP
    in standard form:

    min c^T x

    subject to:
        Ax <= b
        0 <= x
        x \in Z
    """
    A = -np.random.choice(A_coeff_population, size=(n_res_contr, n_items)).astype(np.int32)
    c = np.random.choice(c_population, size=n_items).astype(np.int32)
    b = -np.random.choice(np.arange(9 * n_items, 10 * n_items), size=n_res_contr).astype(np.int32)
    return A, b, c

def random_knapsack_instance(n_items, weight_population, value_population, capacity_population):
    """Returns a random instance of the knapsack problem (without repetition) formulated 
    as an ILP in standard form.
    """
    w = np.random.choice(weight_population, size=n_items).astype(np.int32)
    cap = np.random.choice(capacity_population)
    c = -np.random.choice(value_population, size=n_items).astype(np.int32)
    b = np.array([cap] + [1 for _ in range(n_items)]).astype(np.int32)
    A = np.vstack((np.array([w]), np.eye(n_items))).astype(np.int32)
    return A, b, c

def generate_instances(n_instances, instance_class, args):
    assert instance_class in ['maxcut', 'packing', 'knapsack']
    if instance_class == 'maxcut':
        assert len(args) == 3
        generator = random_maxcut_instance
    elif instance_class == 'packing':
        assert len(args) == 4
        generator = random_packing_instance
    else:
        assert len(args) == 4
        generator = random_knapsack_instance
    As = []
    bs = []
    cs = []
    for _ in range(n_instances):
        A, b, c = generator(*args)
        As.append(A)
        bs.append(b)
        cs.append(c)
    return As, bs, cs

def solve_instance(A, b, c):
    """
        Uses MIP to optimally solve instance given in standard form.
    """

    ilp = mip_from_std_form(A, b, c)
    status = ilp.optimize()
    if status == mip.OptimizationStatus.OPTIMAL:
        print('optimal solution cost {} found'.format(ilp.objective_value))
    elif status == mip.OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(ilp.objective_value, ilp.objective_bound))
    elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(ilp.objective_bound))
    if status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE:
        print('solution:')
        for v in ilp.vars:
            if abs(v.x) > 1e-6: # only printing non-zeros
                print('{} : {}'.format(v.name, v.x))

def save_new_test_set():
    A1, b1, c1 = generate_instances(15, 'maxcut', [15, 30, list(range(1, 20))])
    A2, b2, c2 = generate_instances(15, 'knapsack', [15, np.arange(20), np.arange(20), np.arange(50, 150)])
    A3, b3, c3 = generate_instances(15, 'packing', [200, 20, np.arange(6), np.arange(1, 11)])
    A = A1 + A2 + A3
    b = b1 + b2 + b3
    c = c1 + c2 + c3

if __name__ == '__main__':
    A1, b1, c1 = generate_instances(20, 'knapsack', [200, np.arange(20), np.arange(20), np.arange(200, 1000)])
    for A, b, c in zip(A1, b1, c1):
        solve_instance(A, b, c)

