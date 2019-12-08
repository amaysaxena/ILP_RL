from __future__ import absolute_import, division, print_function, unicode_literals
import time
import warnings  
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.layers import Dense, Flatten, Conv2D
    from tensorflow.keras import Model, Sequential

class BBModel(Model):
    def __init__(self, num_vars, num_constraints):
        super(BBModel, self).__init__()
        self.n = num_vars
        self.m = num_constraints
        self.constraint_encoder = Sequential([
            Dense(64, activation='relu'),
            Dense(64, activation='tanh')])

    def call(self, A, b, c, x_val, branch_candidates):
        """
        Arguments:
            (A, b, c): ILP Problem in standard form.
            x_val: solution to LP relaxation of (A, b, c).
            branch_candidates: indices for variables to choose from to
                branch on.
        """
        encoded_constraints = self.constraint_encoder(np.hstack((A, b[:, None])))
        constraint_lt = np.hstack((np.eye(self.n)[branch_candidates],
                                   np.floor(x_val[branch_candidates])[:, None])).astype(np.float32)
        constraint_gt = np.hstack((-np.eye(self.n)[branch_candidates],
                                   -np.ceil(x_val[branch_candidates])[:, None])).astype(np.float32)
        encoded_lt = self.constraint_encoder(constraint_lt)
        encoded_gt = self.constraint_encoder(constraint_gt)
        encoded_candidates = tf.maximum(encoded_lt, encoded_gt)

        dot_prods = tf.matmul(encoded_candidates, encoded_constraints, transpose_b=True)
        candidate_features = (1 / self.m) * tf.reduce_sum(dot_prods, axis=1)
        return tf.nn.softmax(candidate_features)


if __name__ == '__main__':
    A = np.ones((10,15), dtype=np.float32)
    b = np.arange(10, dtype=np.float32)
    c = np.arange(15, dtype=np.float32)
    x = np.ones(15, dtype=np.float32)
    branch_candidates = np.arange(5, dtype=np.int32)

    model = BBModel(15, 10)
    t0 = time.time()
    out = model(A, b, c, x, branch_candidates)
    print(time.time() - t0)
    tf.print(out)
