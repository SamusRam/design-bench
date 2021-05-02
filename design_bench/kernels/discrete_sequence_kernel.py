from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import GenericKernelMixin
import numpy as np


class DefaultSequenceKernel(GenericKernelMixin, Kernel):

    def __init__(self, size, diagonal=1.0, off_diagonal=0.1):
        self.kernel_matrix = np.full((size, size), off_diagonal)
        np.fill_diagonal(self.kernel_matrix, diagonal)

    def evaluate_kernel(self, x, y):
        return self.kernel_matrix[x][:, y].sum()

    def __call__(self, X, Y=None, eval_gradient=False):
        return np.array([[self.evaluate_kernel(
            x, y) for y in (X if Y is None else Y)] for x in X])

    def diag(self, X):
        return np.array([self.evaluate_kernel(x, x) for x in X])

    def is_stationary(self):
        return False  # the kernel is fixed in advance
