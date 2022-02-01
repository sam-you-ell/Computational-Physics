# Constructing Matrix Equation and boundary value terms. Will have to generalise this to any size matrices of (N-2)x(N-2) -> (why -2?)
# Class methods or just functions?
# create object for certain metal types
# Author - Samuel Hopkins
# Date - January/Febuary 2022

from scipy import linalg, sparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


class Metal:
    """
    Allows me to assign parameters for an arbitrary metal, and then calculate different quantities from them.
    """

    def __init__(self, Name, lamb, rho, C, L):
        self.Name = Name
        self.lamb = lamb  # thermal conductivity
        self.rho = rho  # density
        self.C = C  # Specific Heat
        self.L = L  # Length

    def thermaldiffusivity(self):
        return self.lamb / (self.rho * self.C)


Iron = Metal('Iron', lamb=59, rho=7900, C=450, L=0.5)


class Finite_Difference:

    def __init__(self, Name, N, d_0, d_N, tau, h):
        self.N = N
        self.Name = Name
        self.d_0 = d_0
        self.d_N = d_N
        self.tau = tau
        self.h = h  # spatial step


def MatrixMaker(N, Metal, tau, h,  identity):
    k = 1 if Metal == 'test' else Metal.thermaldiffusivity()
    # alpha = (k * FiniteDifference.tau) / (FiniteDifference.h ** 2)
    alpha = (k * tau) / (h**2)
    if identity is True:
        return sparse.diags([0, 1, 0], [-1, 0, 1], shape=((N-2, N-2)))

    elif identity is False:
        M = sparse.diags([-alpha, (1+2*alpha), -alpha],
                         [-1, 0, 1], shape=((N-2), (N-2)))
        return M.toarray()


def BoundaryCondition(N: int, bc: str, d_0: float, d_N: float) -> np.ndarray:
    alpha = 1
    if bc == 'dirichlet':
        d_N = 273.0  # kelvin
        d_0 = 1273.0
        B = np.zeros(N-2)
        B[0] = - alpha * d_0
        B[-1] = - alpha * d_N
        return B


print(MatrixMaker(7, 'test', 1, 1, False))


print(BoundaryCondition(7, 'dirichlet', d_0=1273, d_N=273))


def spatialtransport(maxt, N):
    L = 50
    t = np.linspace(0, maxt)
    x = np.linspace(0, L)
    tau = t[1] - t[0]
    h = x[1] - x[0]

    u_n = np.zeros(N-2)
