import numpy as np


class flashObj:
    def __init__(self, phase, P, T):
        self.phase = phase
        self.p = P
        self.T = T
        self.betaV = 1.
        self.betaL = 0.
        self.x = np.array([1.])
        self.y = np.array([1.])


class IdealGas:
    def __init__(self, gamma):
        self.gamma = gamma
        self.nc = 1
        self.VAPPH = 1
        self.R = 8.314  # J/(mol K)
        self.cv = self.R / (self.gamma - 1)

    def compmoleweight(self, dummy):
        return 44e-3  # kg/mol

    def two_phase_tpflash(self, T, P, z):
        return flashObj(self.VAPPH, P, T)

    def two_phase_uvflash(self, z, e, v):
        T = e / self.cv
        P = self.R * T / v
        return flashObj(self.VAPPH, P, T)

    def specific_volume(self, T, P, z, phase):
        return self.R * T / P

    def internal_energy_tv(self, T, v, z):
        return self.cv * T

    def speed_of_sound(self, T, p, x, y, z, betaV, betaL, phase):
        return np.sqrt(self.gamma * self.R * T / self.compmoleweight(1))