import numpy as np
import matplotlib.pyplot as plt
import copy
from thermopack.multiparameter import multiparam
from euler_thermopack_singlePhase import get_roe_avg as get_roe_avg_tp
from euler_thermopack_singlePhase import f as f_tp
from euler_thermopack_singlePhase import get_conservative as get_conservative_tp
from euler_thermopack_singlePhase import get_primitive as get_primitive_tp
from euler import get_roe_avg
from euler import f
from idealGasEirik import IdealGas

Q1 = np.array([1.55810002e+01, 0.00000000e+00, -4.01983248e+0])
Q2 = np.array([1.51673743e+00,  0.00000000e+00, -3.25838965e+04])
N_cell = 102
z = [1.]
# test (alexandra shock tube project report)
T0 = 350  # K
p_l, p_r = 1e6, 0.1e6  # Pa
u_l, u_r = 0.5, 0.5  # m/s TODO første komponent blir riktig med 0.1, hvorfor?
p0 = np.concatenate(
    [p_l*np.ones(N_cell//2), p_r*np.ones(N_cell - N_cell//2)])
u0 = np.concatenate(
    [np.ones(N_cell//2) * u_l, np.ones(N_cell - N_cell//2)*u_r])
T0 = np.ones(N_cell) * T0

#GERGCO2 = multiparam("CO2", "GERG2008")
IGEIRIK = IdealGas(1.4)

L = 1
t = 0.0005
# rho: kg / m^3, rhou:  kg/m^2s, E: J/m^3
Q0 = get_conservative_tp(IGEIRIK , p0, u0, T0, z)
ptest, utest, Ttest, ctest = get_primitive_tp(IGEIRIK, Q0, z)
assert np.allclose(ptest, p0), f"ptest: {ptest}, p0: {p0}"
assert np.allclose(utest, u0), f"utest: {utest}, u0: {u0}"
assert np.allclose(Ttest, T0), f"Ttest: {Ttest}, T0: {T0}"


# # TEST
Q1 = Q0[:, 0]
Q2 = Q0[:, -1]

z = np.array([1.0])

delta = Q2 - Q1

GERGCO2 = multiparam("CO2", "GERG2008")

print("ideal gas")
u, H, c, alpha, R, lam = get_roe_avg(Q1, Q2)
A = R @ np.diag(lam) @ np.linalg.inv(R)
#print("H", H, "c", c, "u", u)

#print("Adelta", A @ delta, "Df", (f(Q1) - f(Q2)))

print("thermopack")
u_hat_tp, H_hat_tp, c_hat_tp, alpha_tp, R_tp, lam_tp = get_roe_avg_tp(
    IGEIRIK, z, Q1, Q2)
A = R_tp @ np.diag(lam_tp) @ np.linalg.inv(R_tp)
print("Adelta", A @ delta, "Df", (f_tp(Q1, p_l) - f_tp(Q2, p_r)))


