import numpy as np
import matplotlib.pyplot as plt
import copy
from thermopack.multiparameter import multiparam


def get_conservative(eos, p, u, T, z):
    """
    v is velocity
    """
    rho = np.zeros_like(p)
    rhou = np.zeros_like(p)
    E = np.zeros_like(p)
    mw = sum([z[i]*eos.compmoleweight(i+1) *
              1e-3 for i in range(eos.nc)])  # kg/mol
    for i in range(len(p)):
        flsh = eos.two_phase_tpflash(T[i], p[i], z)
        # Computing vapour and vapour phase specific volume
        if flsh.phase == eos.TWOPH:
            vg, = eos.specific_volume(T[i], p[i], flsh.y, eos.VAPPH)
            vl, = eos.specific_volume(T[i], p[i], flsh.x, eos.LIQPH)
            eg, = eos.internal_energy_tv(T[i], vg, flsh.y)
            el, = eos.internal_energy_tv(T[i], vl, flsh.x)
            e = flsh.betaL * el + flsh.betaV * eg
            v = flsh.betaL * vl + flsh.betaV * vg
        elif flsh.phase == eos.VAPPH:
            v, = eos.specific_volume(T[i], p[i], z, eos.VAPPH)
            e, = eos.internal_energy_tv(T[i], v, z)
        else:
            v, = eos.specific_volume(T[i], p[i], z, eos.LIQPH)
            e, = eos.internal_energy_tv(T[i], v, z)

        # convert to molar units and compute conserved variables
        # Both rho and e seem to be correct compared to NIST
        rho[i] = mw / v  # kg/m^3
        e = e / mw  # J/kg
        rhou[i] = rho[i] * u[i]  # kg/m^2s
        E[i] = rho[i]*(e + 0.5 * u[i]**2)  # J/m^3

    return np.array([rho, rhou, E])


def get_primitive(eos, Q, z):

    rho = Q[0]
    rhou = Q[1]
    E = Q[2]
    p = np.zeros_like(rho)
    u = np.zeros_like(rho)
    T = np.zeros_like(rho)
    c = np.zeros_like(rho)

    mw = sum([z[i]*eos.compmoleweight(i+1) *
              1e-3 for i in range(eos.nc)])  # kg/mol
    for i in range(len(rho)):
        v = mw / rho[i]  # m^3/mol
        e = (E[i] / rho[i] - 0.5 * rhou[i]**2 / rho[i]) * mw  # J/kg
        flsh = eos.two_phase_uvflash(z, e, v)
        T[i] = flsh.T
        p[i] = flsh.p
        u[i] = rhou[i] / rho[i]
        c[i] = eos.speed_of_sound(
            T[i], p[i], flsh.x, flsh.y, z, flsh.betaV, flsh.betaL, flsh.phase)  # m / s
    return p, u, T, c


def f(Q, p):
    """
    Compute flux
    p is included to save computational time
    """
    rho = Q[0]
    rhou = Q[1]
    E = Q[2]

    f0 = rhou
    f1 = rhou**2/rho + p
    f2 = (E + p) * rhou/rho
    return np.array([f0, f1, f2])


def euler_1d_lf(eos, N_cell, Q0, L, time, z):
    """
    Solve 1d euler with Lax-Friedrich method for polytropic gas
    # TODO
    # 0. sammenlikne med alexandra og finn liten bug
    # 1. Roe eller hllc X
    # 2. Thermopack?
        - thermodynamiske deriverte
    # 3. Finn bug i LF
    """
    fig, ax = plt.subplots()
    x = np.linspace(0, L, len(Q0[0]))

    Q = copy.deepcopy(Q0)
    dx = L / N_cell
    # Time stepping
    t = 0
    while t < time:
        p, u, T, c_arr = get_primitive(eos, Q, z)
        c = max(c_arr)
        print("c", c)
        print("max", max(abs(Q[1, 1:-1])))
        dt = min(0.8 * dx / (c + max(abs(Q[1, 1:-1]))), time-t)

        Q[:, 1:-1] = (Q[:, :-2] + Q[:, 2:]) / 2 - dt / \
            (2*dx) * (f(Q[:, 2:], p[2:]) - f(Q[:, :-2], p[:-2]))

        # boundary condition
        Q[:, 0] = Q[:, 1]
        Q[:, -1] = Q[:, -2]

        ax.clear()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("p [Pa]")
        ax.plot(x, p)
        plt.pause(0.001)
        t += dt
        print("t", t)
    return Q


if __name__ == "__main__":
    N_cell = 102
    z = [1.]
    # test (alexandra shock tube project report)
    T0 = 350  # K
    p_l, p_r = 1e6, 0.1e6  # Pa
    u_l, u_r = 0., 0.  # m/s
    p0 = np.concatenate(
        [p_l*np.ones(N_cell//2), p_r*np.ones(N_cell - N_cell//2)])
    u0 = np.concatenate(
        [np.ones(N_cell//2) * u_l, np.ones(N_cell - N_cell//2)*u_r])
    T0 = np.ones(N_cell) * T0

    GERGCO2 = multiparam("CO2", "GERG2008")
    L = 1
    t = 0.0005
    Q0 = get_conservative(GERGCO2, p0, u0, T0, z)
    Qlf = euler_1d_lf(GERGCO2, N_cell, Q0, L, t, z)
    p_lf, u_lf, T_lf, c_2 = get_primitive(GERGCO2, Qlf, z)
    print(p_lf, u_lf, T_lf, c_2)

    # Plotting
    x = np.linspace(0, L, len(Q0[0]))

    fig, ax = plt.subplots()
    ax.plot(x, Q0[0, :], label="initial")
    # ax.plot(x, Qroe[0, :], label="roe solver")
    ax.plot(x, Qlf[0, :], label="lax-friedrichs")
    ax.set_xlabel("x")
    ax.set_ylabel("rho")
    ax.legend()
    plt.savefig("plots/density.pdf")

    fig, ax = plt.subplots()
    ax.plot(x, Q0[1, :] / Q0[0, :], label="initial")
    # ax.plot(x, Qroe[1, :] / Qroe[0, :], label="roe solver")
    ax.plot(x, Qlf[1, :] / Qlf[0, :], label="lax-friedrichs")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend()
    plt.savefig("plots/velocity.pdf")

    # OBS får ikke e = (E/ rho - 0.5 u^2) til å stemme
    fig, ax = plt.subplots()
    ax.plot(x, Q0[2, :], label="initial")
    # ax.plot(x, Qroe[2, :], label="roe solver")
    ax.plot(x, Qlf[2, :], label="lax-friedrichs")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("E [J]")
    ax.legend()
    plt.savefig("plots/energy.pdf")

    fig, ax = plt.subplots()
    ax.plot(x, p0, label="initial")
    # ax.plot(x, p_roe, label="roe solver")
    ax.plot(x, p_lf, label="lax-friedrichs")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.legend()
    plt.savefig("plots/pressure.pdf")

    plt.show()