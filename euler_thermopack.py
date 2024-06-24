import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import root_scalar
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


def get_primitive_single(eos, Q, z):
    rho = Q[0]
    rhou = Q[1]
    E = Q[2]
    mw = sum([z[i]*eos.compmoleweight(i+1) *
              1e-3 for i in range(eos.nc)])  # kg/mol
    v = mw / rho  # m^3/mol
    e = (E / rho - 0.5 * rhou**2 / rho) * mw  # J/kg
    flsh = eos.two_phase_uvflash(z, e, v)
    T = flsh.T
    p = flsh.p
    u = rhou / rho
    c = eos.speed_of_sound(
        T, p, flsh.x, flsh.y, z, flsh.betaV, flsh.betaL, flsh.phase)  # m / s
    return p, u, T, c


def get_primitive(eos, Q, z):
    rho = Q[0]
    p = np.zeros_like(rho)
    u = np.zeros_like(rho)
    T = np.zeros_like(rho)
    c = np.zeros_like(rho)

    for i in range(len(rho)):
        p[i], u[i], T[i], c[i] = get_primitive_single(eos, Q[:, i], z)

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
    # 4. Åsmund, Morten Svend tollak paper
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

        # ax.clear()
        # ax.set_xlabel("x [m]")
        # ax.set_ylabel("p [Pa]")
        # ax.plot(x, p)
        # plt.pause(0.001)
        t += dt
        print("t", t)
    return Q


def calculate_Gamma(eos, z, v, T):
    """
    Calculate Gruneisen coeficient for single phase

    """
    E, dudt = eos.internal_energy_tv(T, v, z, dedt=True)
    p, dpdt = eos.pressure_tv(T, v, z, dpdt=True)
    Gamma = v * dpdt/dudt
    return Gamma


def get_roe_avg(eos, z, Q1, Q2):
    """
    Roe average of left statte Q1 and right state Q2

    OBS: this method only works for single phase
    """
    mw = sum([z[i]*eos.compmoleweight(i+1) *
              1e-3 for i in range(eos.nc)])  # kg/mol

    # left state
    p1, u1, T1, c1 = get_primitive_single(eos, Q1, z)
    rho1 = Q1[0]
    E1 = Q1[2]
    H1 = (E1 + p1) / rho1

    # right state
    p2, u2, T2, c2 = get_primitive_single(eos, Q2, z)
    rho2 = Q2[0]
    E2 = Q2[2]
    H2 = (E2 + p2) / rho2

    # Get Roe averaged quatities
    u_hat = (np.sqrt(rho1) * u1 + np.sqrt(rho2) * u2) / \
        (np.sqrt(rho1) + np.sqrt(rho2))
    H_hat = (H1 * np.sqrt(rho1) + H2 *
             np.sqrt(rho2)) / (np.sqrt(rho1) + np.sqrt(rho2))
    # fra alexandra prosjektoppgave appendix A TODO check
    rho_hat = np.sqrt(rho1 * rho2)

    # Find temperature to compute speed of sound
    h_hat = (H_hat - 0.5 * u_hat**2) * mw  # J / mol
    v_hat = 1/rho_hat * mw  # m^3/mol
    def dh(T): return h_hat - eos.enthalpy_tv(T, v_hat, z)[0]
    solution = root_scalar(dh, x0=T1)
    if not solution.flag == "converged":
        print("could not find temperature root")
        exit()
    else:
        T_hat = solution.root

    volume = 1  # m3
    n = [volume / v_hat]  # mol
    c_hat = eos.speed_of_sound_tv(T_hat, volume, n)
    Gamma = calculate_Gamma(eos, z, v_hat, T_hat)

    R = np.array([
        [1, 1, 1],
        [u_hat-c_hat, u_hat, u_hat+c_hat],
        [H_hat-u_hat*c_hat, H_hat - c_hat**2 / Gamma, H_hat+u_hat*c_hat]]
    )

    # Find wave speeds
    delta = Q2 - Q1
    alpha = np.linalg.inv(R) @ delta

    # create vector of eigenvalues
    lam = np.array([u_hat-c_hat, u_hat, u_hat+c_hat])


    # TEST RH condition
    A = R @ np.diag(lam) @ np.linalg.inv(R)
    delta = Q2 - Q1
    if not np.allclose(A @ delta, f(Q2, p2) - f(Q1, p1), atol=1e-3, rtol=1e-2):
        # print("RH condition: ", "A delta ", A @
        #       delta, "(f(Q[:,i]) - f(Q[:,i-1]))", (f(Q[:, i], p2) - f(Q[:, i-1], p1)), "for cell ", i)
        print("RH cond not satisfied for delta = ", delta)
        print("Rel error", np.abs(A @ delta - (f(Q2, p2) - f(Q1, p1))) / np.abs(A @ delta))
        print("abs error", np.abs(A @ delta - (f(Q2, p2) - f(Q1, p1))))
        print("*********")
    

    return u_hat, H_hat, c_hat, alpha, R, lam


def euler_1d_roe(eos, z, N_cell, Q0, L, T):
    """
    Solve 1d euler with Roe method for general eos
    """

    fig, ax = plt.subplots()
    x = np.linspace(0, L, len(Q0[0]))

    Q = copy.deepcopy(Q0)
    dx = L / N_cell
    # Time stepping
    t = 0
    ApDQ1 = np.zeros_like(Q)
    AmDQ2 = np.zeros_like(Q)
    while t < T:
        # TEST
        c = 0
        for i in range(1, N_cell-1):
            # get roe average
            u1, h1, c1, alpha1, R1, lam1 = get_roe_avg(
                eos, z, Q[:, i-1], Q[:, i])
            u2, h2, c2, alpha2, R2, lam2 = get_roe_avg(
                eos, z, Q[:, i], Q[:, i+1])

            # find max c
            if max(c1, c2) > c:
                c = max(c1, c2)

            lam1p = 0.5 * (lam1 + np.abs(lam1))
            lam2m = 0.5 * (lam2 - np.abs(lam2))

            # calculate A^{\mp} Delta Q_{i \pm 1/2}
            ApDQ1[:, i] = R1 @ np.diag(lam1p) @ alpha1
            AmDQ2[:, i] = R2 @ np.diag(lam2m) @ alpha2

        p_test, u_test, T_test, c_test = get_primitive(eos, Q, z)
        ax.clear()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("p [Pa]")
        ax.plot(x, p_test)
        plt.pause(0.001)

        dt = min(0.9 * dx / (c + max(abs(Q[1, 1:-1]))), T-t)
        if np.any(np.isnan(- dt / dx * (AmDQ2 + ApDQ1))):
            print("NAN")
            return Q
        else:
            Q = Q - dt / dx * (AmDQ2 + ApDQ1)

        # boundary condition
        Q[:, 0] = Q[:, 1]
        Q[:, -1] = Q[:, -2]

        t += dt
        print(t)

    return Q


if __name__ == "__main__":
    N_cell = 32
    z = [1.]
    # test (alexandra shock tube project report)
    T0 = 500  # K
    p_l, p_r = 1e6, 0.1e6  # Pa
    u_l, u_r = 0., 0.  # m/s TODO første komponent blir riktig med 0.1, hvorfor?
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
    Qroe = euler_1d_roe(GERGCO2, z, N_cell, Q0, L, t)
    p_lf, u_lf, T_lf, c_2 = get_primitive(GERGCO2, Qlf, z)
    p_roe, u_roe, T_roe, c_roe = get_primitive(GERGCO2, Qroe, z)
    print(p_lf, u_lf, T_lf, c_2)

    # Plotting
    x = np.linspace(0, L, len(Q0[0]))

    fig, ax = plt.subplots()
    ax.plot(x, Q0[0, :], label="initial")
    ax.plot(x, Qroe[0, :], label="roe solver")
    ax.plot(x, Qlf[0, :], label="lax-friedrichs")
    ax.set_xlabel("x")
    ax.set_ylabel("rho")
    ax.legend()
    plt.savefig("plots/density.pdf")

    fig, ax = plt.subplots()
    ax.plot(x, Q0[1, :] / Q0[0, :], label="initial")
    ax.plot(x, Qroe[1, :] / Qroe[0, :], label="roe solver")
    ax.plot(x, Qlf[1, :] / Qlf[0, :], label="lax-friedrichs")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend()
    plt.savefig("plots/velocity.pdf")

    # OBS får ikke e = (E/ rho - 0.5 u^2) til å stemme
    fig, ax = plt.subplots()
    ax.plot(x, Q0[2, :], label="initial")
    ax.plot(x, Qroe[2, :], label="roe solver")
    ax.plot(x, Qlf[2, :], label="lax-friedrichs")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("E [J]")
    ax.legend()
    plt.savefig("plots/energy.pdf")

    fig, ax = plt.subplots()
    ax.plot(x, p0, label="initial")
    ax.plot(x, p_roe, label="roe solver")
    ax.plot(x, p_lf, label="lax-friedrichs")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.legend()
    plt.savefig("plots/pressure.pdf")

    plt.show()
