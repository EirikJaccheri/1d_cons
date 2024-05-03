import numpy as np
import matplotlib.pyplot as plt
import copy
# Constants
gamma = 1.4
R = 8.314
c_v = 0.718e3  # J/kgK


def f(Q):
    """
    Compute flux
    """
    rho = Q[0]
    rhou = Q[1]
    e = Q[2]
    p = (gamma - 1) * rho * e

    f0 = rhou
    f1 = rhou**2/rho + p
    f2 = (e + p) * rhou/rho
    return np.array([f0, f1, f2])


def euler_1d_lf(N_cell, Q0, L, T):
    """
    Solve 1d euler with Lax-Friedrich method for polytropic gas
    # TODO
    # 0. sammenlikne med alexandra og finn liten bug
    # 1. Roe eller hllc
    # 2. Thermopack?
        - thermodynamiske deriverte
    """

    Q = copy.deepcopy(Q0)
    dx = L / N_cell
    # Time stepping
    t = 0
    while t < T:
        # time step
        p = (gamma - 1) * Q[0, 1:-1] * Q[2, 1:-1]
        c = max(np.sqrt((gamma - 1) * (Q[2, 1:-1] + p / Q[0, 1:-1])))
        dt = min(0.9 * dx / (c + max(abs(Q[1, 1:-1]))), T-t)

        Q[:, 1:-1] = (Q[:, :-2] + Q[:, 2:]) / 2 - dt / \
            (2*dx) * (f(Q[:, 2:]) - f(Q[:, :-2]))
        # boundary condition
        Q[:, 0] = Q[:, 1]
        Q[:, -1] = Q[:, -2]

        t += dt
    return Q


def get_roe_avg(Q1, Q2):
    """
    Roe average of left statte Q1 and right state Q2
    """
    # unpack values
    # left state
    rho1 = Q1[0]
    rhou1 = Q1[1]
    u1 = rhou1 / rho1

    e1 = Q1[2]
    p1 = (gamma - 1) * rho1 * e1
    # right state
    rho2 = Q2[0]
    rhou2 = Q2[1]
    u2 = rhou2 / rho2
    e2 = Q2[2]
    p2 = (gamma - 1) * rho2 * e2

    # Get Roe averaged quatities TODO gjør utregningene selv
    u = (np.sqrt(rho1) * u1 + np.sqrt(rho2) * u2) / \
        (np.sqrt(rho1) + np.sqrt(rho2))
    h = ((e1 + p1) / np.sqrt(rho1) + (e2 + p2) /
         np.sqrt(rho2)) / (np.sqrt(rho1) + np.sqrt(rho2))
    c = np.sqrt((gamma - 1) * (h - 0.5 * u**2))

    # Find wave speeds ??? TODO regne over dette (foreløpig bare kopi av )
    delta = Q2 - Q1
    alpha2 = (gamma - 1) * ((h - u**2) *
                            delta[0] + u * delta[1] - delta[2]) / c**2
    alpha3 = (delta[1] + (c - u) * delta[0] - c * alpha2) / (2 * c)
    alpha1 = delta[0] - alpha2 - alpha3
    alpha = np.array([alpha1, alpha2, alpha3])

    # create matrix of eigenvectors
    # dette er en kopi Leveque s301
    R = np.array([
        [1, 1, 1],
        [u-c, u, u+c],
        [h-u*c, 0.5*u**2, h+u*c]]
    )

    # create vector of eigenvalues
    lam = np.array([u-c, u, u+c])

    return u, h, c, alpha, R, lam


def euler_1d_roe(N_cell, Q0, L, T):
    """
    Solve 1d euler with Roe method for polytropic gas

    TODO:
    1. Plott trykk, temperatur, hastighet, tetthet
    2. Gå fra initsialverdier til
    3. Bare konstante verdier og se at ingen ting endrer seg
    3.5 Bare konstant trykk 
    4. Hvorfor kommer det masse bakover?
    5. Regn ut roe average for eulerlikningene selv
    6. Teste A(Qi - Qi-1) = f(Qi) - f(Qi-1)
    7. Matematika eller Maple
    8. For å bruke thermopack må h byttes ut med e eller s
    9. Løse matriseproblem for å få egenevektor basis numerisk
    10. start 14.8 og sjekke at jeg får det samme når jeg legger inn ideel gas
        - Enklest med disse variablene: rho, u, e
        - E = rho (e + 1/2 u^2)
        - evt bytte etterpå
    """
    Q = copy.deepcopy(Q0)
    dx = L / N_cell
    # Time stepping
    t = 0
    ApDQ1 = np.zeros_like(Q)
    AmDQ2 = np.zeros_like(Q)
    print(Q)
    while t < T:
        # time step
        print("*******************************'")
        p = (gamma - 1) * Q[0, 1:-1] * Q[2, 1:-1]
        c = max(np.sqrt((gamma - 1) * (Q[2, 1:-1] + p / Q[0, 1:-1])))
        dt = min(0.1 * dx / (c + max(abs(Q[1, 1:-1]))), T-t)

        for i in range(1, N_cell-1):
            # get roe average
            u1, h1, c1, alpha1, R1, lam1 = get_roe_avg(Q[:, i-1], Q[:, i])
            u2, h2, c2, alpha2, R2, lam2 = get_roe_avg(Q[:, i], Q[:, i+1])

            lam1m = 0.5 * (lam1 - np.abs(lam1))
            lam2p = 0.5 * (lam2 + np.abs(lam2))

            # calculate A^{\mp} Delta Q_{i \pm 1/2}
            ApDQ1[:, i] = R2 @ np.diag(lam2p) @ alpha2
            AmDQ2[:, i] = R1 @ np.diag(lam1m) @ alpha1

        # update Q

        if np.any(np.isnan(- dt / dx * (AmDQ2 + ApDQ1))):
            return Q
            print("NAN")
            exit()
        else:
            Q = Q - dt / dx * (AmDQ2 + ApDQ1)
            print(Q)

        # boundary condition
        Q[:, 0] = Q[:, 1]
        Q[:, -1] = Q[:, -2]

        t += dt

    return Q


if __name__ == "__main__":
    # TODO
    # Implementere 1d euler likning løser (etter leveque):
    #    - skriv om euler likning til matriselikning
    #      og få egenverdier med ideel gas
    #    - Rieman problem
    #    - ideel gas
    #    - masse vs molar form
    # e = C_V T
    # PV = nRT or P = \rho R T
    # e, rho -> T, P, ...
    N_cell = 102
    # sod shock tube
    rho_l, u_l, p_l = 1, 0, 1
    rho_r, u_r, p_r = 0.125, 0, 0.1

    # initial conditions with ghost cells
    rho0 = np.concatenate(
        [np.ones(N_cell//2) * rho_l, np.ones(N_cell - N_cell//2) * rho_r])
    u0 = np.concatenate(
        [np.ones(N_cell//2) * u_l, np.ones(N_cell - N_cell//2) * u_r])
    p0 = np.concatenate(
        [np.ones(N_cell//2) * p_l, np.ones(N_cell - N_cell//2) * p_r])
    e0 = p0 / (gamma - 1) / rho0
    Q0 = np.array([rho0, rho0*u0, e0])
    L = 1
    T = 0.25
    # Q = euler_1d_lf(N_cell, Q0, L, T)
    Qroe = euler_1d_roe(N_cell, Q0, L, T)
    # plot
    x = np.linspace(0, L, len(p0))

    fig, ax = plt.subplots()
    ax.plot(x, Q0[0, :], label="initial")
    ax.plot(x, Qroe[0, :], label="roe solver")
    ax.set_xlabel("x")
    ax.set_ylabel("rho")
    ax.legend()
    plt.savefig("plots/density.pdf")

    fig, ax = plt.subplots()
    ax.plot(x, Q0[1, :] / Q0[0, :], label="initial")
    ax.plot(x, Qroe[1, :] / Qroe[0, :], label="roe solver")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend()
    plt.savefig("plots/velocity.pdf")

    fig, ax = plt.subplots()
    ax.plot(x, Q0[2, :], label="initial")
    ax.plot(x, Qroe[2, :], label="roe solver")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("e [J]")
    ax.legend()
    plt.savefig("plots/energy.pdf")
    plt.show()
