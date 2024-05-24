import numpy as np
import matplotlib.pyplot as plt
import copy
# Constants
gamma = 1.4
R = 8.314
c_v = 0.718e3  # J/kgK


def get_conservative(rho, u, p):
    E = p / (gamma - 1) + 0.5 * rho * u**2
    return np.array([rho, rho*u, E])


def get_primitive(Q):
    rho = Q[0]
    u = Q[1] / Q[0]
    E = Q[2]
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    return rho, u, p


def f(Q):
    """
    Compute flux
    """
    rho = Q[0]
    rhou = Q[1]
    E = Q[2]
    p = (gamma - 1) * (E - 0.5 * rhou**2 / rho)

    f0 = rhou
    f1 = rhou**2/rho + p
    f2 = (E + p) * rhou/rho
    return np.array([f0, f1, f2])


def euler_1d_lf(N_cell, Q0, L, T):
    """
    Solve 1d euler with Lax-Friedrich method for polytropic gas
    # TODO
    # 0. sammenlikne med alexandra og finn liten bug
    # 1. Roe eller hllc X
    # 2. Thermopack?
        - thermodynamiske deriverte
    # 3. Finn bug i LF
    """

    Q = copy.deepcopy(Q0)
    dx = L / N_cell
    # Time stepping
    t = 0
    while t < T:
        rho, u, p = get_primitive(Q)
        c = max(np.sqrt(gamma * p / rho))
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
    # left state
    rho1, u1, p1 = get_primitive(Q1)
    E1 = p1 / (gamma - 1) + 0.5 * rho1 * u1**2

    # right state
    rho2, u2, p2 = get_primitive(Q2)
    E2 = p2 / (gamma - 1) + 0.5 * rho2 * u2**2

    # Get Roe averaged quatities
    u = (np.sqrt(rho1) * u1 + np.sqrt(rho2) * u2) / \
        (np.sqrt(rho1) + np.sqrt(rho2))

    # TODO check this!
    H1 = (E1 + p1) / rho1
    H2 = (E2 + p2) / rho2
    H = (H1 * np.sqrt(rho1) + H2 *
         np.sqrt(rho2)) / (np.sqrt(rho1) + np.sqrt(rho2))
    c = np.sqrt((gamma - 1) * (H - 0.5 * u**2))

    # create matrix of eigenvector Leveque s301
    R = np.array([
        [1, 1, 1],
        [u-c, u, u+c],
        [H-u*c, 0.5*u**2, H+u*c]]
    )

    # Find wave speeds
    delta = Q2 - Q1
    alpha = np.linalg.inv(R) @ delta

    # create vector of eigenvalues
    lam = np.array([u-c, u, u+c])

    return u, H, c, alpha, R, lam


def euler_1d_roe(N_cell, Q0, L, T):
    """
    Solve 1d euler with Roe method for polytropic gas

    TODO:
    1. Plott trykk, temperatur (er vel ikke noe temperatur?), hastighet, tetthet X
    2. Gå fra primitive til konserverte variabler og tilbake X
    3. Bare konstante verdier og se at ingen ting endrer seg X
    3.5 Bare konstant trykk (må vel også ha konstant tetthet?) X
    4. Hvorfor kommer det masse bakover?
        - dette ser ut til å fikse seg når man bytter fortegn på fluxen
    5. Regn ut roe average for eulerlikningene selv X
    6. Teste A(Qi - Qi-1) = f(Qi) - f(Qi-1) X
    7. Matematika eller Maple
    8. For å bruke thermopack må h byttes ut med e eller s
    9. Løse matriseproblem for å få egenevektor basis numerisk X
        - har testet og gir samme svar som analytisk løsning
    10. start 14.8 og sjekke at jeg får det samme når jeg legger inn ideel gas
        - Enklest med disse variablene: rho, u, e
        - E = rho (e + 1/2 u^2)
        - evt bytte etterpå
    11. Sjekke hvorfor alpha_test2 = Lam R^-1 delta f gir rett svar, mens 1 og 3 blir feil X
        - 2 inneholder bare u X
        -funker nå
    12. Hva er sammenhengen mellom celleverdien av c og c1 og c2?
        -Dette har jeg ikke forstått enda, men ser ut til å funke når man bruker max(c1, c2) i CFL 
    13. Er coordinatene i f egt (rho, rhou, H) ??? X
            - nå tror jeg at det er riktig med (rho, rho u, E)
    14. Gjøre om til thermopack
            - rho_hat, H_hat og u_hat så kan vi gjøre en flash med (rho_hat, h_hat) 
            - h = H - 0.5 u^2
            - case i gassfasen C02,10bar,400k
            - h_hat = enthalpy_tv(T, 1/rho_hat, ...) løst for T
            - husk molbasis
    """
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
            u1, h1, c1, alpha1, R1, lam1 = get_roe_avg(Q[:, i-1], Q[:, i])
            u2, h2, c2, alpha2, R2, lam2 = get_roe_avg(Q[:, i], Q[:, i+1])

            # find max c
            if max(c1, c2) > c:
                c = max(c1, c2)

            # TEST RH condition TODO har vi samme kordinater på f og A?
            A = R1 @ np.diag(lam1) @ np.linalg.inv(R1)
            delta = Q[:, i] - Q[:, i-1]
            if not np.allclose(A @ delta - (f(Q[:, i]) - f(Q[:, i-1])), 0):
                print("RH condition: ", "A delta - (f(Q[:,i]) - f(Q[:,i-1]))", A @
                      delta - (f(Q[:, i]) - f(Q[:, i-1])), "for cell ", i)

            lam1p = 0.5 * (lam1 + np.abs(lam1))
            lam2m = 0.5 * (lam2 - np.abs(lam2))

            # calculate A^{\mp} Delta Q_{i \pm 1/2}
            ApDQ1[:, i] = R1 @ np.diag(lam1p) @ alpha1
            AmDQ2[:, i] = R2 @ np.diag(lam2m) @ alpha2

        dt = min(0.4 * dx / (c + max(abs(Q[1, 1:-1]))), T-t)
        if np.any(np.isnan(- dt / dx * (AmDQ2 + ApDQ1))):
            print("NAN")
            return Q
        else:
            # TODO sjekk fortegn!!!! ustabilitet gikk bort men fortsatt feil svar
            Q = Q - dt / dx * (AmDQ2 + ApDQ1)

        # boundary condition
        Q[:, 0] = Q[:, 1]
        Q[:, -1] = Q[:, -2]

        t += dt
        print(t)

    return Q


if __name__ == "__main__":
    N_cell = 102
    # sod shock tube TODO reset u = 0
    rho_l, u_l, p_l = 1, 0, 1
    rho_r, u_r, p_r = 0.125, 0, 0.1

    # initial conditions with ghost cells
    rho0 = np.concatenate(
        [np.ones(N_cell//2) * rho_l, np.ones(N_cell - N_cell//2) * rho_r])
    u0 = np.concatenate(
        [np.ones(N_cell//2) * u_l, np.ones(N_cell - N_cell//2) * u_r])
    p0 = np.concatenate(
        [np.ones(N_cell//2) * p_l, np.ones(N_cell - N_cell//2) * p_r])

    Q0 = get_conservative(rho0, u0, p0)

    L = 1
    T = 0.125
    Qlf = euler_1d_lf(N_cell, Q0, L, T)
    Qroe = euler_1d_roe(N_cell, Q0, L, T)
    rho_roe, u_roe, p_roe = get_primitive(Qroe)
    rho_lf, u_lf, p_lf = get_primitive(Qlf)
    # plot
    x = np.linspace(0, L, len(p0))

    rho0_test, u0_test, p0_test = get_primitive(get_conservative(rho0, u0, p0))

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
