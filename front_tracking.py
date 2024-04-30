import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import copy


def solve_RP(ul, ur, dt):
    def f(u): return u * (1 - u)
    x = np.linspace(-1, 1)

    if ul <= ur:
        def df(u): return (f(ul) - f(ur)) / (ul - ur)
        def dfinv(z): return None
    else:
        def dfinv(z): return (1 - z) / 2
        def df(u): return 1 - 2 * u

    df_left = df(ul)
    df_right = df(ur)

    u = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < df_left * dt:
            u[i] = ul
        elif df_left * dt <= x[i] <= df_right * dt:
            u[i] = dfinv(x[i] / dt)
        else:
            u[i] = ur

    return x, u


def solve_IV(u, x, dt):
    for i in range(len(u) - 1):
        x_sol, u_sol = solve_RP(u[i], u[i+1], dt)
        plt.plot(x_sol + x[i], u_sol)
    plt.show()


def front_velocity(u_l, u_r, f):
    """calculate velocity s of single front"""
    if u_l != u_r:
        return (f(u_l) - f(u_r)) / (u_l - u_r)
    else:
        return 0


def calculate_front_velocities(xlast, ulast, uint, f):
    """ calculate velocity of all fronts in the system"""
    u = []
    s = []
    x = []
    for i in range(len(xlast)):
        u_l, u_r = ulast[i], ulast[i+1]
        if u_l < u_r:
            # u_l < u_r follow the flux function
            # must iterate through the inflection points of the flux function between u_l and u_r
            # if there are no inflection points, we still need a velocity...
            contains_inflection = False
            for j in range(len(uint) - 1):
                if u_l < uint[j] < u_r:
                    contains_inflection = True
                    s_l = front_velocity(u_l, uint[j], f)
                    s_r = front_velocity(uint[j], u_r, f)
                    u.append(u_l)
                    u.append((u_l + u_r) / 2)
                    s.append(s_l)
                    s.append(s_r)

                    # we now have two shocks at the same position
                    x.append(xlast[i])
                    x.append(xlast[i])
            if not contains_inflection:
                si = front_velocity(u_l, u_r, f)
                u.append(u_l)
                s.append(si)
                x.append(xlast[i])

        else:
            # u_l > u_r shock
            si = front_velocity(u_l, u_r, f)
            u.append(u_l)
            s.append(si)
            x.append(xlast[i])
        # append last value
    u.append(ulast[-1])
    return u, s, x


def calc_minimum_time_step(x, s):
    """ calculate time step to first colision, and index of left front in the colision"""
    dt = np.inf
    icol = None
    for i in range(len(s)-1):
        dti = - (x[i+1] - x[i]) / (s[i+1] - s[i])
        if 0 < dti < dt:
            dt = dti
            icol = i
    if icol == None:
        dt = 0
    return dt


def remove_colisions(u, x):
    """ remove coliding fronts and update u"""
    # xnew, unew = copy.deepcopy(x), copy.deepcopy(u)
    pop_indices = []
    for i in range(len(x)-1):
        if np.isclose(x[i], x[i+1]):
            pop_indices.append(i+1)
    xnew = [x[i] for i in range(len(x)) if i not in pop_indices]
    unew = [u[i] for i in range(len(u)) if i not in pop_indices]
    return unew, xnew


def front_tracking(x0: np.array, u0: np.array, uint: np.array, fint: np.array):
    """Solve scalar rieman problem using front tracking method.

    NB: assume convex flux function

    Input:
    x0: array of len (n) of fronts
    u0: array of len (n + 1) of intitial u values in intervals defined by x0
    uint: array of len (n + 2)  inflection points for flux function f(u)
    fint: array of len (n + 2)  flux function evaluated at u0
    """

    f = interp1d(uint, fint)

    dt = 1
    u = u0
    ulast = u
    xlast = x0
    ulist = []
    xlist = []
    slist = []
    tlist = []
    t = 0
    while dt > 0:
        print("t", t)
        # calculate velocities s of fronts x with values u
        # TODO: only update coliding fronts
        u, s, x = calculate_front_velocities(xlast, ulast, uint, f)
        # store values for plotting
        ulist.append(u)
        xlist.append(x)
        slist.append(s)
        tlist.append(t)
        # calculate time step dt to first colision, and index of left front in the colision
        dt = calc_minimum_time_step(x, s)
        t += dt
        # propagate fronts to time t + dt
        x = [x[i] + s[i] * dt for i in range(len(x))]
        # remove coliding fronts and update u
        u, x = remove_colisions(u, x)
        ulast, xlast = u, x

    # plot results
    return ulist, xlist, slist, tlist


def Lax_Friedrichs(u, x, dt, T, N_cell, f):
    dx = (x[-1] - x[0]) / N_cell
    for i in range(int(T / dt)):
        u[1:-1] = (u[:-2] + u[2:]) / 2 - dt / \
            (2*dx) * (f(u[2:]) - f(u[:-2]))
        # boundary condition
        u[0] = u[1]
        u[-1] = u[-2]
    return u


def plot_fronts(ulist, xlist, slist, tlist):
    N = 100
    for i in range(len(tlist)):
        if i < len(tlist) - 1:
            tarr = np.linspace(0, tlist[i+1] - tlist[i], N)
        else:
            tarr = np.linspace(0, 2, N)
        for j in range(len(xlist[i])):
            plt.plot(xlist[i][j] + slist[i][j]*tarr, tlist[i] + tarr)

    plt.show()


def plot_velocity_profiles(ulist, xlist, slist, tlist, t, x_arr=None, u_arr=None):
    N = 100
    dx = 10
    print("tlist", tlist)
    print("xlist", xlist)
    print("slist", slist)
    print("ulist", ulist)

    for i in range(len(tlist)):
        if tlist[i] <= t < tlist[i+1]:
            xt = np.array(xlist[i]) + np.array(slist[i]) * (t - tlist[i])
            x = np.linspace(xt[0] - dx, xt[0], N)
            plt.plot(x, ulist[i][0]*np.ones_like(x))
            for j in range(1, len(ulist[i])-1):
                print(j)
                x = np.linspace(xt[j-1], xt[j], N)
                plt.plot(x, ulist[i][j]*np.ones_like(x))
            x = np.linspace(xt[-1], xt[-1] + dx, N)
            plt.plot(x, ulist[i][-1]*np.ones_like(x))
            plt.plot()
            break

    if x_arr is not None and u_arr is not None:
        plt.plot(x_arr, u_arr, label="Lax-Friedrichs")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # u = np.array([1, 0.2, 0.4, 0.6, 0.4])
    # x = np.array([-1, 0, 1, 2 ])
    # solve_IV(u, x, 1)

    X0 = np.array([0, 2, 4])
    U0 = np.array([2, -1, 1, -1])
    Uint = np.array([-2, -1, 0, 1, 2, 3, 4, 5])
    def F(u): return 1/2*u**2
    Fint = F(Uint)
    T = 0.5
    ULIST, XLIST, SLIST, TLIST = front_tracking(X0, U0, Uint, Fint)
    # plot_fronts(ULIST, XLIST, SLIST, TLIST)

    # Lax-Friedrichs
    N_CELL = 1000
    DX_EDGE = 10
    X_LF = np.linspace(X0[0]-DX_EDGE, X0[-1]+DX_EDGE, N_CELL)
    U_LF = np.ones_like(X_LF)*U0[0]
    for i in range(len(X_LF)):
        for j in range(len(X0)-1):
            if X0[j] < X_LF[i] < X0[j+1]:
                print("j", j, ": ", X0[j], "<", X_LF[i], "<", X0[j+1])
                U_LF[i] = U0[j+1]
                break
            if X_LF[i] > X0[-1]:
                U_LF[i] = U0[-1]

    DT = 0.001
    U_SOL_LF = Lax_Friedrichs(U_LF, X_LF, DT, T, N_CELL, F)
    plot_velocity_profiles(ULIST, XLIST, SLIST, TLIST,
                           T, x_arr=X_LF, u_arr=U_SOL_LF)
