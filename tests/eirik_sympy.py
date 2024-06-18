# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:29:05 2019
 
SYMPY attempt
@author: alexa
"""

import sympy as sp
import numpy as np


def pressure(rho, u, H):  # test for ideal gas?
    return ((gamma-1)/gamma)*(H-0.5*u**2)*rho


def pressure_h(rho, h):  # test for ideal gas?
    return ((gamma-1)/gamma)*(h)*rho


def pressure_e(rho, e):  # test for ideal gas?
    return rho*(gamma-1)*e


def internal_e(rho, p):
    return p/(rho*(gamma-1))


def enthalpy(rho, u, p, e):
    return e + 0.5*u**2 + p/rho


def csqrd(rho, u, H):
    return (gamma-1)*(H-0.5*u**2)


def csqrd_e(rho, e, p):
    return (gamma-1)*(e+p/rho)


def csqrd_p(rho, p):
    return (gamma-1)*(p/(rho*(gamma-1))+p/rho)


sp.init_printing()
gamma = sp.symbols("gamma")
Gamma = gamma-1

e_L = sp.symbols("e_L")
u_L = sp.symbols("u_L")
rho_L = sp.symbols("rho_L")
E_L = rho_L*(e_L+0.5*u_L**2)
p_L = pressure_e(rho_L, e_L)
H_L = enthalpy(rho_L, u_L, p_L, e_L)

e_R = sp.symbols("e_R")
u_R = sp.symbols("u_R")
rho_R = sp.symbols("rho_R")
E_R = rho_R*(e_R+0.5*u_R**2)
p_R = pressure_e(rho_R, e_R)
H_R = enthalpy(rho_R, u_R, p_R, e_R)

DeltaU = sp.Matrix([rho_R-rho_L,
                    rho_R*u_R-rho_L*u_L,
                    E_R-E_L])

DeltaF = sp.Matrix([rho_R*u_R-rho_L*u_L,
                    rho_R*u_R**2+p_R - (rho_L*u_L**2+p_L),
                    (E_R+p_R)*u_R-(E_L+p_L)*u_L])

rho_hat = sp.sqrt(rho_L*rho_R)
h_hat = (sp.sqrt(rho_L)*(e_L+p_L/rho_L)+sp.sqrt(rho_R) *
         (e_R+p_R/rho_R))/(sp.sqrt(rho_L)+sp.sqrt(rho_R))
# rho_hat = (rho_L+rho_R)/(rho_L*rho_R)
p_hat = pressure_h(rho_hat, h_hat)
# e_hat = h_hat - p_hat/rho_hat
u_hat = (sp.sqrt(rho_L)*u_L+sp.sqrt(rho_R)*u_R)/(sp.sqrt(rho_L)+sp.sqrt(rho_R))
H_hat = (sp.sqrt(rho_L)*H_L+sp.sqrt(rho_R)*H_R)/(sp.sqrt(rho_L)+sp.sqrt(rho_R))
# E_hat = (sqrt(rho_L)*(e_L+0.5*u_L**2)+sqrt(rho_R)*(e_R+0.5*u_R**2))/(sqrt(rho_L)+sqrt(rho_R))
# e_hat = E_hat-0.5*u_hat**2
# p_hat = pressure_e(rho_hat, e_hat)
# e_hat = (sqrt(rho_L)*e_L+sqrt(rho_R)*e_R)/(sqrt(rho_L)+sqrt(rho_R))
# p_hat = pressure(rho_hat, u_hat, H_hat)
# e_hat = internal_e(rho_hat, p_hat)
# p_hat = (sp.sqrt(rho_L)*p_L+sp.sqrt(rho_R)*p_R)/(sp.sqrt(rho_L)+sp.sqrt(rho_R))
e_hat = H_hat - p_hat/rho_hat - 0.5*u_hat**2
# csqrd_hat = csqrd(rho_hat, u_hat, H_hat)
# NB this doesn't give 0 if only csqrd(e) is used
csqrd_hat = csqrd_e(rho_hat, e_hat, p_hat)
c_hat = sp.sqrt(csqrd_hat)
# e_tilde = (1/gamma)*(e_hat + p_hat/rho_hat)
# csqrd_hat = (gamma-1)*(e_tilde+(gamma-1)*e_tilde)
# csqrd_hat = (gamma-1)*(e_hat+(gamma-1)*e_hat)
# csqrd_hat = (gamma-1)*h_hat
# csqrd_hat = csqrd_p(rho_hat, p_hat)

A_hat_alx = sp.Matrix([[0, 1, 0],
                       [csqrd_hat-u_hat**2-Gamma*(e_hat+p_hat/rho_hat - 0.5*u_hat**2),
                        2*u_hat - Gamma*u_hat, Gamma],
                       [u_hat*(csqrd_hat-(Gamma+1)*(e_hat+0.5*u_hat**2+p_hat/rho_hat)+Gamma*u_hat**2),
                        e_hat+p_hat/rho_hat + 0.5*u_hat**2-Gamma*u_hat**2,
                        (Gamma + 1)*u_hat]])

# dette klarer ikke sympy å forenkle
R_hat = sp.Matrix([
    [1, 1, 1],
    [u_hat-c_hat, u_hat, u_hat+c_hat],
    [H_hat-u_hat*c_hat, H_hat - c_hat**2 / Gamma, H_hat+u_hat*c_hat]]
)
lam = sp.Matrix([[u_hat-c_hat, 0, 0],
                 [0, u_hat, 0],
                 [0, 0, u_hat + c_hat]])

A_hat = R_hat * lam * R_hat.inv()
print(sp.simplify(A_hat-A_hat_alx))

# print(sp.simplify(DeltaF-A_hat*DeltaU))
