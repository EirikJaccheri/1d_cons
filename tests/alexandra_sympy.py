# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:29:05 2019
 
SYMPY attempt
@author: alexa
"""
 
from sympy import *
import numpy as np
 
def pressure(rho, u, H): #test for ideal gas?
    return ((gamma-1)/gamma)*(H-0.5*u**2)*rho
 
def pressure_h(rho, h): #test for ideal gas?
    return ((gamma-1)/gamma)*(h)*rho
 
def pressure_e(rho, e): #test for ideal gas?
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
 
init_printing()
gamma = symbols("gamma")
Gamma = gamma-1
 
e_L = symbols("e_L")
u_L = symbols("u_L")
rho_L = symbols("rho_L")
E_L = rho_L*(e_L+0.5*u_L**2)
p_L = pressure_e(rho_L, e_L)
H_L = enthalpy(rho_L, u_L, p_L, e_L)
 
e_R = symbols("e_R")
u_R = symbols("u_R")
rho_R = symbols("rho_R")
E_R = rho_R*(e_R+0.5*u_R**2)
p_R = pressure_e(rho_R, e_R)
H_R = enthalpy(rho_R, u_R, p_R, e_R)
 
DeltaU = Matrix([rho_R-rho_L, 
                 rho_R*u_R-rho_L*u_L,
                 E_R-E_L])
 
DeltaF = Matrix([rho_R*u_R-rho_L*u_L, 
                 rho_R*u_R**2+p_R - (rho_L*u_L**2+p_L),
                 (E_R+p_R)*u_R-(E_L+p_L)*u_L])
 
rho_hat = sqrt(rho_L*rho_R)
h_hat = (sqrt(rho_L)*(e_L+p_L/rho_L)+sqrt(rho_R)*(e_R+p_R/rho_R))/(sqrt(rho_L)+sqrt(rho_R))
#rho_hat = (rho_L+rho_R)/(rho_L*rho_R)
#p_hat = pressure_h(rho_hat, h_hat)
#e_hat = h_hat - p_hat/rho_hat
u_hat = (sqrt(rho_L)*u_L+sqrt(rho_R)*u_R)/(sqrt(rho_L)+sqrt(rho_R))
H_hat = (sqrt(rho_L)*H_L+sqrt(rho_R)*H_R)/(sqrt(rho_L)+sqrt(rho_R))
#E_hat = (sqrt(rho_L)*(e_L+0.5*u_L**2)+sqrt(rho_R)*(e_R+0.5*u_R**2))/(sqrt(rho_L)+sqrt(rho_R))
#e_hat = E_hat-0.5*u_hat**2
#p_hat = pressure_e(rho_hat, e_hat)
#e_hat = (sqrt(rho_L)*e_L+sqrt(rho_R)*e_R)/(sqrt(rho_L)+sqrt(rho_R))
#p_hat = pressure(rho_hat, u_hat, H_hat)
#e_hat = internal_e(rho_hat, p_hat)
p_hat = (sqrt(rho_L)*p_L+sqrt(rho_R)*p_R)/(sqrt(rho_L)+sqrt(rho_R))
e_hat = H_hat - p_hat/rho_hat -0.5*u_hat**2
##csqrd_hat = csqrd(rho_hat, u_hat, H_hat)
csqrd_hat = csqrd_e(rho_hat, e_hat, p_hat)  # NB this doesn't give 0 if only csqrd(e) is used
#e_tilde = (1/gamma)*(e_hat + p_hat/rho_hat)
#csqrd_hat = (gamma-1)*(e_tilde+(gamma-1)*e_tilde)
#csqrd_hat = (gamma-1)*(e_hat+(gamma-1)*e_hat)
##csqrd_hat = (gamma-1)*h_hat
#csqrd_hat = csqrd_p(rho_hat, p_hat)
 
A_hat = Matrix([[0, 1, 0],
              [csqrd_hat-u_hat**2-Gamma*(e_hat+p_hat/rho_hat -0.5*u_hat**2),
               2*u_hat - Gamma*u_hat, Gamma],
              [u_hat*(csqrd_hat-(Gamma+1)*(e_hat+0.5*u_hat**2+p_hat/rho_hat)+Gamma*u_hat**2),
               e_hat+p_hat/rho_hat + 0.5*u_hat**2-Gamma*u_hat**2,
               (Gamma +1)*u_hat]])
 
 
print(simplify(DeltaF-A_hat*DeltaU))