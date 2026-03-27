# linearize_and_control.py
import numpy as np
from scipy.optimize import root
from scipy.linalg import solve_continuous_are, eig
from model import boat3dof_rhs, total_boat_forces, Params

def trim_equations(z, u0, params: Params, sail_model):
    delta_r, delta_w, v = z
    r0 = 0.0
    X, Y, N = total_boat_forces(u0, v, r0, delta_r, delta_w, params, sail_model)
    return np.array([X, Y, N])

def solve_trim(u0_des, params: Params, sail_model, z0=None):
    if z0 is None:
        z0 = np.array([0.0, np.deg2rad(5.0), 0.0])
    sol = root(lambda z: trim_equations(z, u0_des, params, sail_model), z0, method="hybr")
    if not sol.success:
        raise RuntimeError(f"Trim failed: {sol.message}")
    delta_r0, delta_w0, v0 = sol.x
    x_trim = np.array([u0_des, v0, 0.0])
    u_trim = np.array([delta_r0, delta_w0])
    return x_trim, u_trim, sol

def linearize(x_trim, u_trim, params: Params, sail_model):
    nx, nu = 3, 2
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    u0 = x_trim[0]
    h_u = 0.02 * max(1.0, abs(u0))
    h_v = 0.02 * max(1.0, max(abs(u0), 1.0))
    h_r = np.deg2rad(1.0)
    hx = np.array([h_u, h_v, h_r])

    h_dr = np.deg2rad(2.0)
    h_dw = np.deg2rad(2.0)
    hu = np.array([h_dr, h_dw])

    f0 = boat3dof_rhs(x_trim, u_trim, params, sail_model)

    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = hx[i]
        fp = boat3dof_rhs(x_trim + dx, u_trim, params, sail_model)
        fm = boat3dof_rhs(x_trim - dx, u_trim, params, sail_model)
        A[:, i] = (fp - fm) / (2.0 * hx[i])

    for j in range(nu):
        du = np.zeros(nu)
        du[j] = hu[j]
        fp = boat3dof_rhs(x_trim, u_trim + du, params, sail_model)
        fm = boat3dof_rhs(x_trim, u_trim - du, params, sail_model)
        B[:, j] = (fp - fm) / (2.0 * hu[j])

    return A, B, f0

def lqr(A, B, Q, R):
    # Continuous-time LQR
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    poles = eig(A - B @ K)[0]
    return K, P, poles

def lqe(A, C, Qn, Rn):
    # Continuous-time Kalman filter (dual to LQR)
    # Solve ARE for estimator: A^T P + P A - P C^T R^-1 C P + Q = 0
    P = solve_continuous_are(A.T, C.T, Qn, Rn)
    L = P @ C.T @ np.linalg.inv(Rn)
    poles = eig(A - L @ C)[0]
    return L, P, poles