# simulate_closed_loop.py
import numpy
import scipy
import pandas
import openpyxl
import control
import numpy as np
from scipy.integrate import solve_ivp
from model import boat3dof_rhs

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def closed_loop_rhs(t, x, params, sail_model, x_trim, u_trim, K):
    x_tilde = x - x_trim
    u_cmd = u_trim - K @ x_tilde

    delta_r_max = np.deg2rad(20.0)
    delta_w_max = np.deg2rad(12.0)

    u_cmd[0] = float(clamp(u_cmd[0], -delta_r_max, delta_r_max))
    u_cmd[1] = float(clamp(u_cmd[1], -delta_w_max, delta_w_max))

    return boat3dof_rhs(x, u_cmd, params, sail_model)

def run_sim(x0, t_end, params, sail_model, x_trim, u_trim, K, dt=0.01):
    # Guarantee last time is <= t_end (avoid floating-point overshoot)
    n = int(np.floor(t_end / dt))
    t_eval = np.linspace(0.0, n * dt, n + 1)

    sol = solve_ivp(
        fun=lambda t, x: closed_loop_rhs(t, x, params, sail_model, x_trim, u_trim, K),
        t_span=(0.0, t_eval[-1]),
        y0=x0,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9
    )
    return sol.t, sol.y.T