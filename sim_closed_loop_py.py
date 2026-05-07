# sim_closed_loop_py.py
import numpy as np
from scipy.integrate import solve_ivp

from model import boat3dof_rhs


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def closed_loop_rhs(t, x, params, sail_model, x_trim, u_trim, K):
    """
    Original 3-DOF LQR closed-loop simulation.

    State:
        x = [u, v, r]
    """

    x_tilde = x - x_trim
    u_cmd = u_trim - K @ x_tilde

    delta_r_max = np.deg2rad(20.0)
    delta_w_max = np.deg2rad(12.0)

    u_cmd[0] = float(clamp(u_cmd[0], -delta_r_max, delta_r_max))
    u_cmd[1] = float(clamp(u_cmd[1], -delta_w_max, delta_w_max))

    return boat3dof_rhs(x, u_cmd, params, sail_model)


def run_sim(
    x0,
    t_end,
    params,
    sail_model,
    x_trim,
    u_trim,
    K,
    dt=0.01,
):
    """
    Original 3-DOF closed-loop LQR simulation.
    """

    n = int(np.floor(t_end / dt))
    t_eval = np.linspace(0.0, n * dt, n + 1)

    sol = solve_ivp(
        fun=lambda t, x: closed_loop_rhs(
            t,
            x,
            params,
            sail_model,
            x_trim,
            u_trim,
            K,
        ),
        t_span=(0.0, t_eval[-1]),
        y0=x0,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    return sol.t, sol.y.T


def closed_loop_heading_rhs(
    t,
    x,
    params,
    sail_model,
    x_trim,
    u_trim,
    K,
    psi_cmd,
    heading_pid,
):
    """
    Full nonlinear closed-loop model with outer heading loop.

    Full state:
        x = [u, v, r, psi]

    Inner LQR state:
        x_inner = [u, v, r]

    Control:
        heading error -> desired yaw rate r_cmd
        LQR tracks [u_trim, v_trim, r_cmd]
    """

    u_b = x[0]
    v_b = x[1]
    r = x[2]
    psi = x[3]

    x_inner = np.array([u_b, v_b, r], dtype=float)

    # -------------------------
    # Outer heading controller
    # -------------------------

    psi_err = wrap_to_pi(psi_cmd - psi)
    psi_err_dot = -r

    Kp = heading_pid.get("Kp", 0.8)
    Kd = heading_pid.get("Kd", 0.15)
    r_max = heading_pid.get("r_max", np.deg2rad(20.0))

    r_cmd = Kp * psi_err + Kd * psi_err_dot
    r_cmd = float(clamp(r_cmd, -r_max, r_max))

    # -------------------------
    # Inner LQR controller
    # -------------------------

    x_ref = np.array([
        x_trim[0],
        x_trim[1],
        r_cmd,
    ], dtype=float)

    x_tilde = x_inner - x_ref

    u_cmd = u_trim - K @ x_tilde

    # -------------------------
    # Actuator saturation
    # -------------------------

    delta_r_max = np.deg2rad(20.0)
    delta_w_max = np.deg2rad(12.0)

    u_cmd[0] = float(clamp(u_cmd[0], -delta_r_max, delta_r_max))
    u_cmd[1] = float(clamp(u_cmd[1], -delta_w_max, delta_w_max))

    # -------------------------
    # Nonlinear plant
    # -------------------------

    xdot_inner = boat3dof_rhs(
        x_inner,
        u_cmd,
        params,
        sail_model,
    )

    psi_dot = r

    xdot = np.array([
        xdot_inner[0],
        xdot_inner[1],
        xdot_inner[2],
        psi_dot,
    ])

    return xdot


def run_heading_sim(
    x0_3dof,
    psi0,
    psi_cmd,
    t_end,
    params,
    sail_model,
    x_trim,
    u_trim,
    K,
    heading_pid=None,
    dt=0.01,
):
    """
    Runs nonlinear closed-loop heading simulation.

    Inputs:
        x0_3dof : initial [u, v, r]
        psi0    : initial heading [rad]
        psi_cmd : commanded heading [rad]

    Output:
        t       : time vector
        x_hist  : state history [u, v, r, psi]
    """

    if heading_pid is None:
        heading_pid = {
            "Kp": 0.8,
            "Ki": 0.0,
            "Kd": 0.15,
            "r_max": np.deg2rad(20.0),
        }

    x0 = np.array([
        x0_3dof[0],
        x0_3dof[1],
        x0_3dof[2],
        psi0,
    ], dtype=float)

    n = int(np.floor(t_end / dt))
    t_eval = np.linspace(0.0, n * dt, n + 1)

    sol = solve_ivp(
        fun=lambda t, x: closed_loop_heading_rhs(
            t,
            x,
            params,
            sail_model,
            x_trim,
            u_trim,
            K,
            psi_cmd,
            heading_pid,
        ),
        t_span=(0.0, t_eval[-1]),
        y0=x0,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"Heading simulation failed: {sol.message}")

    return sol.t, sol.y.T
