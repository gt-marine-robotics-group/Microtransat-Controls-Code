# model.py
from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    m: float
    Iz: float
    rho_w: float
    rho_a: float
    S_r: float
    l_r: float
    Sf: float
    CDx: float
    Yv_coeff: float
    C_b: float
    Nv_coeff: float
    Va: float
    betaA: float
    x_ce_m: float
    y_ce_m: float
    x_ce_t: float
    y_ce_t: float
    CLalpha_r: float

@dataclass
class SailModel:
    """
    Runtime sail model: table/surrogate functions.
    Must provide forces (L,D) given alpha (deg) and wind speed.
    """
    def main(self, alpha_deg: float):
        raise NotImplementedError

    def back(self, alpha_deg: float):
        raise NotImplementedError


def boat3dof_rhs(x, u, params: Params, sail_model: SailModel):
    # x=[u,v,r], u=[delta_r, delta_w]
    u_b, v_b, r = x
    delta_r, delta_w = u

    X, Y, N = total_boat_forces(u_b, v_b, r, delta_r, delta_w, params, sail_model)

    xdot = np.zeros(3)
    xdot[0] = (X / params.m) + v_b * r
    xdot[1] = (Y / params.m) - u_b * r
    xdot[2] = (N / params.Iz)
    return xdot


def total_boat_forces(u, v, r, delta_r, delta_w, params: Params, sail_model: SailModel):
    # Map "delta_w" to main/back AoA the same way you did
    alpha_m_deg = np.rad2deg(delta_w)
    k_relation = 1.09
    alpha_t_deg = np.rad2deg(-delta_w)
    alpha_t_deg = k_relation * alpha_t_deg

    main = sail_model.main(alpha_m_deg)
    back = sail_model.back(alpha_t_deg)

    Lm, Dm = main["L"], main["D"]
    Lt, Dt = back["L"], back["D"]

    # apparent wind vector (in body frame assumption)
    Va_x = params.Va * np.cos(params.betaA)
    Va_y = params.Va * np.sin(params.betaA)

    gamma = np.arctan2(v - Va_y, u - Va_x)

    # rotate lift/drag into body axes (your sign convention)
    Dx_m = -Dm * np.cos(gamma)
    Dy_m = -Dm * np.sin(gamma)
    Lx_m = -Lm * np.sin(gamma)
    Ly_m =  Lm * np.cos(gamma)
    Xs_m = Lx_m + Dx_m
    Ys_m = Ly_m + Dy_m

    Dx_t = -Dt * np.cos(gamma)
    Dy_t = -Dt * np.sin(gamma)
    Lx_t = -Lt * np.sin(gamma)
    Ly_t =  Lt * np.cos(gamma)
    Xs_t = Lx_t + Dx_t
    Ys_t = Ly_t + Dy_t

    Ns_m = Ys_m * params.x_ce_m + Xs_m * params.y_ce_m
    Ns_t = Ys_t * params.x_ce_t + Xs_t * params.y_ce_t

    # hull drag (quadratic in u)
    D_x = 0.5 * params.rho_w * params.Sf * params.CDx * u * abs(u)
    X_hull = -D_x
    Y_hull = -params.Yv_coeff * v

    # rudder lift (flat plate linear)
    alpha_r = delta_r
    q_w = 0.5 * params.rho_w * max(u*u, 0.01)
    Lr = q_w * params.S_r * params.CLalpha_r * alpha_r

    Xr_r = 0.0
    Yr_r = Lr

    N_damp = -params.C_b * r
    N_r = params.l_r * Lr
    N_stiff = -params.Nv_coeff * v

    X = Xs_m + Xs_t + Xr_r + X_hull
    Y = Ys_m + Ys_t + Yr_r + Y_hull
    N = Ns_m + Ns_t + N_r + N_damp + N_stiff
    return X, Y, N