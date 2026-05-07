# main.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from aero_vlm_py import main_sail_vlm, back_sail_vlm
from sails_runtime_py import TableSailModel
from model import Params
from linearize_and_control_py import solve_trim, linearize, lqr
from sim_closed_loop_py import run_heading_sim


def build_tables(
    b_m: float = 1.3,
    alpha_min: float = 0.0,
    alpha_max: float = 12.0,
    n_alpha: int = 13,
    outfile_name: str = "vlm_tables.xlsx",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Path]:

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / outfile_name

    b_t = b_m * (5.865 / 10.0)

    alphas_deg = np.linspace(alpha_min, alpha_max, n_alpha)

    Lm_list, Dm_list, Lt_list, Dt_list = [], [], [], []
    rows = []

    for a in alphas_deg:

        m = main_sail_vlm(b_m, float(a))
        t = back_sail_vlm(b_t, float(a))

        Lm = float(m["L"])
        Dm = float(m["D"])

        Lt = float(t["L"])
        Dt = float(t["D"])

        Lm_list.append(Lm)
        Dm_list.append(Dm)

        Lt_list.append(Lt)
        Dt_list.append(Dt)

        rows.append({

            "alpha_deg": float(a),

            # main sail
            "ARm": float(m.get("AR", np.nan)),
            "Sm": float(m.get("S", np.nan)),
            "bm": float(m.get("b", np.nan)),
            "macm": float(m.get("mac", np.nan)),
            "CLm": float(m.get("CL", np.nan)),
            "CDm": float(m.get("CD", np.nan)),
            "CMm": float(m.get("CM", np.nan)),
            "Liftm": Lm,
            "Dragm": Dm,
            "CLCDm": float(m.get("CLCD", np.nan)),
            "cr": float(m.get("cr", np.nan)),
            "ct": float(m.get("ct", np.nan)),
            "sweep": float(m.get("sweep", np.nan)),

            # back sail
            "ARt": float(t.get("AR", np.nan)),
            "St": float(t.get("S", np.nan)),
            "bt": float(t.get("b", np.nan)),
            "mact": float(t.get("mac", np.nan)),
            "CLt": float(t.get("CL", np.nan)),
            "CDt": float(t.get("CD", np.nan)),
            "CMt": float(t.get("CM", np.nan)),
            "Liftt": Lt,
            "Dragt": Dt,
            "CLCDt": float(t.get("CLCD", np.nan)),
        })

    Lm_arr = np.asarray(Lm_list, dtype=float)
    Dm_arr = np.asarray(Dm_list, dtype=float)

    Lt_arr = np.asarray(Lt_list, dtype=float)
    Dt_arr = np.asarray(Dt_list, dtype=float)

    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)

    print("\n=== VLM TABLE BUILD ===")
    print("Script folder :", out_dir)
    print("CWD (terminal) :", Path.cwd())
    print("Wrote Excel    :", out_path)
    print("=======================\n")

    return alphas_deg, Lm_arr, Dm_arr, Lt_arr, Dt_arr, df, out_path


def main():

    plt.close("all")

    # ============================================================
    # 1) Build VLM aerodynamic lookup tables
    # ============================================================

    alphas_deg, Lm, Dm, Lt, Dt, df, out_path = build_tables(
        b_m=1.3,
        alpha_min=0.0,
        alpha_max=12.0,
        n_alpha=13,
        outfile_name="vlm_tables.xlsx",
    )

    # ============================================================
    # 2) Runtime sail interpolation model
    # ============================================================

    sail_model = TableSailModel(
        alpha_deg=np.asarray(alphas_deg, dtype=float),
        L_main=np.asarray(Lm, dtype=float),
        D_main=np.asarray(Dm, dtype=float),
        L_back=np.asarray(Lt, dtype=float),
        D_back=np.asarray(Dt, dtype=float),
    )

    # ============================================================
    # 3) Boat / sail / rudder parameters
    # ============================================================

    params = Params(

        # mass properties
        m=10.0,
        Iz=25.0,

        # fluid properties
        rho_w=1025.0,
        rho_a=1.225,

        # rudder
        S_r=0.1524 * 0.1524,
        l_r=0.5,
        CLalpha_r=5.0,

        # hull
        Sf=0.9 * 1.0 * (0.2 + 2.0 * 0.25),
        CDx=0.6 / 1025.0,

        # hydrodynamic stability
        Yv_coeff=40.0,
        C_b=0.5 * 1025.0 * (1.143**5) * 0.01,
        Nv_coeff=0.0,

        # wind
        Va=4.0,
        betaA=np.pi / 2.0,

        # sail aerodynamic centers
        x_ce_m=0.26 / 2.0,
        y_ce_m=0.0,

        x_ce_t=(1.3 / 2.0) + (0.26 / 2.0),
        y_ce_t=0.0,
    )

    # ============================================================
    # 4) Solve trim condition
    # ============================================================

    x_trim, u_trim, sol = solve_trim(
        u0_des=2.5,
        params=params,
        sail_model=sail_model,
    )

    print("\n=== TRIM RESULTS ===")
    print("x_trim =", x_trim)
    print("u_trim =", u_trim)
    print("delta_w trim (deg) =", np.rad2deg(u_trim[1]))
    print("====================\n")

    # ============================================================
    # 5) Linearize nonlinear model
    # ============================================================

    A, B, f0 = linearize(
        x_trim,
        u_trim,
        params,
        sail_model,
    )

    print("\nA matrix:\n", A)
    print("\nB matrix:\n", B)

    # ============================================================
    # 6) LQR controller
    # ============================================================

    Q = np.diag([10.0, 50.0, 100.0])
    R = np.diag([1.0, 5.0])

    K, P, poles = lqr(A, B, Q, R)

    print("\n=== LQR RESULTS ===")
    print("Closed-loop poles:")
    print(poles)

    print("\nK gain matrix:")
    print(K)
    print("===================\n")

    # ============================================================
    # 7) Heading controller setup
    # ============================================================

    psi0 = np.deg2rad(0.0)

    # desired heading command
    psi_cmd = np.deg2rad(30.0)

    # initial perturbation from trim
    x0_3dof = x_trim + np.array([
        0.5,
        0.2,
        np.deg2rad(1.0)
    ])

    heading_pid = {
        "Kp": 0.8,
        "Ki": 0.0,
        "Kd": 0.15,
        "r_max": np.deg2rad(20.0),
    }

    # ============================================================
    # 8) Run nonlinear heading simulation
    # ============================================================

    t, x_hist = run_heading_sim(
        x0_3dof=x0_3dof,
        psi0=psi0,
        psi_cmd=psi_cmd,
        t_end=40.0,
        params=params,
        sail_model=sail_model,
        x_trim=x_trim,
        u_trim=u_trim,
        K=K,
        heading_pid=heading_pid,
        dt=0.01,
    )

    # ============================================================
    # 9) Save simulation results
    # ============================================================

    out_dir = Path(__file__).resolve().parent

    csv_path = out_dir / "closed_loop_heading.csv"

    pd.DataFrame({
        "t": t,
        "u": x_hist[:, 0],
        "v": x_hist[:, 1],
        "r_rad_s": x_hist[:, 2],
        "r_deg_s": np.rad2deg(x_hist[:, 2]),
        "psi_rad": x_hist[:, 3],
        "psi_deg": np.rad2deg(x_hist[:, 3]),
    }).to_csv(csv_path, index=False)

    print("Saved simulation CSV:")
    print(csv_path)

    # ============================================================
    # 10) Plot states
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(t, x_hist[:, 0], label="u (m/s)")
    plt.plot(t, x_hist[:, 1], label="v (m/s)")
    plt.plot(t, np.rad2deg(x_hist[:, 2]), label="r (deg/s)")
    plt.plot(t, np.rad2deg(x_hist[:, 3]), label="psi (deg)")

    plt.xlabel("Time (s)")
    plt.ylabel("States")

    plt.title("Closed-Loop Response with Heading Outer Loop + LQR Inner Loop")

    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
