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
from sim_closed_loop_py import run_sim


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
        m = main_sail_vlm(b_m, float(a))   # dict
        t = back_sail_vlm(b_t, float(a))   # dict

        Lm = float(m["L"]);  Dm = float(m["D"])
        Lt = float(t["L"]);  Dt = float(t["D"])

        Lm_list.append(Lm); Dm_list.append(Dm)
        Lt_list.append(Lt); Dt_list.append(Dt)

        rows.append({
            "alpha_deg": float(a),

            # main
            "ARm": float(m.get("AR", np.nan)),
            "Sm":  float(m.get("S",  np.nan)),
            "bm":  float(m.get("b",  np.nan)),
            "macm":float(m.get("mac",np.nan)),
            "CLm": float(m.get("CL", np.nan)),
            "CDm": float(m.get("CD", np.nan)),
            "CMm": float(m.get("CM", np.nan)),
            "Liftm": Lm,
            "Dragm": Dm,
            "CLCDm": float(m.get("CLCD", np.nan)),
            "cr": float(m.get("cr", np.nan)),
            "ct": float(m.get("ct", np.nan)),
            "sweep": float(m.get("sweep", np.nan)),

            # back
            "ARt": float(t.get("AR", np.nan)),
            "St":  float(t.get("S",  np.nan)),
            "bt":  float(t.get("b",  np.nan)),
            "mact":float(t.get("mac",np.nan)),
            "CLt": float(t.get("CL", np.nan)),
            "CDt": float(t.get("CD", np.nan)),
            "CMt": float(t.get("CM", np.nan)),
            "Liftt": Lt,
            "Dragt": Dt,
            "CLCDt": float(t.get("CLCD", np.nan)),
        })

    # Convert to clean numeric arrays
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

    # 1) Build aero tables and save Excel next to this file
    alphas_deg, Lm, Dm, Lt, Dt, df, out_path = build_tables(
        b_m=1.3,
        alpha_min=0.0,
        alpha_max=12.0,
        n_alpha=13,
        outfile_name="vlm_tables.xlsx",  
    )

    # 2) Runtime sail model from tables
    sail_model = TableSailModel(
        alpha_deg=np.asarray(alphas_deg, dtype=float),
        L_main=np.asarray(Lm, dtype=float),
        D_main=np.asarray(Dm, dtype=float),
        L_back=np.asarray(Lt, dtype=float),
        D_back=np.asarray(Dt, dtype=float),
    )

    # 3) Parameters
    params = Params(
        m=10.0,
        Iz=25.0,
        rho_w=1025.0,
        rho_a=1.225,
        S_r=0.1524 * 0.1524,
        l_r=0.5,
        Sf=0.9 * 1.0 * (0.2 + 2.0 * 0.25),
        CDx=0.6 / 1025.0,
        Yv_coeff=40.0,
        C_b=0.5 * 1025.0 * (1.143**5) * 0.01,
        Nv_coeff=0.0,
        Va=4.0,
        betaA=np.pi / 2.0,
        x_ce_m=0.26 / 2.0,
        y_ce_m=0.0,
        x_ce_t=(1.3 / 2.0) + (0.26 / 2.0),
        y_ce_t=0.0,
        CLalpha_r=5.0,
    )

    # 4) Solve trim
    x_trim, u_trim, sol = solve_trim(u0_des=2.5, params=params, sail_model=sail_model)
    print("trim x:", x_trim, "trim u:", u_trim)
    print("trim delta_w (deg):", np.rad2deg(u_trim[1]))

    # 5) Linearize
    A, B, f0 = linearize(x_trim, u_trim, params, sail_model)

    # 6) LQR
    Q = np.diag([10.0, 50.0, 100.0])
    R = np.diag([1.0, 5.0])
    K, P, poles = lqr(A, B, Q, R)
    print("poles:", poles)
    print("K:\n", K)

    # 7) Simulate nonlinear closed-loop
    x0 = x_trim + np.array([0.5, 0.2, np.deg2rad(1.0)])
    t, x_hist = run_sim(
        x0=x0,
        t_end=20.0,
        params=params,
        sail_model=sail_model,
        x_trim=x_trim,
        u_trim=u_trim,
        K=K,
        dt=0.01
    )

    # 8) Save CSV next to this script
    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "closed_loop.csv"
    pd.DataFrame({"t": t, "u": x_hist[:, 0], "v": x_hist[:, 1], "r": x_hist[:, 2]}).to_csv(csv_path, index=False)
    print("wrote:", csv_path)

    # 9) Plot
    plt.figure(); plt.plot(t, x_hist[:, 0]); plt.xlabel("t (s)"); plt.ylabel("u (m/s)"); plt.grid(True)
    plt.figure(); plt.plot(t, x_hist[:, 1]); plt.xlabel("t (s)"); plt.ylabel("v (m/s)"); plt.grid(True)
    plt.figure(); plt.plot(t, x_hist[:, 2]); plt.xlabel("t (s)"); plt.ylabel("r (rad/s)"); plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()