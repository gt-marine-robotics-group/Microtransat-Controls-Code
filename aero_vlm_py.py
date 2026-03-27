# aero_vlm_py.py
import numpy as np


def main_sail_vlm(b_in, alpha_deg, U_inf=4.0, rho=1.225, AR=2.5, N=20):
   
    alpha = np.deg2rad(alpha_deg)
    eps = 1e-9


    c = b_in / AR
    b = b_in / 2.0
    S = b * c
    mac = b / AR

    # taper/sweep model (same)
    crct = (S / b)
    cr = crct / 1.5
    ct = cr * 0.5
    deltac = cr - ct
    sweep = np.arctan(deltac / (2.0 * b))
    leadsweep = sweep
    trailsweep = sweep

    # wing discretization (same)
    ywing = np.linspace(0.0, b, N + 1)
    ydiff = np.diff(ywing)
    ym = ywing[:-1] + 0.5 * ydiff
    y1n = ywing[:-1]
    y2n = ywing[1:]

    cvec0 = cr - ym * np.tan(leadsweep) - ym * np.tan(trailsweep)
    cvec1 = cr - y1n * np.tan(leadsweep) - y1n * np.tan(trailsweep)
    cvec2 = cr - y2n * np.tan(leadsweep) - y2n * np.tan(trailsweep)

    xm = cvec0 * (3.0 / 4.0) + np.tan(leadsweep) * ym
    x1n = cvec1 / 4.0 + ywing[:-1] * np.tan(leadsweep)
    x2n = cvec2 / 4.0 + ywing[1:] * np.tan(leadsweep)

    # mirror (port wing) (same)
    xmp = xm.copy()
    ymp = ym.copy()
    x1np = x2n.copy()
    x2np = x1n.copy()
    y1np = (-y2n).copy()
    y2np = (-y1n).copy()

    # build downwash matrices 
    wstar = np.zeros((N, N))
    wport = np.zeros((N, N))

    for m in range(N):
        for n in range(N):
            # starboard contribution
            denom = (xm[m] - x1n[n]) * (ym[m] - y2n[n]) - (xm[m] - x2n[n]) * (ym[m] - y1n[n])
            a1 = 1.0 / (denom + eps)

            a2 = ((x2n[n] - x1n[n]) * (xm[m] - x1n[n]) + (y2n[n] - y1n[n]) * (ym[m] - y1n[n])) / (
                np.sqrt((xm[m] - x1n[n]) ** 2 + (ym[m] - y1n[n]) ** 2) + eps
            )
            a3 = ((x2n[n] - x1n[n]) * (xm[m] - x2n[n]) + (y2n[n] - y1n[n]) * (ym[m] - y2n[n])) / (
                np.sqrt((xm[m] - x2n[n]) ** 2 + (ym[m] - y2n[n]) ** 2) + eps
            )
            a43 = a1 * (a2 - a3)

            b43 = (1.0 / (y1n[n] - ym[m] + eps)) * (
                1.0 + (xm[m] - x1n[n]) / (np.sqrt((xm[m] - x1n[n]) ** 2 + (ym[m] - y1n[n]) ** 2) + eps)
            )
            c43 = (1.0 / (y2n[n] - ym[m] + eps)) * (
                1.0 + (xm[m] - x2n[n]) / (np.sqrt((xm[m] - x2n[n]) ** 2 + (ym[m] - y2n[n]) ** 2) + eps)
            )
            wstar[m, n] = a43 + b43 - c43

            # port contribution
            denom_p = (xmp[m] - x1np[n]) * (ymp[m] - y2np[n]) - (xmp[m] - x2np[n]) * (ymp[m] - y1np[n])
            a1p = 1.0 / (denom_p + eps)

            a2p = ((x2np[n] - x1np[n]) * (xmp[m] - x1np[n]) + (y2np[n] - y1np[n]) * (ymp[m] - y1np[n])) / (
                np.sqrt((xmp[m] - x1np[n]) ** 2 + (ymp[m] - y1np[n]) ** 2) + eps
            )
            a3p = ((x2np[n] - x1np[n]) * (xmp[m] - x2np[n]) + (y2np[n] - y1np[n]) * (ymp[m] - y2np[n])) / (
                np.sqrt((xmp[m] - x2np[n]) ** 2 + (ymp[m] - y2np[n]) ** 2) + eps
            )
            a43p = a1p * (a2p - a3p)

            b43p = (1.0 / (y1np[n] - ymp[m] + eps)) * (
                1.0 + (xmp[m] - x1np[n]) / (np.sqrt((xmp[m] - x1np[n]) ** 2 + (ymp[m] - y1np[n]) ** 2) + eps)
            )
            c43p = (1.0 / (y2np[n] - ymp[m] + eps)) * (
                1.0 + (xmp[m] - x2np[n]) / (np.sqrt((xmp[m] - x2np[n]) ** 2 + (ymp[m] - y2np[n]) ** 2) + eps)
            )
            wport[m, n] = a43p + b43p - c43p

    w = wstar + wport

    freeb = -4.0 * np.pi * U_inf * np.sin(alpha) * np.ones((N, 1))
    gamma = np.linalg.solve(w, freeb).flatten()

    deltay = b / N
    L = rho * U_inf * np.sum(gamma) * deltay

    q_inf = 0.5 * rho * (U_inf ** 2)
    CL = L / (q_inf * S)

    alpha_i = CL / (np.pi * AR * 2.0)
    Di = alpha_i * np.sum(gamma) * deltay * rho * U_inf
    CDi = Di / (q_inf * S)

    # parasite estimation 
    Sref = S
    Sw = 2.0 * Sref
    Q_inf = rho * (U_inf ** 2)
    mu = 1.789e-5
    FF = 1.0 + 2.0 * 0.24 + 60.0 * (0.24 ** 4)

    Re = (rho * U_inf * mac) / mu
    Cf = 1.328 / np.sqrt(Re)
    Do = Q_inf * Cf * Sw * FF
    CDo = Do / (Q_inf * Sref)

    D = Di + Do
    CD = CDi + CDo
    CLCD = CL / CD if CD != 0 else np.inf

    return {
        "CL": CL * 2.0,
        "CD": CD * 4.0,
        "CM": 0.0,
        "L":  L  * 2.0,
        "D":  D  * 4.0,
        "b":  b  * 2.0,
        "mac": mac,
        "AR": AR * 2.0,
        "S":  S,
        "CLCD": CLCD,
        "ywing": ywing,    
        "cr": cr,
        "ct": ct,
        "sweep": sweep,
    }


def back_sail_vlm(b_in, alpha_deg, U_inf=4.0, rho=1.225, AR=2.5, N=20):
    
    alpha = np.deg2rad(alpha_deg)
    eps = 1e-9

    b = b_in / 2.0
    btot = b * 2.0
    mac = b / 2.5
    S = (b ** 2) / AR

   
    sweep = alpha

    ywing = np.linspace(0.0, b, N + 1)
    ydiff = np.diff(ywing)
    ym = ywing[:-1] + 0.5 * ydiff

    xm = mac * (3.0 / 4.0) + np.tan(sweep) * ym
    x1n = mac / 4.0 + ywing[:-1] * np.tan(sweep)
    x2n = mac / 4.0 + ywing[1:] * np.tan(sweep)
    y1n = ywing[:-1]
    y2n = ywing[1:]

    # mirror
    xmp = xm.copy()
    ymp = ym.copy()
    x1np = x2n.copy()
    x2np = x1n.copy()
    y1np = (-y2n).copy()
    y2np = (-y1n).copy()

    wstar = np.zeros((N, N))
    wport = np.zeros((N, N))

    for m in range(N):
        for n in range(N):
            # starboard
            denom = (xm[m] - x1n[n]) * (ym[m] - y2n[n]) - (xm[m] - x2n[n]) * (ym[m] - y1n[n])
            a1 = 1.0 / (denom + eps)

            a2 = ((x2n[n] - x1n[n]) * (xm[m] - x1n[n]) + (y2n[n] - y1n[n]) * (ym[m] - y1n[n])) / (
                np.sqrt((xm[m] - x1n[n]) ** 2 + (ym[m] - y1n[n]) ** 2) + eps
            )
            a3 = ((x2n[n] - x1n[n]) * (xm[m] - x2n[n]) + (y2n[n] - y1n[n]) * (ym[m] - y2n[n])) / (
                np.sqrt((xm[m] - x2n[n]) ** 2 + (ym[m] - y2n[n]) ** 2) + eps
            )
            a43 = a1 * (a2 - a3)

            b43 = (1.0 / (y1n[n] - ym[m] + eps)) * (
                1.0 + (xm[m] - x1n[n]) / (np.sqrt((xm[m] - x1n[n]) ** 2 + (ym[m] - y1n[n]) ** 2) + eps)
            )
            c43 = (1.0 / (y2n[n] - ym[m] + eps)) * (
                1.0 + (xm[m] - x2n[n]) / (np.sqrt((xm[m] - x2n[n]) ** 2 + (ym[m] - y2n[n]) ** 2) + eps)
            )
            wstar[m, n] = a43 + b43 - c43

            # port
            denom_p = (xmp[m] - x1np[n]) * (ymp[m] - y2np[n]) - (xmp[m] - x2np[n]) * (ymp[m] - y1np[n])
            a1p = 1.0 / (denom_p + eps)

            a2p = ((x2np[n] - x1np[n]) * (xmp[m] - x1np[n]) + (y2np[n] - y1np[n]) * (ymp[m] - y1np[n])) / (
                np.sqrt((xmp[m] - x1np[n]) ** 2 + (ymp[m] - y1np[n]) ** 2) + eps
            )
            a3p = ((x2np[n] - x1np[n]) * (xmp[m] - x2np[n]) + (y2np[n] - y1np[n]) * (ymp[m] - y2np[n])) / (
                np.sqrt((xmp[m] - x2np[n]) ** 2 + (ymp[m] - y2np[n]) ** 2) + eps
            )
            a43p = a1p * (a2p - a3p)

            b43p = (1.0 / (y1np[n] - ymp[m] + eps)) * (
                1.0 + (xmp[m] - x1np[n]) / (np.sqrt((xmp[m] - x1np[n]) ** 2 + (ymp[m] - y1np[n]) ** 2) + eps)
            )
            c43p = (1.0 / (y2np[n] - ymp[m] + eps)) * (
                1.0 + (xmp[m] - x2np[n]) / (np.sqrt((xmp[m] - x2np[n]) ** 2 + (ymp[m] - y2np[n]) ** 2) + eps)
            )
            wport[m, n] = a43p + b43p - c43p

    w = wstar + wport

    freeb = -4.0 * np.pi * U_inf * np.sin(alpha) * np.ones((N, 1))
    gamma = np.linalg.solve(w, freeb).flatten()

    deltay = b / N
    L = rho * U_inf * np.sum(gamma) * deltay

    q_inf = 0.5 * rho * (U_inf ** 2)
    CL = L / (q_inf * S)

    alpha_i = CL / (np.pi * AR * 2.0)
    Di = alpha_i * np.sum(gamma) * deltay * rho * U_inf
    CDi = Di / (q_inf * S)

    # parasite estimation 
    Sref = S
    Sw = 2.0 * Sref
    Q_inf = rho * (U_inf ** 2)
    mu = 1.789e-5
    FF = 1.0 + 2.0 * 0.24 + 60.0 * (0.24 ** 4)

    Re = (rho * U_inf * mac) / mu
    Cf = 1.328 / np.sqrt(Re)
    Do = Q_inf * Cf * Sw * FF
    CDo = Do / (Q_inf * Sref)

    D = Di + Do
    CD = CDi + CDo
    CLCD = CL / CD if CD != 0 else np.inf

    return {
        "CL": CL,
        "CD": CD,
        "CM": 0.0,
        "L": L,
        "D": D,
        "b": btot,
        "mac": mac,
        "AR": AR * 2.0,
        "S": S,
        "CLCD": CLCD,
    }