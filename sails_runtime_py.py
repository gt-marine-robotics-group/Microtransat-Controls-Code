# sails_runtime.py
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import interp1d

@dataclass
class TableSailModel:
    # alpha grid (deg) and L,D values on that grid
    alpha_deg: np.ndarray
    L_main: np.ndarray
    D_main: np.ndarray
    L_back: np.ndarray
    D_back: np.ndarray

    def __post_init__(self):
        self._Lm = interp1d(self.alpha_deg, self.L_main, kind="linear",
                            fill_value="extrapolate", bounds_error=False)
        self._Dm = interp1d(self.alpha_deg, self.D_main, kind="linear",
                            fill_value="extrapolate", bounds_error=False)
        self._Lt = interp1d(self.alpha_deg, self.L_back, kind="linear",
                            fill_value="extrapolate", bounds_error=False)
        self._Dt = interp1d(self.alpha_deg, self.D_back, kind="linear",
                            fill_value="extrapolate", bounds_error=False)

    def main(self, alpha_deg: float):
        return {"L": float(self._Lm(alpha_deg)), "D": float(self._Dm(alpha_deg))}

    def back(self, alpha_deg: float):
        return {"L": float(self._Lt(alpha_deg)), "D": float(self._Dt(alpha_deg))}