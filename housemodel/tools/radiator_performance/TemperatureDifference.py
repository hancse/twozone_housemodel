## Functions to return radiator excess temperature
#   prepared by Hakan İbrahim Tol, PhD

#   INPUTS
#   Ts  : [°C] Radiator supply temperature
#   Tr  : [°C] Radiator return temperature
#   Ti  : [°C] Indoor (set) temperature

# import math
import numpy as np


def AMTD(Ts, Tr, Ti):
    """returns Arithmetic Mean Temperature Difference (AMTD)"""
    Ts = np.asarray(Ts)
    Tr = np.asarray(Tr)
    Ti = np.asarray(Ti)
    scalar_input = False
    if Ts.ndim == 0:
        Ts = Ts[np.newaxis]  # convert scalar to 1d array
        Tr = Tr[np.newaxis]  # convert scalar to 1d array
        Ti = Ti[np.newaxis]  # convert scalar to 1d array
        scalar_input = True

    result = (Ts+Tr-2*Ti) / 2
    if scalar_input:
        return result.item(0)
    return result


def GMTD(Ts, Tr, Ti):
    """returns Geometric Mean Temperature Difference (GMTD)"""
    Ts = np.asarray(Ts)
    Tr = np.asarray(Tr)
    Ti = np.asarray(Ti)
    scalar_input = False
    if Ts.ndim == 0:
        Ts = Ts[np.newaxis]  # convert scalar to 1d array
        Tr = Tr[np.newaxis]  # convert scalar to 1d array
        Ti = Ti[np.newaxis]  # convert scalar to 1d array
        scalar_input = True

    result = np.sqrt(Ts-Ti) * np.sqrt(Tr-Ti)
    if scalar_input:
        return result.item(0)
    return result


def LMTD(Ts, Tr, Ti):
    """returns Logarithmic Mean Temperature Difference (LMTD)"""
    Ts = np.asarray(Ts)
    Tr = np.asarray(Tr)
    Ti = np.asarray(Ti)
    scalar_input = False
    if Ts.ndim == 0:
        Ts = Ts[np.newaxis]  # convert scalar to 1d array
        Tr = Tr[np.newaxis]  # convert scalar to 1d array
        Ti = Ti[np.newaxis]  # convert scalar to 1d array
        scalar_input = True

    result = (Ts-Tr) / np.log((Ts-Ti)/(Tr-Ti))
    if scalar_input:
        return result.item(0)
    return result
