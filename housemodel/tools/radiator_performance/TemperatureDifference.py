## Functions to return radiator excess temperature
#   prepared by Hakan İbrahim Tol, PhD

#   INPUTS
#   Ts  : [°C] Radiator supply temperature
#   Tr  : [°C] Radiator return temperature
#   Ti  : [°C] Indoor (set) temperature

import math

def AMTD(Ts, Tr, Ti):
    """returns Arithmetic Mean Temperature Difference (AMTD)"""
    return (Ts+Tr-2*Ti) / 2


def GMTD(Ts, Tr, Ti):
    """returns Geometric Mean Temperature Difference (GMTD)"""
    return math.sqrt(Ts-Ti) * math.sqrt(Tr-Ti)


def LMTD(Ts, Tr, Ti):
    """returns Logarithmic Mean Temperature Difference (LMTD)"""
    return (Ts-Tr) / math.log((Ts-Ti)/(Tr-Ti))
