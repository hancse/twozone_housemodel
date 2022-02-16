# Calculates the return temperature from the radiator unit, as based on, respectively:
#   - Arithmetic Mean Temperature Difference (AMTD)
#   - Geometric Mean Temperature Difference (GMTD)
#   - Logarithmic Mean Temperature Difference (LMTD)
# Ref: Phetteplace - Optimal Design of Piping Systems for District Heating
# prepared by Hakan ibrahim Tol, PhD

# INPUTS
# q     : [kW] Heat demand at actual condition
# Ts    : [°C] Radiator supply temperature at actual condition
# Ti    : [°C] Indoor (set) temperature at actual condition

# q_o   : [kW] Heat demand at design condition
# Ts_o  : [°C] Radiator supply temperature at design condition
# Tr_o  : [°C] Radiator return temperature at design condition
# Ti_o  : [°C] Indoor (set) temperature at design condition

# n     : [-]  Emprical radiator constant

from housemodel.tools import TemperatureDifference as TD
import math

def Tr_AMTD(q,Ts,Ti,q_o,Ts_o,Tr_o,Ti_o,n):
# Calculates the return temperature from the radiator unit - based on AMTD

    AMTD_o=TD.AMTD(Ts_o,Tr_o,Ti_o)

    Tr=2*(Ti+(AMTD_o*(q/q_o)**(1/n)))-Ts

    # Checking Error
    AF=(Tr-Ti)/(Ts-Ti)

    #if AF>=0.5:
    #    print("Warning: Approach factor is ",AF," - Error less than 0.04")
    #else:
    #    print("Warning: Approach factor is ",AF," - Error larger than 0.04")

    if Tr>=Ts or Tr<=20:
        return math.nan
    else:
        return Tr

def Tr_GMTD(q,Ts,Ti,q_o,Ts_o,Tr_o,Ti_o,n):
# Calculates the return temperature from the radiator unit - based on GMTD

    GMTD_o=TD.GMTD(Ts_o,Tr_o,Ti_o)

    Tr=Ti+((Ts-Ti)**(-1)*GMTD_o**2*(q/q_o)**(2/n));

    # Checking Error
    AF=(Tr-Ti)/(Ts-Ti)

    #if AF>=0.33:
    #    print("Warning: Approach factor is ",AF," - Error less than 0.05")
    #else:
    #    print("Warning: Approach factor is ",AF," - Error larger than 0.05")

    if Tr>=Ts:
        return math.nan
    else:
        return Tr

def Tr_LMTD(q,Ts,Ti,q_o,Ts_o,Tr_o,Ti_o,n):
    # Calculates the return temperature from the radiator unit - based on LMTD


    LMTD_o=TD.LMTD(Ts_o,Tr_o,Ti_o)

    # Iteration for the implicit LMTD method
    fTol=0.001                                       # Iteration tolerance
    error=10                                        # Iteration error

    Tr_it1=Tr_GMTD(q,Ts,Ti,q_o,Ts_o,Tr_o,Ti_o,n)    # Initial iteration value - based on GMTD

    while error>fTol:
        Tr_it2 = Ti + ((Ts - Ti) / math.exp((q / q_o) ** (-1 / n) * (Ts - Tr_it1) / LMTD_o))
        error = abs(Tr_it2 - Tr_it1)
        Tr_it1 = Tr_it2

    if Tr_it2>=Ts:
        return math.nan
    else:
        return Tr_it2
