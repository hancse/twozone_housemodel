## Functions to return radiator excess temperature
#   prepared by Hakan İbrahim Tol, PhD

#   INPUTS
#   Ts  : [°C] Radiator supply temperature
#   Tr  : [°C] Radiator return temperature
#   Ti  : [°C] Indoor (set) temperature

# returns Arithmetic Mean Temperature Difference (AMTD)
def AMTD(Ts,Tr,Ti):
    import math
    return (Ts+Tr-2*Ti)/2

# returns Geometric Mean Temperature Difference (GMTD)
def GMTD(Ts,Tr,Ti):
    import math
    return math.sqrt(Ts-Ti)*math.sqrt(Tr-Ti)

# returns Logarithmic Mean Temperature Difference (LMTD)
def LMTD(Ts,Tr,Ti):
    import math
    return (Ts-Tr)/math.log((Ts-Ti)/(Tr-Ti))
