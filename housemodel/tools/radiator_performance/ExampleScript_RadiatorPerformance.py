# Example Script - calculates Radiator Return Temperature
# prepared by Hakan İbrahim Tol, PhD

# INPUTS
q=0.5     # [kW] Actual heat demand rate
Ts=90   # [°C] Actual radiator supply temperature
Ti=20   # [°C] Actual indoor (set) temperature

q_o=1   # [kW] Desing (peak) heat demand rate
Ts_o=90 # [°C] Design radiator supply temperature
Tr_o=70 # [°C] Design radiator return temperature
Ti_o=20 # [°C] Design indoor (set) temperature

n=1.3   # [-]  Emprical radiator constant

import ReturnTemperature as Tr
Tr_AMTD=Tr.Tr_AMTD(q,Ts,Ti,q_o,Ts_o,Tr_o,Ti_o,n)
Tr_GMTD=Tr.Tr_GMTD(q,Ts,Ti,q_o,Ts_o,Tr_o,Ti_o,n)
Tr_LMTD=Tr.Tr_LMTD(q,Ts,Ti,q_o,Ts_o,Tr_o,Ti_o,n)

# Print to Screen as Table
table={'Tr via AMTD': Tr_AMTD, 'Tr via GMTD': Tr_GMTD, 'Tr via LMTD': Tr_LMTD}
for name, value in table.items():
    print(f'{name:10} ==> {value:10f}')
