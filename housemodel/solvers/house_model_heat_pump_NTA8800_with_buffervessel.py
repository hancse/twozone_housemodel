"""
house model base on 2R2C model with a buffervessel and a radiator
"""

from scipy.integrate import solve_ivp       # ODE solver
import numpy as np                       # linear algebra
# from housemodel.tools.PIDsim import PID
from housemodel.controls.ivPID.PID import PID
from housemodel.sourcesink.heatpumps.Heatpump_HM import Heatpump_NTA
from housemodel.controls.heating_curves import hyst, outdoor_reset
from housemodel.sourcesink.heatpumps.NTA8800_Q.HPQ9 import calc_WP_general
import housemodel.tools.ReturnTemperature as Tr
q_o=1   # [kW] Desing (peak) heat demand rate
Ts_o=50 # [°C] Design radiator supply temperature
Tr_o=30 # [°C] Design radiator return temperature
Ti_o=20 # [°C] Design indoor (set) temperature
n=1.3   # [-]  Emprical radiator constant

def model_radiator_m(t, x, cap_mat_inv, cond_mat, q_vector,
                     control_interval):
    """model function for scipy.integrate.odeint.

    Args:
        t:           (array):   variable array dependent on time with the vairable Air temperature, Wall temperature Radiator
        x:           (float):
        cap_mat_inv: (float):  diagonal heat capacity matrix
        cond_mat:    (float):  symmetric conductance matrix
        q_vector:    (float):  external heat sources and sinks
        SP_T:        (float): thermostat setpoints

    Returns:
        (list): vector elements of dx/dt
    """
    # States :
    # Tair = x[0]

    # Parameters :
    index = int(t/control_interval)

    # Equations :
    local_q_vector = np.zeros((2,1))
    local_q_vector[0,0] = q_vector[0,index]
    local_q_vector[1,0] = q_vector[1,index]
    #local_q_vector[2,0] = q_vector[2,index]

    # Conversion of 1D array to a 2D array
    # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
    x = np.array(x)[np.newaxis]

    dTdt = (-cond_mat @ x.T) + local_q_vector
    dTdt = np.dot(cap_mat_inv, dTdt)

    return dTdt.flatten().tolist()

def model_stratified_buffervessel(t, x, U, As, Aq, Tamb, Tsupply, Treturn, cpwater, lamb, mdots, mdotd, mass_water, z):
    """model function for scipy.integrate.odeint.

    :param x:            (array):   variable array dependent on time with the vairable Air temperature, Wall temperature Return water temperature and buffervessel temperature
    :param t:            (float):
    :param Pin:          (float):  Power input in [W]
    :param U:            (float):
    :param A:            (float):  Area of
    :param T_amb:        (float):
    :param rho:          (float):
    :param volume:       (float):
    :param cp: (float):  Thermal resistance from indoor air to outdoor air [K/W]

    x,t: ode input function func : callable(x, t, ...) or callable(t, x, ...)
    Computes the derivative of y at t.
    If the signature is ``callable(t, y, ...)``, then the argument tfirst` must be set ``True``.
    """
    #Water supply
    mdote = mdots - mdotd

    if mdote > 0:
        deltaPlus = 1
    else:
        deltaPlus = 0

    if mdote < 0:
        deltaMinus = 1
    else:
        deltaMinus = 0

    dT1 = ((mdots * cpwater * (Tsupply - x[0])) + (mdote *cpwater*(x[0] - x[1]) * deltaMinus) - (U * As * (x[0]- Tamb)) + ((Aq * lamb) / z) * (x[0] - x[1])) / (mass_water*cpwater)
    dT2 = ((mdote *cpwater*(x[0] - x[1]) * deltaPlus) + (mdote *cpwater*(x[1] - x[2]) * deltaMinus) - (U * As * (x[1]- Tamb)) + ((Aq * lamb) / z) * (x[0] + x[2] - (2*x[1]))) / (mass_water*cpwater)
    dT3 = ((mdote *cpwater*(x[1] - x[2]) * deltaPlus) + (mdote *cpwater*(x[2] - x[3]) * deltaMinus) - (U * As * (x[2]- Tamb)) + ((Aq * lamb) / z) * (x[1] + x[3] - (2*x[2]))) / (mass_water*cpwater)
    dT4 = ((mdote *cpwater*(x[2] - x[3]) * deltaPlus) + (mdote *cpwater*(x[3] - x[4]) * deltaMinus) - (U * As * (x[3]- Tamb)) + ((Aq * lamb) / z) * (x[2] + x[4] - (2*x[3]))) / (mass_water*cpwater)
    dT5 = ((mdote *cpwater*(x[3] - x[4]) * deltaPlus) + (mdote *cpwater*(x[4] - x[5]) * deltaMinus) - (U * As * (x[4]- Tamb)) + ((Aq * lamb) / z) * (x[3] + x[5] - (2*x[4]))) / (mass_water*cpwater)
    dT6 = ((mdote *cpwater*(x[4] - x[5]) * deltaPlus) + (mdote *cpwater*(x[5] - x[6]) * deltaMinus) - (U * As * (x[5]- Tamb)) + ((Aq * lamb) / z) * (x[4] + x[6] - (2*x[5]))) / (mass_water*cpwater)
    dT7 = ((mdote *cpwater*(x[5] - x[6]) * deltaPlus) + (mdote *cpwater*(x[6] - x[7]) * deltaMinus) - (U * As * (x[6]- Tamb)) + ((Aq * lamb) / z) * (x[5] + x[7] - (2*x[6]))) / (mass_water*cpwater)
    dT8 = ((mdotd * cpwater * (Treturn - x[7])) + (mdote * cpwater * (x[6] - x[7]) * deltaPlus) - (U * As * (x[7] - Tamb)) + ((Aq * lamb) / z) * (x[6] - x[7])) / (mass_water*cpwater)

    return [dT1, dT2, dT3, dT4, dT5, dT6, dT7, dT8]

def house_radiator_m(cap_mat_inv, cond_mat, q_vector,
                     SP_T, time_sim, control_interval, control_parameters, T_outdoor_sim):
    """Compute air and wall temperature inside the house.

    Args:
        cap_mat:    (float):  diagonal heat capacity matrix
        cond_mat:   (float):  symmetric conductance matrix
        q_vector:   (float):  external heat sources and sinks
        SP_T:       (array):  Setpoint temperature from thermostat.
        time_sim:   (array)  :  simulation time

    Returns:
        tuple :  (array) containing Tair, Twall, Tradiator and evaluation time:

    Note:
        - Tair (float):   air temperature inside the house in degree C.
        - Twall (float):  wall temperature inside the house in degree C

        - Qinst ?	  (array):  instant heat from heat source such as HP or boiler [W].
    """
    #initial values for solve_ivp
    Tair0 = 15
    Twall0 = 20
    TBuffervessel0 = 80

    y0 = [Tair0, Twall0]
    y0buffervessel = [TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0]

    t = time_sim           # Define Simulation time with sampling time
    Tair = np.ones(len(t)) * Tair0
    Twall = np.ones(len(t)) * Twall0
    Treturn = np.ones(len(t)) * Twall0
    Power = np.ones(len(t)) * Twall0
    TBuffervessel1 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel2 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel3 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel4 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel5 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel6 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel7 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel8 = np.ones(len(t)) * TBuffervessel0

    # Controller initialization
    # heatingPID = PID(Kp=5000, Ki=0, Kd=0, beta=1, MVrange=(0, 12000), DirectAction=False)
    # heating = 0
    kp = control_parameters[0]
    ki = control_parameters[1]
    kd = control_parameters[2]

    pid = PID(kp, ki, kd, t[0])

    pid.SetPoint=17.0
    pid.setSampleTime(0)
    pid.setBounds(0, 5000)
    pid.setWindup(5000/control_interval)

    # Heat pump initialization
    nta = Heatpump_NTA()
    nta.Pmax = 8
    nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])

    nta.c_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_COP_val, order=1)

    nta.p_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_Pmax_val, order=1)

    water_temp = np.zeros_like(T_outdoor_sim)
    cop_hp = np.zeros_like(T_outdoor_sim)

    # define hysteresis object for heat pump
    hp_hyst = hyst(dead_band=0.5, state=True)

    inputs = (cap_mat_inv, cond_mat, q_vector, control_interval)

    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g SP_T[8760] is called.
    # Therefore set "first_step" equal or smaller than the spacing of "t".
    # https://github.com/scipy/scipy/issues/9198
    for i in range(len(t)-1):

        # here comes the "arduino style" controller
        pid.SetPoint = SP_T[i]
        pid.update(Tair[i], t[i])
        Qinst = pid.output

        # Simple PID controller
        # Qinst = (SP_T[i] - Tair[i]) * 5000
        # Qinst = np.clip(Qinst, 0, 5000)
        # q_vector[2, i] = Qinst

        # Velocity PID controller (not working properly)
        # heating  = heatingPID.update(t[i], SP_T[i], Tair[i], heating)
        # print(f"{heating}")
        # heating  = heatingPID.update(t[i], SP_T[i], Tair[i], heating)
        # print(f"{heating}")
        # q_vector[2, i] = heating

        # Heat pump NTA800
        # p_hp = 0
        # determine new setting for COP and heat pump power
        water_temp[i] = outdoor_reset(T_outdoor_sim[i], 0.7, 20)
        cop_hp[i], p_hp = nta.update(T_outdoor_sim[i], water_temp[i])

        # incorporate hysteresis to control
        p_hp = hp_hyst.update(Tair[i], SP_T[i], p_hp)
        if Qinst<2500:
            Qinst = 0

        Tr_GMTD = Tr.Tr_GMTD(Qinst/1000, 80, 20, 5, 80, 60, 20, 1.33)



        # update q_vector
        q_vector[0, i] = q_vector[0, i] + Qinst

        ts = [t[i], t[i+1]]
        result = solve_ivp(model_radiator_m, ts, y0,
                        method='RK45', args=inputs,
                        first_step=control_interval)
        toplevel = TBuffervessel1[i]
        mdots = np.clip((80 - toplevel)*0.001, 0, 0.05)
        mdotd = np.clip(Qinst / ((TBuffervessel1[i]-Tr_GMTD)*4180), 0, 0.05)
        inputs_buffervessel = (0.12, 0.196, 0.196, 10, 80, Tr_GMTD, 4190, 0.644, mdots, mdotd, 150 / 8, 1 / 8)
        result_buffervessel = solve_ivp(model_stratified_buffervessel, ts, y0buffervessel, method='RK45', args=inputs_buffervessel,
                        first_step=control_interval)


        Tair[i+1] = result.y[0, -1]
        Twall[i+1] = result.y[1, -1]
        Treturn[i] = Tr_GMTD
        Power[i] = Qinst
        TBuffervessel1[i+1] = result_buffervessel.y[0, -1]
        TBuffervessel2[i+1] = result_buffervessel.y[1, -1]
        TBuffervessel3[i+1] = result_buffervessel.y[2, -1]
        TBuffervessel4[i+1] = result_buffervessel.y[3, -1]
        TBuffervessel5[i+1] = result_buffervessel.y[4, -1]
        TBuffervessel6[i+1] = result_buffervessel.y[5, -1]
        TBuffervessel7[i+1] = result_buffervessel.y[6, -1]
        TBuffervessel8[i+1] = result_buffervessel.y[7, -1]


        y0 = result.y[:, -1]
        y0buffervessel = result_buffervessel.y[:, -1]


    return t, Tair, Twall, Treturn, Power, water_temp, cop_hp, TBuffervessel1, TBuffervessel2, TBuffervessel3, TBuffervessel4, TBuffervessel5, TBuffervessel6, TBuffervessel7, TBuffervessel8

