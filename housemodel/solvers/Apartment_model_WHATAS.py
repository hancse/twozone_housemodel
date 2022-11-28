"""
house model base on 2R2C model with a buffervessel and a radiator
"""

from scipy.integrate import solve_ivp       # ODE solver
import numpy as np                          # linear algebra
from housemodel.controls.ivPID.PID import PID
from housemodel.controls.heating_curves import hyst
from housemodel.sourcesink.buffervessels.buffer_vessel import StratifiedBuffer
cp_water = 4190

def relay(d, SP):
    MV = 0
    while True:
        PV = yield MV
        MV_prev = MV
        MV = 1 if PV < SP - d else 0 if PV > SP + d else MV_prev

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
                     SP_T, time_sim, control_interval, control_parameters, T_outdoor_sim, waterflow_sim, interp_func_retour, interp_func_flow):
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
    TBuffervessel0 = 70
    TBuffervessel80 = 40

    y0 = [Tair0, Twall0]
    y0buffervessel = [TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0]

    # Define Simulation time with sampling time and get the containers for the data
    t = time_sim
    Tair = np.ones(len(t)) * Tair0
    Twall = np.ones(len(t)) * Twall0
    Treturn = np.ones(len(t)) * Twall0
    Power = np.ones(len(t)) * Twall0
    Power_buffervessel = np.ones(len(t)) * Twall0
    electrical_power = np.ones(len(t)) * Twall0
    mdots_array = np.ones(len(t)) * Twall0
    TBuffervessel1 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel2 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel3 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel4 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel5 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel6 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel7 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel8 = np.ones(len(t)) * TBuffervessel80

    # Controller initialization for roomt temperature and buffervessel top layer temperature
    kp = control_parameters[0]
    ki = control_parameters[1]
    kd = control_parameters[2]
    max_power = control_parameters[3]

    # Room temp controller
    pid_Room_Temp = PID(kp, ki, kd, t[0])
    pid_Room_Temp.SetPoint=20.0
    pid_Room_Temp.setSampleTime(0)
    pid_Room_Temp.setBounds(0, max_power)
    pid_Room_Temp.setWindup(max_power/control_interval)

    # Room temp controller
    pid_buffer_flow = PID(1, 0, 0, t[0])
    pid_buffer_flow.SetPoint=74.0
    pid_buffer_flow.setSampleTime(0)
    pid_buffer_flow.setBounds(0, 0.717)
    pid_buffer_flow.setWindup(2/control_interval)

    # define hysteresis object for Room temperature and buffervessel top layer temperature
    buffervessel_hyst = hyst(dead_band=3, state=True)
    thermostat = relay(1.5, 71.5)  # create thermostat
    thermostat.send(None)  # initialize thermostat

    thermostat2 = relay(1, 71)  # create thermostat
    thermostat2.send(None)  # initialize thermostat

    thermostat3 = relay(.5, 70.5)  # create thermostat
    thermostat3.send(None)  # initialize thermostat

    #Buffervessel initialization
    sb = StratifiedBuffer(2, 2.5, 8)

    inputs = (cap_mat_inv, cond_mat, q_vector, control_interval)

    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g SP_T[8760] is called.
    # Therefore set "first_step" equal or smaller than the spacing of "t".
    # https://github.com/scipy/scipy/issues/9198
    for i in range(len(t)-1):

        # here comes the "arduino style" controller
        pid_Room_Temp.SetPoint = SP_T[i]
        pid_Room_Temp.update(Tair[i], t[i])
        Qinst = pid_Room_Temp.output

        return_temp = float(interp_func_retour(Qinst/max_power))
        radiator_flow_normalized = np.clip(float(interp_func_flow(Qinst/max_power)), 0, 1)

        #mdotd = Qinst/(cp_water*LMTD)
        mdotd = radiator_flow_normalized * (max_power/(cp_water*(TBuffervessel1[i]-return_temp)))

        # Calculate the supply and demand waterflow for the buffervessel
        toplevel = TBuffervessel1[i]

        # Supply is
        #pid_buffer_flow.setBounds(0, (47110*3)/(cp_water*(TBuffervessel1[i]-TBuffervessel8[i])))
        #pid_buffer_flow.update(toplevel, t[i])
        #mdots = buffervessel_hyst.update(toplevel, 74, pid_buffer_flow.output)
        MV = thermostat.send(toplevel)
        MV2 = thermostat2.send(toplevel)
        MV3 = thermostat3.send(toplevel)
        mdots = (49770) / (cp_water * (TBuffervessel1[i] - TBuffervessel8[i])) * MV
        mdots2 = (49770) / (cp_water * (TBuffervessel1[i] - TBuffervessel8[i])) * MV2
        mdots3 = (49770) / (cp_water * (TBuffervessel1[i] - TBuffervessel8[i])) * MV3
        mdots = mdots + mdots2 + mdots3


        """
        if(toplevel>73):
            mdots = (47110*1)/(cp_water*(TBuffervessel1[i]-TBuffervessel8[i]))*MV
        elif(toplevel>72):
            mdots = (47110 * 2) / (cp_water * (TBuffervessel1[i] - TBuffervessel8[i])) * MV
        elif(toplevel>71):
            mdots = (47110 * 3) / (cp_water * (TBuffervessel1[i] - TBuffervessel8[i])) * MV
        else:mdots
        """
        heating_power_buffervessel = mdots*cp_water*(TBuffervessel1[i]-TBuffervessel8[i])
        tapwater_flow = waterflow_sim[i]/60
        total_flow = mdotd + tapwater_flow
        combined_return_temperature = (mdotd/total_flow)*return_temp + (tapwater_flow/total_flow)*35

        # update q_vector
        q_vector[0, i] = q_vector[0, i] + Qinst

        ts = [t[i], t[i+1]]
        result = solve_ivp(model_radiator_m, ts, y0,
                        method='RK45', args=inputs,
                        first_step=control_interval)

        inputs_buffervessel = (75, combined_return_temperature, mdots, total_flow)
        result_buffervessel = solve_ivp(sb.model_stratified_buffervessel, ts, y0buffervessel, method='RK45', args=inputs_buffervessel,
                        first_step=control_interval)

        # Write data into the containers
        Tair[i+1] = result.y[0, -1]
        Twall[i+1] = result.y[1, -1]
        Treturn[i] = return_temp
        Power[i] = Qinst
        Power_buffervessel[i] = heating_power_buffervessel
        electrical_power[i] = (heating_power_buffervessel/2.78) + (333 * ((1-MV) + (1-MV2) + (1-MV3)))
        TBuffervessel1[i+1] = result_buffervessel.y[0, -1]
        TBuffervessel8[i+1] = result_buffervessel.y[7, -1]


        #Use last result as a initial value for the ext step
        y0 = result.y[:, -1]
        y0buffervessel = result_buffervessel.y[:, -1]

    return t, Tair, Twall, Power, Treturn,  TBuffervessel1, TBuffervessel8, Power_buffervessel, electrical_power

