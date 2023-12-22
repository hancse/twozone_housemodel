"""
house model base on 2R2C model with a buffervessel and a radiator
"""

from scipy.integrate import solve_ivp       # ODE solver
import numpy as np                          # linear algebra
from housemodel.controls.ivPID.PID import PID
from housemodel.controls.heating_curves import hyst
from housemodel.sourcesink.buffervessels.buffer_vessel import StratifiedBuffer
import pandas as pd
from tqdm import tqdm

cp_water = 4190

def relay(d, SP):
    MV = 0
    while True:
        [PV, SP] = yield MV
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
                     SP_T, time_sim, control_interval, control_parameters, T_outdoor_sim, interp_func_retour, interp_func_flow):
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
    TBuffervessel0 = 50
    TBuffervessel80 = 20

    y0 = [Tair0, Twall0]
    y0buffervessel = [TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0]

    # Define Simulation time with sampling time and get the containers for the data
    t = time_sim
    Tair = np.ones(len(t)) * Tair0
    Twall = np.ones(len(t)) * Twall0
    Treturn = np.ones(len(t)) * Twall0
    Buffervessel_setpoint = np.ones(len(t)) * 50
    Power = np.ones(len(t)) * Twall0
    Power_buffervessel = np.ones(len(t)) * Twall0
    electrical_power = np.ones(len(t)) * Twall0
    TBuffervessel1 = np.ones(len(t)) * TBuffervessel0
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

    # buffervessel top level controller
    pid_buffervessel_power_kW = PID(100, 1, 0, t[0])
    pid_buffervessel_power_kW.SetPoint=74.0
    pid_buffervessel_power_kW.setSampleTime(0)
    pid_buffervessel_power_kW.setBounds(0, 100)
    pid_buffervessel_power_kW.setWindup(100/control_interval)

    # define hysteresis object for Room temperature and buffervessel top layer temperature
    #buffervessel_hyst = hyst(dead_band=3, state=True)
    thermostat = relay(1.5, 71.5)  # create thermostat
    thermostat.send(None)  # initialize thermostat

    #Buffervessel initialization
    sb = StratifiedBuffer(2, 2.5, 8)

    inputs = (cap_mat_inv, cond_mat, q_vector, control_interval)

    # Compressor lookup table
    df2 = pd.read_excel(r'Temperature-Power map.xlsx')
    Power_compressor = np.array(df2.loc[:, "Power"])
    condens_temp = np.array(df2.loc[:, "Condensation Temperature"])
    COP = np.array(df2.loc[:, "COP"])
    compressor_data = list(zip(Power_compressor / 1000, condens_temp))
    compressor_data = np.array(compressor_data)

    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g SP_T[8760] is called.
    # Therefore set "first_step" equal or smaller than the spacing of "t".
    # https://github.com/scipy/scipy/issues/9198
    for i in tqdm(range(len(t)-1)):

        # here comes the "arduino style" controller
        pid_Room_Temp.SetPoint = SP_T[i]
        pid_Room_Temp.update(Tair[i], t[i])
        Qinst = pid_Room_Temp.output
        if(Qinst<(max_power/10)):
            Qinst = 0
            return_temp = 40
            radiator_flow_normalized = 0
        else:
            return_temp = np.clip(float(interp_func_retour(Qinst / max_power, TBuffervessel1[i])), 20, 40)
            radiator_flow_normalized = np.clip(float(interp_func_flow(Qinst / max_power, TBuffervessel1[i])), 0, 1)

        mdotd = radiator_flow_normalized * (max_power/(cp_water*(TBuffervessel1[i]-return_temp)))

        # Calculate buffervessel setpoint from outdoor temp
        buffer_setpoint = (-4/3 * T_outdoor_sim[i]) + (56+2/3)

        #-------------------------
        pid_buffervessel_power_kW.SetPoint = buffer_setpoint
        pid_buffervessel_power_kW.update(TBuffervessel1[i], t[i])
        Q_buffervessel = pid_buffervessel_power_kW.output
        rslt_df = df2.loc[df2['Condensation Temperature'] > buffer_setpoint]
        max_cop_frame = df2.loc[rslt_df['COP'].idxmax()]
        condensation_temperature = max_cop_frame['Condensation Temperature']
        compressor_power = max_cop_frame['Power']
        COP = max_cop_frame['COP']

        #print('\nResult dataframe :\n', max_cop_frame['Condensation Temperature'], max_cop_frame['COP'],
        #      max_cop_frame['Power'])
        #desired_point = np.array((Q_buffervessel, buffer_setpoint))
        #distances = np.linalg.norm(compressor_data - desired_point, axis=1)
        #min_index = np.argmin(distances)

        if Q_buffervessel<10:
            mdots=0
            heating_power_buffervessel = 0
            elec_power = 666
        else:
            if (condensation_temperature <= TBuffervessel8[i]):
                mdots = 0
                elec_power = 666
            else:
                mdots = (compressor_power)*2 / (cp_water * (condensation_temperature - TBuffervessel8[i]))
                heating_power_buffervessel = compressor_power*2
                elec_power = compressor_power*2/COP
        #-------------------------

        # update q_vector
        q_vector[0, i] = q_vector[0, i] + Qinst

        ts = [t[i], t[i+1]]
        result = solve_ivp(model_radiator_m, ts, y0,
                        method='RK45', args=inputs,
                        first_step=control_interval)

        inputs_buffervessel = (condensation_temperature, return_temp, mdots, mdotd)
        result_buffervessel = solve_ivp(sb.model_stratified_buffervessel, ts, y0buffervessel, method='RK45', args=inputs_buffervessel,
                        first_step=control_interval)

        # Write data into the containers
        Tair[i+1] = result.y[0, -1]
        Twall[i+1] = result.y[1, -1]
        Treturn[i] = return_temp
        Power[i] = Qinst
        Buffervessel_setpoint[i] = buffer_setpoint
        Power_buffervessel[i] = heating_power_buffervessel
        electrical_power[i] = elec_power
        TBuffervessel1[i+1] = result_buffervessel.y[0, -1]
        TBuffervessel8[i+1] = result_buffervessel.y[7, -1]

        #Use last result as a initial value for the ext step
        y0 = result.y[:, -1]
        y0buffervessel = result_buffervessel.y[:, -1]

    return t, Tair, Twall, Power, Treturn,  TBuffervessel1, TBuffervessel8, Power_buffervessel, electrical_power, Buffervessel_setpoint

