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
    local_q_vector = np.zeros((3,1))
    local_q_vector[0,0] = q_vector[0,index]
    local_q_vector[1,0] = q_vector[1,index]
    local_q_vector[2,0] = q_vector[2,index]

    # Conversion of 1D array to a 2D array
    # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
    x = np.array(x)[np.newaxis]

    dTdt = (-cond_mat @ x.T) + local_q_vector
    dTdt = np.dot(cap_mat_inv, dTdt)

    return dTdt.flatten().tolist()

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
    Tradiator0 = 40

    y0 = [Tair0, Twall0, Tradiator0]

    t = time_sim           # Define Simulation time with sampling time
    Tair = np.ones(len(t)) * Tair0
    Twall = np.ones(len(t)) * Twall0
    Tradiator = np.ones(len(t)) * Tradiator0

    # Controller initialization
    # heatingPID = PID(Kp=5000, Ki=0, Kd=0, beta=1, MVrange=(0, 12000), DirectAction=False)
    # heating = 0
    #kp = control_parameters[0]
    #ki = control_parameters[1]
    #kd = control_parameters[2]

    #pid = PID(kp, ki, kd, t[0])

    #pid.SetPoint=17.0
    #pid.setSampleTime(0)
    #pid.setBounds(0, 12000)
    #pid.setWindup(12000/control_interval)

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
        #pid.SetPoint = SP_T[i]
        #pid.update(Tair[i], t[i])
        #q_vector[2, i] = pid.output

        # Simple PID controller
        # Qinst = (SP_T[i] - Tair[i]) * kp
        # Qinst = np.clip(Qinst, 0, 12000)
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

        # update q_vector
        q_vector[2, i] = p_hp*1000

        ts = [t[i], t[i+1]]
        result = solve_ivp(model_radiator_m, ts, y0,
                        method='RK45', args=inputs,
                        first_step=control_interval)

        Tair[i+1] = result.y[0, -1]
        Twall[i+1] = result.y[1, -1]
        Tradiator[i+1] = result.y[2, -1]

        y0 = result.y[:, -1]


    return t, Tair, Twall, Tradiator, q_vector[2,:]/1000, water_temp, cop_hp

