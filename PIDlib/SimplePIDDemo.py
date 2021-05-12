
"""
Example:
    See: https://www.fatalerrors.org/a/python-implementation-of-pid-control-based-on-simple_pid-library.html

    from simple_pid import PID

    pid = PID(1, 0.1, 0.05, setpoint=1)

    # assume we have a system we want to control in controlled_system
    v = controlled_system.update(0)

    while True:
        # compute new ouput from the PID according to the systems current value
        control = pid(v)

        # feed the PID output to the system and get its current value
        v = controlled_system.update(control)
"""
import time
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from simple_pid import PID


class Heater:
    def __init__(self):
        self.temp = 25

    def update(self, power, dt):
        if power > 0:
            # Variation of room temperature with power and time variable dt during heating
            self.temp += 2 * power * dt
        # Indicates heat loss in a room
        self.temp -= 0.5 * dt
        return self.temp


if __name__ == '__main__':
    # Write the created model into the main function
    heater = Heater()
    temp = heater.temp
    # Set three parameters of PID and limit output
    pid = PID(2, 2, 0.1, setpoint=temp)
    pid.output_limits = (0, None)
    # Used to set time parameters
    start_time = time.time()
    last_time = start_time
    # Visualize Output Results
    setpoint, y, x = [], [], []
    # Set System Runtime
    while time.time() - start_time < 5:

        # Setting the time variable dt
        current_time = time.time()
        dt = (current_time - last_time)

        # The variable temp is used as the output in the whole system,
        # and the difference between the variable temp and the ideal value
        # is used as the input in the feedback loop to adjust the change of the variable power.
        power = pid(temp)
        temp = heater.update(power, dt)

        # Visualize Output Results
        x += [current_time - start_time]
        y += [temp]
        setpoint += [pid.setpoint]
        # Used for initial value assignment of variable temp
        if current_time - start_time > 0:
            pid.setpoint = 100

        last_time = current_time

    # Visualization of Output Results
    plt.plot(x, setpoint, label='target')
    plt.plot(x, y, label='PID')
    plt.xlabel('time')
    plt.ylabel('temperature')
    plt.legend()
    plt.show()